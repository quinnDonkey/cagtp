#pragma once

#include <list>
#include <pthread.h>
#include <tbb/concurrent_queue.h>

#define GTP_VERSION_MAJOR 0
#define GTP_VERSION_MINOR 9

#ifndef SAFE_CALL
template < typename status_t > inline
void check_call ( status_t error, char const * file, size_t line, char const * call ) {
	if ( 0 == error ) return;
	fprintf ( stderr, "%s failed in \"%s:%lu\": %d\n", call, file, line, ( int ) error );
	exit ( error );
}

template < > inline
void check_call < cudaError_t > ( cudaError_t error, char const * file, size_t line, char const * call ) {
	if ( cudaSuccess == error ) return;
	fprintf ( stderr, "%s failed in \"%s:%lu\": %s\n", call, file, line, cudaGetErrorString ( error ) );
	exit ( error );
}

template < > inline
void check_call < int > ( int error, char const * file, size_t line, char const * call ) {
	if ( EXIT_SUCCESS == error ) return;
	fprintf ( stderr, "%s failed in \"%s:%lu\": %s\n", call, file, line, strerror ( error ) );
	exit ( error );
}

#define SAFE_CALL(call) check_call ( ( call ), __FILE__, __LINE__, #call )
#endif // SAFE_CALL

enum task_slot_stat_t {
	IDLE,
	FINISHED,
	GPU_CONCERN, // chayoneko_kernel ( ) cares only following stats.
	READY,
	EXIT,
};

template < typename task_t, typename common_data_t, int type_id >
__device__ void do_task ( task_t const & task, common_data_t const & common_data ) {
	printf ( "You must specialize your do_task() function.\n" );
}

template < typename task_slot_t, typename common_data_t, int type_id >
__global__ void chayoneko_kernel ( char * task_slots, uint32_t task_slot_padded_size, common_data_t common_data ) {
	volatile task_slot_t * p = ( volatile task_slot_t * ) ( task_slots + task_slot_padded_size * blockIdx.x );
	typename task_slot_t::kernel_task_t kernel_task;

	while ( 1 ) {
		if ( 0 == threadIdx.x && 0 == threadIdx.y && 0 == threadIdx.z ) {
			while ( p->kernel_task.stat < GPU_CONCERN ) { }
		}
		__syncthreads ( );

		kernel_task = ( ( task_slot_t * ) p )->kernel_task;
		if ( EXIT == kernel_task.stat ) return;

		do_task < task_slot_t::user_task_t, common_data_t, type_id > ( kernel_task.task, common_data );
		__syncthreads ( );

		if ( 0 == threadIdx.x && 0 == threadIdx.y && 0 == threadIdx.z ) {
			p->kernel_task.stat = FINISHED;
		}
	}
}

template < typename task_t, typename common_data_t, int type_id >
class gpu_thread_pool_t {
public :
	struct input_task_t {
		input_task_t ( ) { }
		input_task_t ( task_t const & task, int output_index ) : task ( task ), output_index ( output_index ) { }
		task_t		task;
		int		output_index;
	};
	struct task_slot_t {
		task_slot_t ( ) { }
		typedef task_t user_task_t;
		struct kernel_task_t {
			__host__ __device__ kernel_task_t ( ) { }
			user_task_t task;
			task_slot_stat_t stat;
		} kernel_task;
		int output_index;
	};
private :
	size_t                                           num_active_blocks;
	size_t                                           task_slot_padded_size;
	char *                                           task_slots;
	tbb::concurrent_bounded_queue < input_task_t >   input_queues;
	tbb::concurrent_bounded_queue <       task_t > * finish_queues;
	size_t                                           num_finish_queues;
	pthread_t                                        gpu_scheduler_thread;
	volatile int                                     exiting;
	size_t *                                         task_count_for_block;
	common_data_t                                    common_data;
	dim3                                             dim_block;
	volatile size_t *                                unfinished_count;
	cudaStream_t                                     gtp_stream;
	int                                              device;

	static void * gpu_scheduler_fun ( void * args );
	bool try_pop_task ( input_task_t & input_task ) {
		return ( input_queues.try_pop ( input_task ) );
	}
public :
	gpu_thread_pool_t (
		common_data_t common_data,
		size_t blocks_per_sm,
		dim3 dim_block,
		size_t num_finish_queues = 1,
		int device = 0
	) :
		common_data ( common_data ),
		dim_block ( dim_block ),
		num_finish_queues ( num_finish_queues ),
		device ( device ) {

		SAFE_CALL ( cudaSetDevice ( device ) );

		cudaDeviceProp prop;
		SAFE_CALL ( cudaGetDeviceProperties ( &prop, device ) );
		num_active_blocks = blocks_per_sm * prop.multiProcessorCount;

		finish_queues = new tbb::concurrent_bounded_queue <       task_t > [num_finish_queues];

		// unfinished_count[-1] is used for counting tasks which need not to be put into finish_queue
		unfinished_count = new size_t[num_finish_queues + 1];
		for ( int i = 0; i < num_finish_queues + 1; ++i ) unfinished_count[i] = 0;
		++unfinished_count;

		// gtp_task structures in pinned memory
#define GTP_PINNED_GRANULARITY	128
		task_slot_padded_size = ( sizeof ( task_slot_t ) + GTP_PINNED_GRANULARITY - 1 ) / GTP_PINNED_GRANULARITY * GTP_PINNED_GRANULARITY;
		SAFE_CALL ( cudaMallocHost ( ( void ** ) &task_slots, task_slot_padded_size * num_active_blocks ) );
		for ( int i = 0; i < num_active_blocks; ++i ) {
			task_slot_t * p = ( struct task_slot_t * ) ( task_slots + task_slot_padded_size * i );
			p->kernel_task.stat = IDLE;
		}
#undef GTP_PINNED_GRANULARITY

		task_count_for_block = new size_t[num_active_blocks];
		for ( int i = 0; i < num_active_blocks; ++i ) {
			task_count_for_block[i] = 0;
		}

		SAFE_CALL ( cudaStreamCreate ( &gtp_stream ) );
	}

	void print_statistics ( FILE * f = stderr ) const {
		size_t total = 0;
		fprintf ( f, "========================================\n"               );
		fprintf ( f, "%-32s%8lu\n", "task size:",        sizeof ( task_t )      );
		fprintf ( f, "%-32s%8lu\n", "task slot size:",   sizeof ( task_slot_t ) );
		fprintf ( f, "%-32s%8lu\n", "task padded size:", task_slot_padded_size  );
		fprintf ( f, "%-32s%8lu\n", "number of blocks:", num_active_blocks      );
		fprintf ( f, "task count for each block:\n" );
		for ( int i = 0; i < num_active_blocks; ++i ) {
			fprintf ( f, "block %3d: %16lu\n", i, task_count_for_block[i] );
			total += task_count_for_block[i];
		}
		fprintf ( f, "%11s%16lu\n", "total: ", total );
	}

	void run ( void ) {
		exiting = 0;
		for ( int i = 0; i < num_active_blocks; ++i ) {
			task_slot_t * p = ( struct task_slot_t * ) ( task_slots + task_slot_padded_size * i );
			p->kernel_task.stat = IDLE;
		}
		SAFE_CALL ( pthread_create ( &gpu_scheduler_thread, NULL, gpu_scheduler_fun, this ) );
		chayoneko_kernel < task_slot_t, common_data_t, type_id > <<< num_active_blocks, dim_block, 0, gtp_stream >>> ( task_slots, task_slot_padded_size, common_data );
	}

	void set_exiting_flag ( void ) {
		exiting = 1;
	}

	void synchronize ( void ) {
		SAFE_CALL ( pthread_join ( gpu_scheduler_thread, NULL ) );
		SAFE_CALL ( cudaStreamSynchronize ( gtp_stream ) );
	}

	size_t get_num_active_blocks ( void ) const {
		return num_active_blocks;
	}

	~gpu_thread_pool_t ( void ) {
		SAFE_CALL ( cudaStreamDestroy ( gtp_stream ) );
		SAFE_CALL ( cudaFreeHost ( ( void * ) task_slots ) );
		delete [] task_count_for_block;
		delete [] ( unfinished_count - 1 );
	}

	void push_task ( task_t const & task, int output_index = 0, int priority = 0 ) {
// fprintf ( stderr, "gtp push_task <%d,%d,%d>\n", task.type, output_index, priority );
		input_queues.push ( input_task_t ( task, output_index ) );
		++unfinished_count[output_index];
	}

	void pop_finish ( task_t & task, int output_index = 0 ) {
		finish_queues[output_index].pop ( task );
		--unfinished_count[output_index];
	}

	bool try_pop_finish ( task_t & task, int output_index = 0 ) {
		if ( finish_queues[output_index].try_pop ( task ) ) {
			--unfinished_count[output_index];
			return true;
		} else {
			return false;
		}
	}

	size_t get_unfinished_count ( int output_index = 0 ) {
		return unfinished_count[output_index];
	}
};

template < typename task_t, typename common_data_t, int type_id >
void * gpu_thread_pool_t < task_t, common_data_t, type_id >::gpu_scheduler_fun ( void * args ) {
	gpu_thread_pool_t < task_t, common_data_t, type_id > * gtp = ( gpu_thread_pool_t * ) args;
	typedef gpu_thread_pool_t < task_t, common_data_t, type_id >::task_slot_t task_slot_t;
	for ( int i = 0; ; i = ( i + 1 ) % gtp->num_active_blocks ) {
		volatile task_slot_t * p = ( volatile task_slot_t * ) ( gtp->task_slots + gtp->task_slot_padded_size * i );
		if ( FINISHED == p->kernel_task.stat ) {
			task_slot_t task_slot = * ( task_slot_t * ) p;
// fprintf ( stderr, "gtp got finished task <%d,%d>\n", task_slot.task.type, task_slot.output_index );
			if ( -1 == task_slot.output_index ) {
				--gtp->unfinished_count[-1];
			} else {
				gtp->finish_queues[task_slot.output_index].push ( task_slot.kernel_task.task );
			}
			p->kernel_task.stat = IDLE;
		}
		if ( IDLE == p->kernel_task.stat ) {
			input_task_t input_task;
			if ( gtp->try_pop_task ( input_task ) ) {
// fprintf ( stderr, "gtp assign task <%d,%d> to block %d\n", task_slot.task.type, task_slot.output_index, i );
				++gtp->task_count_for_block[i];
				( ( task_slot_t * ) p )->output_index = input_task.output_index;
				( ( task_slot_t * ) p )->kernel_task.task = input_task.task;
				p->kernel_task.stat = READY;
			} else if ( gtp->exiting ) {
				break;
			}
		}
	}

	std::list < int > running_blocks;
	for ( int i = 0; i < gtp->num_active_blocks; ++i ) {
		running_blocks.push_back ( i );
	}

	for ( std::list < int >::iterator it = running_blocks.begin ( ); !running_blocks.empty ( ); ) {
		if ( running_blocks.end ( ) == it ) it = running_blocks.begin ( );
		volatile task_slot_t * p = ( volatile task_slot_t * ) ( gtp->task_slots + gtp->task_slot_padded_size * *it );
		if ( FINISHED == p->kernel_task.stat ) {
			task_slot_t task_slot = * ( task_slot_t * ) p;
			if ( -1 == task_slot.output_index ) {
				--gtp->unfinished_count[-1];
			} else {
				gtp->finish_queues[task_slot.output_index].push ( task_slot.kernel_task.task );
			}
			p->kernel_task.stat = IDLE;
		}
		if ( IDLE == p->kernel_task.stat ) {
			p->kernel_task.stat = EXIT;
			running_blocks.erase ( it++ );
			continue;
		}
		++it;
	}
	fprintf ( stderr, "GTP scheduler exiting...\n" );

	return NULL;
}
