#include <cublas_v2.h>
#include <sys/time.h>
#include <queue>

#include "cuda_util.hpp"
#include "gpu_thread_pool.hpp"
#include "dgemm_chayoneko_magma141.cu"

using namespace std;

typedef double real_t;

struct n_t {
	n_t ( int idx, float v ) : idx ( idx ), v ( v ) { }
	bool operator < ( n_t const & o ) const { return v < o.v; }
	int idx;
	float v;
};

char const * TNN_type = "dTNN_chayoneko_magma141";
double run_TNN ( real_t * A, real_t * C, int n, int d, int t, int rep, int dev ) {
	SAFE_CALL ( cudaSetDevice ( dev ) );

	size_t offsetA = 0;
	size_t offsetB = 0;

	cudaError_t errt;
	errt = cudaBindTexture ( &offsetA, tex_ref_A, ( int2 * ) A, n * d * sizeof ( double ) );
	if ( errt != cudaSuccess || offsetA ) {
		printf ( "can not bind A to texture\n" );
		exit ( 1 );
	}
	errt = cudaBindTexture ( &offsetB, tex_ref_B, ( int2 * ) A, n * d * sizeof ( double ) );
	if ( errt != cudaSuccess || offsetB ) {
		printf ( "can not bind B to texture\n" );
		exit ( 1 );
	}

	struct common_data_t data;
	data.A       = A;
	data.B       = A;
	data.C       = C;
	data.alpha   = ( real_t ) -2.0;
	data.beta    = ( real_t ) 1.0;
	data.m       = n;
	data.n       = n;
	data.k       = d;
	data.lda     = n;
	data.ldb     = n;
	data.ldc     = n;
	data.offsetA = 0;
	data.offsetB = 0;

	task_t task;

	dim3 dim_block ( 16, 16 );
	int nby = CEIL_DIV ( n, 64 );

	priority_queue < n_t > * pq = new priority_queue < n_t >[n];

	cublasHandle_t handle;
	SAFE_CALL ( cublasCreate ( &handle ) );
	real_t *norm;
	SAFE_CALL( cudaHostAlloc ( &norm, sizeof( real_t ) * n, cudaHostAllocDefault ) );
	for ( int i = 0; i < n; ++i )
		SAFE_CALL( cublasDnrm2( handle, n, A + i, n, norm + i ) );
	for ( int i = 0; i < n; ++i ) for( int j = 0; j <= n; ++j )
		C[ i * n + j] = norm[i] + norm[j];

	SAFE_CALL ( cublasDestroy ( handle ) );

	gpu_thread_pool_t < task_t, common_data_t, 1 > gtp ( data, 2, dim_block, 1, dev );
	gtp.run ( );

	struct timeval tb, te;
	gettimeofday ( &tb, NULL );
	for ( int r = 0; r < rep; ++r ) {
		int cnt = 0;
		for ( int j = 0; j < nby; ++j ) for ( int i = 0; i <= j; ++i ) {
			task.bx = i;
			task.by = j;
			gtp.push_task ( task );
			++cnt;
		}
		fprintf ( stderr, "%d tasks pushed.\n", cnt );
		while ( cnt-- ) {
			gtp.pop_finish ( task );
			int jb  = task.by * 64;
			int je = jb + 64;
			int ib  = task.bx * 64;
			int ie = ib + 64;
			if ( task.bx == task.by ) {
				for ( int j = jb; j < je; ++j ) for ( int i = ib; i < j; ++i ) {
					float v = C[j * n + i];
					pq[i].push ( n_t ( j, v ) );
					if ( pq[i].size ( ) > t ) { pq[i].pop ( ); }
					pq[j].push ( n_t ( i, v ) );
					if ( pq[j].size ( ) > t ) { pq[j].pop ( ); }
				}
			} else {
				for ( int j = jb; j < je; ++j ) for ( int i = ib; i < ie; ++i ) {
					float v = C[j * n + i];
					pq[i].push ( n_t ( j, v ) );
					if ( pq[i].size ( ) > t ) { pq[i].pop ( ); }
					pq[j].push ( n_t ( i, v ) );
					if ( pq[j].size ( ) > t ) { pq[j].pop ( ); }
				}
			}
		}
	}
	gettimeofday ( &te, NULL );
	gtp.set_exiting_flag ( );
	gtp.synchronize ( );

	delete [] pq;
	SAFE_CALL( cudaFreeHost( norm ) );
	cudaUnbindTexture ( tex_ref_B );
	cudaUnbindTexture ( tex_ref_A );
	return te.tv_sec - tb.tv_sec + ( te.tv_usec - tb.tv_usec ) * 1E-6;
}
