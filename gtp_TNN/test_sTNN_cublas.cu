#include <cublas_v2.h>
#include <sys/time.h>
#include <queue>

#include "cuda_util.hpp"

using namespace std;

typedef float real_t;

struct n_t {
	n_t ( int idx, float v ) : idx ( idx ), v ( v ) { }
	bool operator < ( n_t const & o ) const { return v < o.v; }
	int idx;
	float v;
};

char const * TNN_type = "sTNN_cublas";
double run_TNN ( real_t * A, real_t * C, int n, int d, int t, int rep, int dev ) {
	SAFE_CALL ( cudaSetDevice ( dev ) );

	cublasHandle_t handle;
	SAFE_CALL ( cublasCreate ( &handle ) );

	int lda = n;
	int ldc = n;
	real_t alpha = ( real_t ) -2.0;
	real_t  beta = ( real_t ) 1.0;

	priority_queue < n_t > * pq = new priority_queue < n_t >[n];

	real_t *norm;
	SAFE_CALL( cudaHostAlloc ( &norm, sizeof( real_t ) * n, cudaHostAllocDefault ) );

	struct timeval tb, te;
	gettimeofday ( &tb, NULL );

	for ( int i = 0; i < n; ++i )
		SAFE_CALL( cublasSnrm2( handle, n, A + i, n, norm + i ) );

	for ( int i = 0; i < n; ++i ) for( int j = 0; j <= n; ++j )
		C[ i * n + j] = norm[i] + norm[j];

	for ( int r = 0; r < rep; ++r ) {
		SAFE_CALL ( cublasSsyrk (
			handle,
			CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
			n, d,
			&alpha,
			A, lda,
			&beta,
			C, ldc
		) );
		SAFE_CALL ( cudaDeviceSynchronize ( ) );
		for ( int j = 0; j < n; ++j ) {
			for ( int i = 0; i < j; ++i ) {
				float v = ( float ) C[j + n + i];
				pq[i].push ( n_t ( j, v ) );
				if ( pq[i].size ( ) > t ) { pq[i].pop ( ); }
				pq[j].push ( n_t ( i, v ) );
				if ( pq[j].size ( ) > t ) { pq[j].pop ( ); }
			}
		}
	}
	gettimeofday ( &te, NULL );

	delete [] pq;
	SAFE_CALL( cudaFreeHost( norm ) );
	SAFE_CALL ( cublasDestroy ( handle ) );
	return te.tv_sec - tb.tv_sec + ( te.tv_usec - tb.tv_usec ) * 1E-6;
}
