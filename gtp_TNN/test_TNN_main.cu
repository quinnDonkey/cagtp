#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#include "cuda_util.hpp"
#include "test_gemm_common.hpp"

#ifdef USING_DOUBLE
typedef double real_t;
#define CUBLAS_GEMM	cublasDgemm
#elif defined USING_FLOAT
typedef float real_t;
#define CUBLAS_GEMM	cublasSgemm
#else
#error "One of USING_DOUBLE and USING_FLOAT must be defined to compile this file."
#endif

using namespace std;

extern
char const * TNN_type;

extern
double run_TNN ( real_t * A, real_t * C, int n, int d, int t, int rep, int dev );

int main ( int argc, char * argv[] ) {
	int n, d, t;
	int rep;
	int dev;

	if ( 6 != argc ) {
		ERR ( "Usage: %s <n> <d> <t> <rep> <dev>\n", argv[0] );
		exit ( 1 );
	}

	  n = atoi ( argv[1] );
	  d = atoi ( argv[2] );
	  t = atoi ( argv[3] );
	rep = atoi ( argv[4] );
	dev = atoi ( argv[5] );

	SAFE_CALL ( cudaSetDevice ( dev ) );

	real_t * A;
	real_t * C;
	A =  init_global_matrix < real_t > ( n, d );
	C = alloc_pinned_matrix < real_t > ( n, n );

	randomize_global_matrix ( A, n, n, d );
	memset ( C, 0x00, n * n * sizeof ( real_t ) );

	ERR ( "==================== TNN test ====================\n" );
	ERR ( "type: %s\n",                                 TNN_type );
	ERR ( "n: %d\n",                                           n );
	ERR ( "d: %d\n",                                           d );
	ERR ( "t: %d\n",                                           t );
	ERR ( "rep: %d\n",                                       rep );
	ERR ( "dev: %d\n",                                       dev );

	double _t = run_TNN ( A, C, n, d, t, rep, dev );
	ERR ( "time: %.6lf second\n", _t / rep );

	free_pinned_matrix ( C );
	free_global_matrix ( A );

	return 0;
}
