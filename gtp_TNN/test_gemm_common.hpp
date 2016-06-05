#pragma once

#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "cuda_util.hpp"

#define FMULS_GEMM( m, n, k ) ( ( m ) * ( n ) * ( k ) )
#define FADDS_GEMM( m, n, k ) ( ( m ) * ( n ) * ( k ) )
#define FLOPS_GEMM( m, n, k ) ( FMULS_GEMM ( ( double ) ( m ), ( double ) ( n ), ( double ) ( k ) ) + FADDS_GEMM ( ( double ) ( m ), ( double ) ( n ), ( double ) ( k ) ) )

#define ERR(fmt,...) fprintf ( stderr, fmt, ## __VA_ARGS__ )

template < typename real_t >
real_t * init_global_matrix ( int ld, int c ){
	void * p;
	SAFE_CALL ( cudaMalloc ( &p,       sizeof ( real_t ) * ld * c ) );
	SAFE_CALL ( cudaMemset (  p, 0x00, sizeof ( real_t ) * ld * c ) );
	return ( real_t * ) p;
}

inline
void free_global_matrix ( void *p ) {
	SAFE_CALL ( cudaFree ( p ) );
}

template < typename real_t >
real_t * alloc_pinned_matrix ( int ld, int c ){
	void * p;
	SAFE_CALL ( cudaHostAlloc ( &p, sizeof ( real_t ) * ld * c, cudaHostAllocDefault ) );
	return ( real_t * ) p;
}

inline
void free_pinned_matrix ( void *p ) {
	SAFE_CALL ( cudaFreeHost ( p ) );
}

template < typename real_t >
void randomize_global_matrix ( real_t * p, int r, int ld, int c ) {
	srand48 ( time ( NULL ) );
	real_t * h = alloc_pinned_matrix < real_t > ( r, c );
	for ( int i = 0; i < r * c; h[i++] = ( real_t ) drand48 ( ) ) { }
	SAFE_CALL ( cudaMemcpy2D (
		p, ld * sizeof ( real_t ),
		h,  r * sizeof ( real_t ),
		r * sizeof( real_t ), c,
		cudaMemcpyHostToDevice
	) );
	free_pinned_matrix ( h );
}
