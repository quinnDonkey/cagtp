#pragma once

#include <stdio.h>
#include <stdlib.h>

#define CEIL_DIV( a, b ) ( ( ( a ) + ( b ) - 1 ) / ( b ) )

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
#endif

template < typename real_t >
inline void print_host_matrix ( real_t * h, int r, int c, int ld, char const * title = NULL ) {
	if ( title ) fprintf ( stderr, "==================== %s ====================\n", title );
	for ( int i = 0; i < r; ++i ) {
		fprintf ( stderr, "%d:", i );
		for ( int j = 0; j < c; ++j ) {
			fprintf ( stderr, " %lg", ( double ) h[j * ld + i] );
		}
		fprintf ( stderr, "\n" );
	}
}

template < typename real_t >
inline void print_gpu_matrix ( real_t * d, int r, int c, int ld, char const * title = NULL ) {
	real_t * h;
	SAFE_CALL ( cudaMallocHost ( ( void ** ) &h, r * c * sizeof ( real_t ) ) );
	SAFE_CALL ( cudaMemcpy2D ( h, r * sizeof ( real_t ), d, ld * sizeof ( real_t ), r * sizeof ( real_t ), c, cudaMemcpyDeviceToHost ) );
	print_host_matrix ( h, r, c, r, title );
	SAFE_CALL ( cudaFreeHost ( h ) );
}
