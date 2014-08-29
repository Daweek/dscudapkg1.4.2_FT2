//                             -*- Mode: C++ -*-
// Filename         : gpu_stress01.cu
// Description      : GPU device stress test #01.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-08-22 09:15:27
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TILE_SIZE  (64 * 1024 * 1024) //64 MiB
#define TILE_COUNT (16)               //16 page

//#define GPU_PRINT

extern "C" __global__
void gpuStress( int t, float **table ) {
    float *p_tile = table[ blockIdx.x ];
    int i,k,m;

#ifdef GPU_PRINT
    if ( threadIdx.x == 0 ) {
	printf("t=%d:%d:%p\n", t, blockIdx.x, p_tile );
    }
#endif

    for ( i=threadIdx.x; i < TILE_SIZE/sizeof(float); i+=blockDim.x) {
	p_tile[i] = sin( p_tile[i] * p_tile[i] * p_tile[i] );
	p_tile[i] = sqrt( p_tile[i] * p_tile[i] * p_tile[i] );
	p_tile[i] = cos( p_tile[i] * p_tile[i] * p_tile[i] );
	p_tile[i] = sqrt( p_tile[i] * p_tile[i] * p_tile[i] );
    }
    
    return;
}

int main() {
    const int t_max = 10;
    float **d_tbl;
    float *d_tile[ TILE_COUNT ];
    float *h_tile[ TILE_COUNT ];
    cudaError_t cuerr;

    cudaSetDevice(0);

    /*
     * cudaMalloc
     */
    for ( int i=0; i<TILE_COUNT; i++ ) {
	h_tile[i] = (float *)malloc( TILE_SIZE );
	if ( h_tile[i] == NULL ) {
	    fprintf( stderr, "malloc() failed.\n" );
	    exit( EXIT_FAILURE );
	}
	cuerr = cudaMalloc( &d_tile[i], TILE_SIZE );
	if ( cuerr != cudaSuccess ) {
	    fprintf( stderr, "cudaMalloc( &d_tile[%d] ) failed.\n", i );
	    exit( EXIT_FAILURE );
	}
    }
    
    cuerr = cudaMalloc( &d_tbl, sizeof(float *)*TILE_COUNT );
    if ( cuerr != cudaSuccess ) {
	fprintf( stderr, "cudaMalloc( &d_tbl) failed.\n" );
	exit( EXIT_FAILURE );
    }

    /*
     * init data.
     */
    float *pf;
    for (int i=0; i<TILE_COUNT; i++) {
	pf = h_tile[i];
	for (int k=0; k<(TILE_SIZE/sizeof(float)); k++ ) {
	    *(pf+k) = (float)i + (float)k;
//	    printf("i=%d, k=%d\n", i, k);
	}
    }
    fprintf(stderr, "init data ends.\n");

    /*
     * cudaMemcpy
     */
    for (int i=0; i<TILE_COUNT; i++) {
	cuerr = cudaMemcpy( d_tile[i], h_tile[i], TILE_SIZE, cudaMemcpyHostToDevice );
	if ( cuerr != cudaSuccess ) {
	    fprintf( stderr, "i=%d, cudaMemcpy(HtoD) failed.\n", i );
	    exit ( EXIT_FAILURE );
	}
    }
    cuerr = cudaMemcpy( d_tbl, d_tile, sizeof(float *)*TILE_COUNT, cudaMemcpyHostToDevice );
    if ( cuerr != cudaSuccess ) {
	fprintf( stderr, "cudaMemcpy( d_tbl, d_tile, HtoD) failed.\n" );
	exit ( EXIT_FAILURE );
    }
    
    fprintf(stderr, "cudaMemcpyH2D ends.\n");


    for (int t=0; t<t_max; t++ ){
#if 1
	/*
	 * 
	 */
	gpuStress <<< TILE_COUNT, 256 >>> ( t, d_tbl );
#endif
	/*
	 * cudaMemcpy
	 */
	for (int i=0; i<TILE_COUNT; i++) {
	    cuerr = cudaMemcpy( h_tile[i], d_tile[i], TILE_SIZE, cudaMemcpyDeviceToHost );
	    if ( cuerr != cudaSuccess ) {
		fprintf( stderr, "cudaMemcpy() failed.\n" );
		exit ( EXIT_FAILURE );
	    }
	}
    }
    sleep(1);

    /*
     * cudaFree
     */
    for ( int i=0; i<TILE_COUNT; i++ ) {
	cuerr = cudaFree( d_tile[i] );
	if ( cuerr != cudaSuccess ) {
	    fprintf( stderr, "cudaFree( d_tile[%d] ) failed.\n", i );
	    exit( EXIT_FAILURE );
	}
    }
    cuerr = cudaFree( d_tbl );
    if ( cuerr != cudaSuccess ) {
	fprintf( stderr, "cudaFree( d_tbl ) failed.\n" );
	exit( EXIT_FAILURE );
    }

    puts("completed.\n");
    return EXIT_SUCCESS;
}
