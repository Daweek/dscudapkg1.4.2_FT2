#include <stdio.h>
#include <unistd.h>

#define MAX_NGPU 16

#define MAX_TILES 64
#define NUM_TILES 16

#define TILE_SIZE (16 * 1024 * 1024) 

int Ngpu;

__global__
void gpuerr( float *d_tiles, int num_tile, int tile_size ) {
    
}

int main( int argc, char *argv[] ) {
    float *d_tiles[MAX_NGPU * MAX_TILES];
    float *h_tiles[MAX_NGPU * MAX_TILES];
    int i, j, k, d;
    cudaError_t cuerr;

    Ngpu = 4;

    // malloc
    for (d=0; d<Ngpu; d++) {
	cuerr = cudaSetDevice(d);
	if (cuerr!=cudaSuccess) {
	    fprintf( stderr, "cudaSetDevice() failed.\n");
	    exit(1);
	}
	for (i=0; i<NUM_TILES; i++) {
	    cuerr = cudaMalloc( &d_tiles[NUM_TILES*d + i], sizeof(float)*TILE_SIZE);
	    if (cuerr!=cudaSuccess) {
		fprintf( stderr, "cudaMalloc() failed.\n");
		exit(1);
	    } else {
		printf(" d_tiles[%3d] = %p\n", NUM_TILES*d+i, d_tiles[i]);
	    }

	    h_tiles[NUM_TILES*d + i] = (float *)malloc(sizeof(float)*TILE_SIZE);
	    if (h_tiles[NUM_TILES*d + i]==NULL) {
		fprintf(stderr, "malloc() failed.\n");
		exit(1);
	    }
	}
    }

    sleep(10);

    // free
    for (d=0; d<Ngpu; d++) {
	cuerr = cudaSetDevice(d);
	if (cuerr!=cudaSuccess) {
	    fprintf( stderr, "cudaSetDevice() failed.\n");
	    exit(1);
	}
	for (i=0; i<16; i++) {
	    cuerr = cudaFree( d_tiles[NUM_TILES*d + i] );
	    if (cuerr!=cudaSuccess) {
		fprintf( stderr, "%d:cudaFree(%d) failed.\n", d, i);
		exit(1);
	    }
	    free( h_tiles[NUM_TILES*d + i] );
	}

    }

    printf("Program completed.\n");
    return 0;
}
