#include <stdio.h>
#include <unistd.h>

#define MAX_NGPU 16
#define MAX_TILES 64
#define TILE_SIZE (16 * 1000 * 1000) 

int Ngpu;

__global__
void gpuerr() {
    
}

int main( int argc, char *argv[] ) {
    float *d_tiles[MAX_TILES * MAX_NGPU];
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
	for (i=0; i<16; i++) {
	    cuerr = cudaMalloc( &d_tiles[i], sizeof(float)*TILE_SIZE);
	    if (cuerr!=cudaSuccess) {
		fprintf( stderr, "cudaMalloc() failed.\n");
		exit(1);
	    } else {
		printf(" d_tiles[%3d] = %p\n", i, d_tiles[i]);
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
	    cuerr = cudaFree( d_tiles[i] );
	    if (cuerr!=cudaSuccess) {
		fprintf( stderr, "%d:cudaFree(%d) failed.\n", d, i);
		exit(1);
	    } 
	}
    }


    return 0;
}
