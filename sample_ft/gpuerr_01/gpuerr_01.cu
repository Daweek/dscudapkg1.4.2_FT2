#include <stdio.h>
#include <unistd.h>

#define MAX_NGPU (16)

#define MAX_TILES (1024)
#define NUM_TILES (256)
#define TILE_SIZE (1024 * 1024) 

int Ngpu;

__global__
void gpuerr( float *d_tiles, int num_tile, int tile_size ) {
    
}
#if 0
//
//
//
void initTiles( int num_gpu, float **h_tiles, int tile_size ) {
    for (int i=0; i<tile_size; i++) {
	*h_tiles[NUM_TILES*d + i] = (float)i + ((float)num_gpu * NUM_TILES);
    }
}
#endif
//
//
//
void sendTiles( float *h_tiles, float *d_tiles, int num_tiles ) {
    cudaError_t cuerr;

    for (int i=0; i<num_tiles; i++) {
	cuerr = cudaMemcpy( d_tiles, h_tiles, sizeof(float)*TILE_SIZE, cudaMemcpyHostToDevice );
	if (cuerr != cudaSuccess) {
	    fprintf( stderr, "cudaMemcpy() failed.\n");
	    exit(1);
	}
    }
}
#if 0
//
//
//
void recvTiles( int Ngpu, float *h_tiles, float *d_tiles, int num_tiles ) {
    cudaError_t cuerr;

    cuerr = cudaMemcpy(, cudaMemcpyDeviceToHost );
}
#endif

//
//
//
int main( int argc, char *argv[] )
{
    float *d_tiles[MAX_TILES * MAX_NGPU];
    float *h_tiles[MAX_TILES * MAX_NGPU];

    float *h_val;
    int i, j, k, d;
    cudaError_t cuerr;

    Ngpu = 1;

    // malloc@DEV
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
	}
    }
    // malloc@HOST
    for (d=0; d<Ngpu; d++) {
	for (i=0; i<NUM_TILES; i++) {
	    h_tiles[NUM_TILES*d + i] = (float *)malloc(sizeof(float)*TILE_SIZE);
	    if (h_tiles[NUM_TILES*d + i] == NULL) {
		fprintf(stderr, "malloc() failed.\n");
		exit(1);
	    }
	}
    }

    for (d=0; d<Ngpu; d++) {
	h_val = h_tiles[NUM_TILES * d];
	for (i=0; i<TILE_SIZE; i++) {
	    h_val[i] = (float)i;
	}
    }

#if 0
    for (d=0; d<Ngpu; d++) {
	h_val = h_tiles[NUM_TILES * d];
	for (i=0; i<TILE_SIZE; i++) {
	    printf("h_val[%d] = %f\n", i, h_val[i]);
	}
    }
#endif


    for (d=0; d<Ngpu; d++) {
	cuerr = cudaSetDevice(d);
	if (cuerr!=cudaSuccess) {
	    fprintf( stderr, "cudaSetDevice() failed.\n");
	    exit(1);
	}
	sendTiles( h_tiles, d_tiles, NUM_TILES);
    }
    
#if 0    
    //GPU kernel call
    for (d=0; d<Ngpu; d++) {
	cuerr = cudaSetDevice(d);
	if (cuerr!=cudaSuccess) {
	    fprintf( stderr, "cudaSetDevice() failed.\n");
	    exit(1);
	}
	gpuerr <<<NUM_TILES, 1024>>> ( d_tiles[NUM_TILES*d], NUM_TILES, TILE_SIZE );
    }

    for (d=0; d<Ngpu; d++) {
	cuerr = cudaSetDevice(d);
	if (cuerr!=cudaSuccess) {
	    fprintf( stderr, "cudaSetDevice() failed.\n");
	    exit(1);
	}
	recvTiles( h_tiles, d_tiles, NUM_TILES);
    }
#endif
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
