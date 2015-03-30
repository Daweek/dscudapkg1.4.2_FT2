#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#define MAXDEV 4
static const double MEGA  = 1e6;
static const double MICRO = 1e-6;

cudaError_t cudaMemcpyToAlldev(int ndev, void **dst, const void *src, size_t count, enum cudaMemcpyKind kind);

static void
get_cputime(double *nowp, double *deltap)
{
    struct timeval t;
    double now0;
    gettimeofday(&t, NULL);
    now0 = t.tv_sec + t.tv_usec/1000000.0;
    *deltap = now0 - *nowp;
    *nowp   = now0;
}

static void
bcastperf(int argc, char **argv)
{
    int maxsize = 1024 * 1024 * 10;
    int i, j;
    size_t size;
    double sized;
    double now = 0.0, dt = 0.0;
    double ratio = 2.5;
    double nloop = 2e8;
    char *src = (char *)malloc(sizeof(char) * maxsize);
    char *dst[MAXDEV];
    int ndev;
    static int nthread = 0;

    printf("# %d device%s found.\n", ndev, ndev > 1 ? "s" : "");

    for (i = 0; i < ndev; i++) {
        cudaSetDevice(i);
        cudaMalloc((void**) &dst[i], sizeof(char) * maxsize);
    }
    printf("\n#\n# cudaMemcpy (HostToDevice)\n");
    printf("# broadcast to %d servers.\n#\n", ndev);

    for (sized = 4096; sized < maxsize; sized *= ratio) {
        //    for ( nloop = 2e8, sized = 4096 * 1; ; ) { // !!!
        size = (size_t)sized;

	get_cputime(&now, &dt);
	for (j = 0; j < nloop/size; j++) {
#if 1
#pragma omp parallel for
	    for (i = 0; i < ndev; i++) {
#ifdef _OPENMP
                if (nthread == 0) {
                    nthread = omp_get_num_threads();
                    fprintf(stderr, "nthread:%d\n", nthread);
                }
#endif // _OPENMP
  	        cudaSetDevice(i);
                cudaMemcpy(dst[i], src, size, cudaMemcpyHostToDevice);
	    }
#else
            cudaMemcpyToAlldev(ndev, (void **)dst, src, size, cudaMemcpyHostToDevice);
#endif
	}
        //cudaDeviceSynchronize();
	get_cputime(&now, &dt);
	printf("%d byte    %f sec    %f MB/s\n",
               size, dt, nloop/MEGA/dt);
	fflush(stdout);
    }
}

/*
 *
 */
static
void sendperf( int argc, char **argv, size_t minsize, size_t maxsize, size_t nloop)
{
    int i, j;
    size_t size;
    double now = 0.0, dt = 0.0;
    int  ratio = 2;
    char *src[MAXDEV];
    char *dst[MAXDEV];
    int ndev;

    ndev = 1; // !!!

    printf("# %d device%s found.\n", ndev, ndev > 1 ? "s" : "");
    for (i = 0; i < ndev; i++) {
        cudaSetDevice(i);
        cudaMalloc((void**) &dst[i], sizeof(char) * maxsize);
	src[i] = (char *)malloc(sizeof(char) * maxsize);
    }
    
    printf("#\n");
    printf("# cudaMemcpy(HostToDevice)\n");
    printf("#   - Total size of transfered is %6.1f MByte.\n", nloop/1e6);
    printf("#\n");

    printf("\n#\n# cudaMemcpy (HostToDevice)\n#\n");

#if 1
    for (size = minsize; size < maxsize; size *= ratio) {
	size_t packet_count = nloop/size;
	get_cputime(&now, &dt);
	for (j = 0; j < packet_count; j++) {
	    for (i = 0; i < ndev; i++) {
		cudaSetDevice(i);
                cudaMemcpy(dst[i], src[i], size, cudaMemcpyHostToDevice);
	    }
	}
        cudaDeviceSynchronize();
	get_cputime(&now, &dt);

#if 0 // with estimated RPC overhead.
	double throughput = 1700.0; // MB/s
	double latency    = 60.0; // us
	double ibsec = nloop / (throughput * MEGA) + latency * MICRO * nloop / size;
	printf("%d byte    %f sec    %f MB/s    %f ib_sec  %f MB/s\n",
	       size, lt, nloop/MEGA/lt, ibsec, nloop/MEGA/(lt + ibsec));
#else
	  printf("%12d Byte  %12d times %12.6f sec  %12.6f MB/s\n",
		 size, packet_count, dt, (size * packet_count)/MEGA/dt);
#endif
	fflush(stdout);
    }

#else
    size = 40;
    for (i = 0; i < ndev; i++) {
	for (j = 0; j < size; j++) {
	    src[i][j] = j;
	}
    }
    for (i = 0; i < ndev; i++) {
	cudaSetDevice(i);
	cudaMemcpy(dst[i], src[i], size, cudaMemcpyHostToDevice);
    }
#endif
    
#if 0
    for (i = 0; i < ndev; i++) {
	cudaSetDevice(i);
	cudaFree(dst[i]);
	free(src[i]);
    }
#endif
}
static
void sendperf_analyze( int argc, char **argv, size_t minsize, size_t maxsize, size_t nloop)
{
    int i, j;
    size_t size;
    double now = 0.0, dt = 0.0;
    double sta = 0.0, sta_dt = 0.0;
    int  ratio = 2;
    char *src[MAXDEV];
    char *dst[MAXDEV];
    int ndev;

    ndev = 1; // !!!

    printf("# %d device%s found.\n", ndev, ndev > 1 ? "s" : "");
    for (i = 0; i < ndev; i++) {
        cudaSetDevice(i);
        cudaMalloc((void**) &dst[i], sizeof(char) * maxsize);
	src[i] = (char *)malloc(sizeof(char) * maxsize);
    }
    
    printf("#\n");
    printf("# cudaMemcpy(HostToDevice)\n");
    printf("#   - Total size of transfered is %6.1f MByte.\n", nloop/1e6);
    printf("#\n");

    printf("\n#\n# cudaMemcpy (HostToDevice)\n#\n");

    for (size = minsize; size < maxsize; size *= ratio) {
	get_cputime(&now, &dt);
	for (j = 0; j < nloop/size; j++) {
	    for (i = 0; i < ndev; i++) {
		get_cputime(&sta, &sta_dt);
  	        cudaSetDevice(i);
                cudaMemcpy(dst[i], src[i], size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		get_cputime(&sta, &sta_dt);
		printf("%12d Byte %f sec %f MB/s\n", size, sta_dt, size / MEGA / sta_dt);
	    }
	}
        cudaDeviceSynchronize();
	get_cputime(&now, &dt);
	
	printf("%12d Byte    %f sec    %f MB/s\n",
	       size, dt, nloop/MEGA/dt);
	fflush(stdout);
    }
}

static
void recvperf( int argc, char **argv,
	       size_t minsize, size_t maxsize, size_t nloop) {
    int i, j;
    size_t size;
    double now = 0.0, dt = 0.0;
    size_t ratio = 2;
    char *src[MAXDEV];
    char *dst[MAXDEV];
    int ndev;

    ndev = 1; // !!!

    printf("# %d device%s found.\n", ndev, ndev > 1 ? "s" : "");
    for (i = 0; i < ndev; i++) {
        cudaSetDevice(i);
        cudaMalloc((void**) &src[i], sizeof(char) * maxsize);
	dst[i] = (char *)malloc(sizeof(char) * maxsize);
    }
    printf("#\n");
    printf("# cudaMemcpy(DeviceToHost)\n");
    printf("#   - Total size of transfered is %6.1f MByte.\n", (double)nloop/1e6);
    printf("#\n");

    for (size = minsize; size < maxsize; size *= ratio) {
	size_t packet_count = nloop / size;
	get_cputime(&now, &dt);
	for (j = 0; j < packet_count; j++) {
	    for (i = 0; i < ndev; i++) {
		cudaSetDevice(i);
                cudaMemcpy(dst[i], src[i], size, cudaMemcpyDeviceToHost);
	    }
	}
        cudaDeviceSynchronize();
	get_cputime(&now, &dt);
	printf("%12d Byte  %12d times %12.6f sec  %12.6f MB/s\n",
	       size, packet_count, dt, (size * packet_count)/MEGA/dt);
	fflush(stdout);
    }
#if 0
    for (i = 0; i < ndev; i++) {
	cudaSetDevice(i);
	cudaFree(src[i]);
	free(dst[i]);
    }
#endif
}

static
void selfperf( int argc, char **argv,
	       size_t minsize, size_t maxsize, size_t nloop) {
    int i, j;
    size_t size;
    double now = 0.0, dt = 0.0;
    size_t ratio = 2;
    char *src[MAXDEV];
    char *dst[MAXDEV];
    int ndev;

    ndev = 1; // !!!

    printf("# %d device%s found.\n", ndev, ndev > 1 ? "s" : "");
    for (i = 0; i < ndev; i++) {
        cudaSetDevice(i);
        cudaMalloc((void**) &src[i], sizeof(char) * maxsize);
        cudaMalloc((void**) &dst[i], sizeof(char) * maxsize);
    }
    printf("#\n");
    printf("# cudaMemcpy(D2D)\n");
    printf("#   - Total size of transfered is %6.1f MByte.\n", (double)nloop/1e6);
    printf("#\n");
    for (size = minsize; size <= maxsize; size *= ratio) {
	get_cputime(&now, &dt);
	for (j = 0; j < nloop/size; j++) {
	    for (i = 0; i < ndev; i++) {
  	        cudaSetDevice(i);
                cudaMemcpy(dst[i], src[i], size, cudaMemcpyDeviceToDevice);
	    }
	}
        cudaDeviceSynchronize();
	get_cputime(&now, &dt);
	printf("%12d Byte    %f sec    %f MB/s\n",
               size, dt, (double)nloop/MEGA/dt);
	fflush(stdout);
    }
}

int
main(int argc, char **argv)
{
    int ndev;
    //size_t minsize = 4 * 1024;
    size_t minsize = 4 * 1000;
    size_t midsize = 512 * 1000;
    size_t maxsize = 65 * 1024 * 1024;
    size_t nloop   = 1024 * 1024 * 1024;
    cudaGetDeviceCount(&ndev);

    sendperf( argc, argv, minsize, midsize, nloop/4);
    sendperf( argc, argv, midsize, maxsize, nloop);
    
    recvperf( argc, argv, minsize, midsize, nloop/4);
    recvperf( argc, argv, midsize, maxsize, nloop);
    
    sleep(1);
    fprintf(stderr, "going to quit...\n");
    exit(0);
}
