#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <cutil.h>
#include <cutil_inline.h>
#include "dscuda.h"

/* Size of GPU threads */
#define Ngpu (2)
const int Nx = 4; //8192;
const int Ny = 4; //8192;
const int Nz = 1;
const int Nxyz = Nx * Ny * Nz;
#define Nthreads (256)

extern "C" __global__ void
matAdd(float *a, float *b, float *c, float *d,
       int Nx, int Ny, int Nz)
{
    int bid = blockIdx.z * ( gridDim.x * gridDim.y ) +
              blockIdx.y * ( gridDim.x ) + blockIdx.x ;
    int tid = threadIdx.z * ( blockDim.x * blockDim.y ) +
              threadIdx.y * ( blockDim.x ) + threadIdx.x;

#if 0
    int *fault_cnt = fault_conf.d_Nfault;
    
    if (!fault_conf.fault_en || *fault_cnt==0) {
      if (tid == 0) {
	d[bid] = a[bid] + b[bid] + c[bid];
      }
    }
    else {
      if (tid == 0) {
	d[bid] = a[bid] + b[bid] + c[bid] + 0.1f;
	if (bid==0) { (*fault_cnt)--; }
      }
    }
#endif
    if (tid == 0) {
	d[bid] = a[bid] + b[bid] + c[bid];
    }
	  
}

extern "C" __global__ void
vecAdd0(float *a, float *b, float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	c[i] = a[i] + b[i];
	printf("%f + %f =? %f\n", a[i], b[i], c[i]);
}

extern "C" __global__ void
matMul(float *a, float *b, float *c, float *d,
       int Nx, int Ny, int Nz)
{
    __shared__ float smem[Nthreads];
    int bid = blockIdx.z * ( gridDim.x * gridDim.y ) +
              blockIdx.y * ( gridDim.x ) + blockIdx.x ;
    int tid = threadIdx.z * ( blockDim.x * blockDim.y ) +
              threadIdx.y * ( blockDim.x ) + threadIdx.x;
    int i, j;
    //int *fault_cnt = fault_conf.d_Nfault;

    __syncthreads();
    if (tid < Nthreads) {
      smem[tid] = 0.0;
    }
    
    __syncthreads();
    for (i=0; i<1; i++) {
      smem[tid] += sqrt( a[bid]*a[bid] + b[bid]*b[bid] + c[bid]*c[bid] ) / (float)(tid + 1);
      smem[tid] = cos( smem[tid] * smem[tid] * smem[tid] );
    }
    
#if 0
    if (bid==7 && tid==0) {
      printf("smem[0]= %f\n", smem[0]);
    }
#endif
    
    __syncthreads();
    if (tid==0) {
      for (i=1; i<Nthreads; i++) {
	smem[0] += smem[i];

      }
#if 0
      if (bid==7) {
	printf("smem[0]= %f\n", smem[0]);
      }
#endif
      d[bid] = smem[0];
    }
    
}

void
init(float *a, float *b, float *c, int Nx, int Ny, int Nz)
{
  int idx;
  for (int k=0; k < Nz; k++) {
    for (int j=0; j < Ny; j++) {
      for (int i=0; i < Nx; i++) {
	idx = k * ( Nx * Ny ) + j * ( Nx ) + i;
	a[idx] = (float)i;
	b[idx] = (float)j;
	c[idx] = (float)k;
      }
    }
  }
}

int
main(void)
{
    int  i, j, k, idx; 
    int t, t_total=1000;
    float *h_a, *h_b, *h_c, *h_d;
    float *d_a, *d_b, *d_c, *d_d;
    
    dim3 blocks(Nx, Ny, Nz);
    dim3 threads(Nthreads, 1, 1);

    int err_flag;

    printf("(info.) start vecadd\n");
    printf("(info.) gridDim= (%d,%d,%d)\n", blocks.x, blocks.y, blocks.z);
    printf("(info.) blockDim=(%d,%d,%d)\n", threads.x, threads.y, threads.z);

    h_a = (float *)malloc(Nxyz * sizeof(float));
    h_b = (float *)malloc(Nxyz * sizeof(float));
    h_c = (float *)malloc(Nxyz * sizeof(float));
    h_d = (float *)malloc(Nxyz * sizeof(float));

    if ( h_a==NULL || h_b==NULL || h_c==NULL || h_d==NULL ) {
      fprintf(stderr, "malloc() failed\n");
      exit(-1);
    }
    
    //FaultConf_t FAULT_CONF(2);
    //FAULT_CONF.overwrite_en = 1;
    //printf("The size of FAULT_CONF is %d Byte.\n", sizeof(FAULT_CONF));
    printf("begin cudas\n"); fflush(stdout);
    for (i=0; i<Ngpu; i++) {
	cutilSafeCall(cudaSetDevice(i));
	cutilSafeCall(cudaMalloc((void**) &d_a, sizeof(float) * Nxyz));
	cutilSafeCall(cudaMalloc((void**) &d_b, sizeof(float) * Nxyz));
	cutilSafeCall(cudaMalloc((void**) &d_c, sizeof(float) * Nxyz));
	cutilSafeCall(cudaMalloc((void**) &d_d, sizeof(float) * Nxyz));
    }    

#if defined(__DSCUDA__)
    dscudaClearHist();     /*** <--- Clear Recall List.        ***/
    dscudaRecordHistOff();  /*** <--- Enable recording history. ***/ 
#endif

    //dscudaRecordHistOn();
    for (t=0; t<t_total; t++) {
	printf("# Try: %d/%d\n", t+1, t_total);
	init(h_a, h_b, h_c, Nx, Ny, Nz);

#if defined(__DSCUDA__)
	dscudaClearHist();     /*** <--- Clear Recall List.        ***/
	dscudaRecordHistOn();  /*** <--- Enable recording history. ***/ 
#endif

	for (i=0; i<Ngpu; i++) {
	    cudaSetDevice(i);
	    cudaMemcpy(d_a, h_a, sizeof(float) * Nxyz, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_b, h_b, sizeof(float) * Nxyz, cudaMemcpyHostToDevice);
	    cudaMemcpy(d_c, h_c, sizeof(float) * Nxyz, cudaMemcpyHostToDevice);
	}
	for (i=0; i<Ngpu; i++) {
	    cudaSetDevice(i);
	    //vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, FAULT_CONF);
	    //matAdd <<<blocks, threads>>> (d_a, d_b, d_c, d_d, Nx, Ny, Nz, FAULT_CONF);
	    //matMul <<<blocks, threads>>> (d_a, d_b, d_c, d_d, Nx, Ny, Nz, FAULT_CONF);
    	    matMul <<<blocks, threads>>> (d_a, d_b, d_c, d_d, Nx, Ny, Nz);
	    //matrix3D <<< blocks, threads >>> (d_a, d_b, d_c, d_d, Nxyz, FAULT_CONF);
	    //cudaThreadSynchronize();
	}
	for (i=0; i<Ngpu; i++) {
	    cudaSetDevice(i);
	    cudaMemcpy(h_d, d_d, sizeof(float) * Nxyz, cudaMemcpyDeviceToHost); /* verify */
	}
#if defined(__DSCUDA__)
	dscudaRecordHistOff();
#endif
	
#if 1
	for (k=0; k < Nz; k++) {
	  for (j=0; j < Ny; j++) {
	    for (i=0; i < Nx; i++) {
	      idx = k * ( Nx * Ny ) + j * ( Nx ) + i;
	      
	      if (i<3 && j<3 && k<3) {
		printf("[%d, %d, %d] %6.2f + %6.2f + %6.2f = %9.6f",
		       i, j, k, h_a[idx], h_b[idx], h_c[idx], h_d[idx]);
		if (h_a[idx] + h_b[idx] + h_c[idx] != h_d[idx]) {
		  printf("   NG");
		  err_flag++;
		}
		printf("\n");
	      }
	      else {
		if (h_a[idx] + h_b[idx] + h_c[idx] != h_d[idx]) {
		  err_flag++;
		}
	      }
	    }
	  }
	}
	if (err_flag==0) printf("completely OK.\n");
	else             printf("%d errors NG.\n");
#endif
    }

    for (i=0; i<Ngpu; i++) {
	cudaSetDevice(i);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);
    }
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);

    return 0;
}
