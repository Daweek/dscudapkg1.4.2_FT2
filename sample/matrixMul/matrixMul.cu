#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cutil.h>
#include <cutil_inline.h>


//16からはじめて倍々
//プロットはlogscale
#define N (1024)


__global__ void matrixMul(float *A, float *B, float *C, int size) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < size && col < size) {
    float x = 0.0f;
    for (int k = 0; k < size; k++) {
      x += A[row * size + k] * B[k * size + col];
    }

    C[row * size + col] = x;
  }
}

void randomInit(float *x, int size, float max = 1.0f) {
  for (int i = 0; i < size; i++) {
    x[i] = max * (rand() / (float)RAND_MAX);
  }
}

int main(void) {
  float *hA, *hB, *hC;
  float *dA, *dB, *dC;
  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
  struct timeval start, end;

  hA = new float[N * N];
  hB = new float[N * N];
  hC = new float[N * N];

  randomInit(hA, N * N, 10);
  randomInit(hB, N * N, 10);


  int size = N * N * sizeof(float);

  gettimeofday(&start, NULL);
  cudaMalloc((void**)&dA, size);
  cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&dB, size);
  cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&dC, size);


  matrixMul<<<grid, block>>>(dA, dB, dC, N);

  cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);
  gettimeofday(&end, NULL);

  printf("%f\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0);


  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  delete [] hA;
  delete [] hB;
  delete [] hC;
  
  return 0;
}
