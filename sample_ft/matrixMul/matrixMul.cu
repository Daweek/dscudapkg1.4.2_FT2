/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11. 
 *
 */

// Utilities and system includes
#include "sdkHelper.h"  // helper for shared functions common to CUDA SDK samples
#include "shrQATest.h"

#include <cuda_runtime.h>

#include "matrixMul.h"

// includes, kernels
#include "matrixMul_kernel.cu"

static char *sSDKsample = "matrixMul";

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( cudaError err, const char *file, const int line ) {
    if( cudaSuccess != err) {
	fprintf(stderr, "%s(%i) : CUDA Runtime API error %d.\n", file, line, (int)err);
	exit(-1);
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)

inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )  {
    cudaError_t err = cudaGetLastError();
        if( cudaSuccess != err) {
            fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d).\n",
                    file, line, errorMessage, (int)err);
            exit(-1);
        }
}

// This function returns the best GPU (with maximum GFLOPS)
int gpuGetMaxGflopsDeviceId() {
    int current_device   = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, max_perf_device  = 0;
    int device_count     = 0, best_SM_arch     = 0;
    cudaDeviceProp deviceProp;

    cudaGetDeviceCount( &device_count );
    // Find the best major SM Architecture GPU device
    while ( current_device < device_count ) {
	cudaGetDeviceProperties( &deviceProp, current_device );
	if (deviceProp.major > 0 && deviceProp.major < 9999) {
	    best_SM_arch = MAX(best_SM_arch, deviceProp.major);
	}
	current_device++;
    }
    
    // Find the best CUDA capable GPU device
    current_device = 0;
    while( current_device < device_count ) {
	cudaGetDeviceProperties( &deviceProp, current_device );
	if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
	    sm_per_multiproc = 1;
	} else {
	    sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	}
	
	int compute_perf  = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
	if( compute_perf  > max_compute_perf ) {
	    // If we find GPU with SM major > 2, search only these
	    if ( best_SM_arch > 2 ) {
		// If our device==dest_SM_arch, choose this, or else pass
		if (deviceProp.major == best_SM_arch) {	
		    max_compute_perf  = compute_perf;
		    max_perf_device   = current_device;
		}
	    } else {
		max_compute_perf  = compute_perf;
		max_perf_device   = current_device;
	    }
	}
	++current_device;
    }
    return max_perf_device;
}

// Initialization code to find the best CUDA Device
int findCudaDevice(int argc, const char **argv) {
    cudaDeviceProp deviceProp;
    int devID = 0;
    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device")) {
	devID = getCmdLineArgumentInt(argc, argv, "device=");
	if (devID < 0) {
	    printf("Invalid command line parameters\n");
	    exit(-1);
	} else {
	    devID = 0;
	    if (devID < 0) {
                   printf("exiting...\n");
                   shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
                   exit(-1);
	    }
	}
    } else {
	// Otherwise pick the device with highest Gflops/s
	devID = gpuGetMaxGflopsDeviceId();
	checkCudaErrors( cudaSetDevice( devID ) );
	checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
	printf("> Using CUDA device [%d]: %s\n", devID, deviceProp.name);
    }
    return devID;
}
// end of CUDA Helper Functions

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int, int, float);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {
    shrQAStart(argc, argv);
    printf("[ %s ]\n", sSDKsample);

    printf("%s\n\tStarting (CUDA and CUBLAS tests)...\n\n", argv[0]);

    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char** argv) {
    if(checkCmdLineFlag(argc, (const char**)argv, "device"))  {
        int devID = getCmdLineArgumentInt(argc, (const char **)argv, "device=");
        if (devID < 0) {
            printf("Invalid command line parameters\n");
            exit(-1);
        } else {
            devID = 0;
            if (devID < 0) {
               printf("exiting...\n");
               shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
               exit(-1);
            }
        }
    } else {
        checkCudaErrors( cudaSetDevice(gpuGetMaxGflopsDeviceId()) );
    }

    int devID = 0;
    cudaDeviceProp props;

    // get number of SMs on this GPU
    checkCudaErrors(cudaGetDeviceProperties(&props, devID));

    // use a larger block size for Fermi and above
    int block_size = (props.major < 2) ? 16 : 32;
    printf("block_size = %d\n", block_size);
    printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name, props.major, props.minor);

    // set seed for rand()
    srand(2006);

    // Optional Command-line multiplier for matrix sizes
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
    int iSizeMultiple = 5;
    if (checkCmdLineFlag( argc, (const char **)argv, "sizemult" )) {
        iSizeMultiple = getCmdLineArgumentInt(argc, (const char**)argv, "sizemult"); 
    }
    
    //iSizeMultiple = CLAMP(iSizeMultiple, 1, 10);
    if ( iSizeMultiple < 1 ) {
	iSizeMultiple = 1;
    } else if ( iSizeMultiple > 128 ) {
	iSizeMultiple = 128;
    }

    // For GPUs with fewer # of SM's, we limit the maximum size of the matrix
    printf("multiProcessorCount = %d\n", props.multiProcessorCount);

    if (props.multiProcessorCount <= 4) {
	uiWA = 2 * block_size * iSizeMultiple;
	uiHA = 4 * block_size * iSizeMultiple;
	uiWB = 2 * block_size * iSizeMultiple;
	uiHB = 4 * block_size * iSizeMultiple;
	uiWC = 2 * block_size * iSizeMultiple;
	uiHC = 4 * block_size * iSizeMultiple;
    } else {
	uiWA = WA * iSizeMultiple;
	uiHA = HA * iSizeMultiple;
	uiWB = WB * iSizeMultiple;
	uiHB = HB * iSizeMultiple;
	uiWC = WC * iSizeMultiple;
	uiHC = HC * iSizeMultiple;
    }
    printf("\nUsing Matrix Sizes: A(%u x %u), B(%u x %u), C(%u x %u)\n\n", 
            uiWA, uiHA, uiWB, uiHB, uiWC, uiHC);

    // allocate host memory for matrices A and B
    unsigned int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    
    // allocate device memory
    float* d_A, *d_B, *d_C;
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float* h_C      = (float*) malloc(mem_size_C);
    float* h_CUBLAS = (float*) malloc(mem_size_C);

    unsigned int mem_size_DEV = 0;
    checkCudaErrors(cudaMalloc((void**) &d_A, mem_size_A));
    checkCudaErrors(cudaMalloc((void**) &d_B, mem_size_B));
    mem_size_DEV += mem_size_A;
    mem_size_DEV += mem_size_B;

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice) );
    checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice) );
    
    checkCudaErrors(cudaMalloc((void**) &d_C, mem_size_C));
    mem_size_DEV += mem_size_C;
   
    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(uiWC / threads.x, uiHC / threads.y);

    // create and start timer
    printf("Runing Kernels...\n\n");

    StopWatchInterface * timer_matrixMul;

    // execute the kernel
    int nIter = 2;
    {
	//Performs warmup operation using matrixMul CUDA kernel
	if (block_size == 16) {
            matrixMul<16><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
        } else {
            matrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
        }
        cudaDeviceSynchronize();

	// Start Timing	
	sdkCreateTimer(&timer_matrixMul);
	sdkStartTimer(&timer_matrixMul);
	for (int j = 0; j < nIter; j++) {
	    if (block_size == 16) {
		matrixMul<16><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
	    } else {
		matrixMul<32><<< grid, threads >>>(d_C, d_A, d_B, uiWA, uiWB);
	    }
	}
	// check if kernel execution generated and error
	getLastCudaError("CUDA matrixMul Kernel execution failed");
	
        cudaDeviceSynchronize();
	// stop and destroy timer
	sdkStopTimer(&timer_matrixMul);
	
	double dSeconds0 = sdkGetTimerValue(&timer_matrixMul)/(1000.0);
	double dSeconds = sdkGetTimerValue(&timer_matrixMul)/((double)nIter * 1000.0);
	double dNumOps = 2.0 * (double)uiWA * (double)uiHA * (double)uiWB;
	double gflops = 1.0e-9 * dNumOps/dSeconds;

	//Log througput, etc
	printf("> CUDA matrixMul %.4f GFlop/s, Time = %.5f s(Elapsed = %.5f s), Size = %.0f Ops.\n", 
	       gflops, dSeconds, dSeconds0, dNumOps);
	
	printf("NumDevsUsed = %d, Workgroup = %u, Elapsed_memsize_DEV = %u [kB]\n", 1, threads.x * threads.y, mem_size_DEV/1000);
	
	sdkDeleteTimer(&timer_matrixMul);
	
	// copy result from device to host
	checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost) );
    }
#if 0 // Bypass comparing with golden.
    // compute reference solution
    printf("\nComparing GPU results with Host computation...\n\n");    
    float* reference = (float*)malloc(mem_size_C);
    computeGold(reference, h_A, h_B, uiHA, uiWA, uiWB);

    // check result (matrixMul)
    printf("Comparing CUDA matrixMul & Host results\n");
    bool resCUDA = sdkCompareL2fe(reference, h_C, size_C, 1.0e-6f);
    if (resCUDA != true) {
        printDiff(reference, h_C, uiWC, uiHC, 100, 1.0e-5f);
    }
    printf("CUDA matrixMul compares %s\n\n", (true == resCUDA) ? "OK" : "FAIL");
    free(reference);
#endif
    
    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
#if 0 // Bypass comparing with golden.
    cudaDeviceReset();
    shrQAFinishExit(argc, (const char **)argv, (resCUDA == true) ? QA_PASSED : QA_FAILED);
#endif
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol) {
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++) {
        if (error_count < iListLength) {
            printf("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) {                
                if (error_count < iListLength) {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    printf(" \n  Total Errors = %d\n\n", error_count);
}
