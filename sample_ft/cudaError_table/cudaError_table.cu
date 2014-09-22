#include <stdio.h>

#define CUDAERROR 6 

int main(int argc, char** argv) {
    cudaError_t i;

    printf("cudaSuccess = %d\n", cudaSuccess);
    printf("cudaErrorMemoryAllocation = %d\n", cudaErrorMemoryAllocation);
    printf("cudaErrorLaunchTimeout = %d\n", cudaErrorLaunchTimeout);
	
    return 0;
}
