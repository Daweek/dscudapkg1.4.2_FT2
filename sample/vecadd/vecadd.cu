#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <cutil.h>
#include <cutil_inline.h>
#include "dscuda.h"
#include "vecadd.H"

#define N (8)

int main(void) {
    int i, t, t_total=3;
    float a[N], b[N], c[N];
    float *d_a, *d_b, *d_c;

    printf("start vecadd\n"); fflush(stdout);
    //dscudaRecordHistOff();    
    FaultConf_t FAULT_CONF(2);
    FAULT_CONF.overwrite_en = 1;
    printf("The size of FAULT_CONF is %d Byte.\n", sizeof(FAULT_CONF));

    cutilSafeCall(cudaMalloc((void**) &d_a, sizeof(float) * N));
    cutilSafeCall(cudaMalloc((void**) &d_b, sizeof(float) * N));
    cutilSafeCall(cudaMalloc((void**) &d_c, sizeof(float) * N));
    //dscudaRecordHistOn();
    for (t=0; t<t_total; t++) {
	printf("#\n");
	printf("# Try: %d/%d\n", t+1, t_total);
	printf("#\n"); fflush(stdout);
        for (i = 0; i < N; i++) {
            a[i] = rand()%64;
            b[i] = rand()%64;
        }

	//	dscudaClearHist();
	//dscudaRecordHistOn();

        cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);
        vecAdd<<<N, 1>>>(d_a, d_b, d_c, FAULT_CONF);
        cudaMemcpy(c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost); /* verify */

	//dscudaRecordHistOff();

        for (i=0; i<N; i++) {
            printf("% 6.2f + % 6.2f = % 7.2f",
                   a[i], b[i], c[i]);
            if (a[i] + b[i] != c[i]) printf("   NG");
            printf("\n");
        }
        printf("\n"); fflush(stdout);
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
