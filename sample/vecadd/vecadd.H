
extern "C" __global__ void
vecAdd(float *a, float *b, float *c, ReplacedVar_t fault_conf)
{
    //    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.x;
    int *fault_cnt = fault_conf.d_Nfault;
    printf("[vecadd] fault_conf is %d/%d  %s.\n",
	   *fault_cnt, fault_conf.fault_on, fault_conf.tag);

    if (!fault_conf.fault_on || *fault_cnt==0) {
	c[i] = a[i] + b[i];
	printf("[normal] %f + %f =? %f\n", a[i], b[i], c[i]);
    }
    else {
	c[i] = a[i] + b[i] + 0.1f;
	printf("[fault] %f + %f =? %f\n", a[i], b[i], c[i]);
	if (blockIdx.x==0) { (*fault_cnt)--; }
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
vecMul(float *a, float *b, float c, float *d, int e, int * f)
{
    int i = threadIdx.x;
    d[i] = a[i] * b[i] + c + e + f[i];
}
/*
extern "C" __global__ void
printFaultConfig()
{
    printf("FaultInjectionConfig is 0x%x @ 0x%llx\n",
	   Fault_Injection_Config[0], Fault_Injection_Config) ;
}
*/
