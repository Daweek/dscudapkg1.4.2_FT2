//                             -*- Mode: C++ -*-
// Filename         : dscuda.h
// Description      : DS-CUDA client node library for Remote Procedure Call.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-02-12 20:57:57
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef _DSCUDA_H
#define _DSCUDA_H

#include <cuda_runtime_api.h>
#include <cutil.h>
#include <builtin_types.h>
#include <driver_types.h>
#include <cuda_texture_types.h>
#include <texture_types.h>
#include "dscudautil.h"
#include "dscudarpc.h"
#include "dscudadefs.h"
#include "ibv_rdma.h"
#include "dscudamacros.h"
#include "dscudaverb.h"
#include "libdscuda.h"

enum {
    RC_REMOTECALL_TYPE_RPC = 1,
    RC_REMOTECALL_TYPE_IBV = 2,
};

typedef struct ReplacedVar {
    char tag[32];     /* <= "DSCUDA_FAULT_INJECTION" */
    int  overwrite_en;
    int  fault_on;    /* ==0: no-fault, >0: fault-count. OVERWRITTEN by SERVER */
    int   h_Nfault;   /* */
    int  *d_Nfault;
    ReplacedVar(int fault_set=0, const char *s="DSCUDA_FAULT_INJECTION") {
	cudaError_t err;
	overwrite_en=1;
	fault_on = 0; /* Default */
	h_Nfault = fault_set;
	strcpy(tag, s);
	/* malloc on device */
#if defined(__DSCUDA__)
	dscudaRecordHistOff();
#endif
	cudaSetDevice(0); /* temporary */
	err = cudaMalloc(&d_Nfault, sizeof(int));
	if (err != cudaSuccess) {
	    fprintf(stderr, "#Error. cudaMalloc() failed in consructor %s().\n", __func__);
	    exit(1);
	}
	/* set initial value on device */
	err = cudaMemcpy(d_Nfault, &h_Nfault, sizeof(h_Nfault), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "#Error. cudaMalloc() failed in consructor %s().\n", __func__);
	    exit(1);
	}
#if defined(__DSCUDA__)
	dscudaRecordHistOn();
	verbAllocatedMemUnregister(d_Nfault);
#endif
    }
#if 0
    ~ReplacedVar() {
	cudaError_t err;
	cudaSetDevice(0); /* temporary */
	err = cudaFree(p_Nfault);
	if (err != cudaSuccess) {
	    fprintf(stderr, "#Error. cudaFree() failed in destructor %s().\n", __func__);
	    exit(1);
	}
    }
#endif
} ReplacedVar_t;
   
// defined in libdscuda.cu

#endif // _DSCUDA_H
