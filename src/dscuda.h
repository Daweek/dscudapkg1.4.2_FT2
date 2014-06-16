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
#ifndef DSCUDA_H
#define DSCUDA_H

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

/*   */
const char DELIM_VDEV[]  = " "; // Virtual device is seperated by space.
const char DELIM_REDUN[] = ","; // Redundant device is seperated by comma.
const char DELIM_CAND[]  = " "; //
/*   */
const char IDTAG_0[]     = "DSCUDA_FAULT_INJECTION" ; //
/* DS-CUDA search daemons */
const char SEARCH_PING[] = "DSCUDA_DAEMON_PING" ;
const char SEARCH_ACK[]  = "DSCUDA_DAEMON_ACK" ;
const char SEARCH_DELIM[] = ":" ;
const int  SEARCH_BUFLEN = 256 ;    // length of buffer using for dscudad searching.
const int  SEARCH_NUM_TOKEN = 2 ;


enum {
    RC_REMOTECALL_TYPE_RPC = 1,
    RC_REMOTECALL_TYPE_IBV = 2,
};

typedef struct FaultConf
{
    char tag[32];      /* <= "DSCUDA_FAULT_INJECTION" */
    int  overwrite_en; /* Overwrite by DS-CUDA server */
    int  fault_en;     /* ==0: no-fault, >0: fault-count. OVERWRITTEN by SERVER */
    int  fault_count;  /* Number of fault occur */
    int  *d_Nfault;

    FaultConf(int fault_set=0, const char *s=IDTAG_0) {
	cudaError_t err;
	overwrite_en=1;
	fault_en = 0; /* Default */
	fault_count = fault_set;
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
	err = cudaMemcpy(d_Nfault, &fault_count, sizeof(fault_count), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
	    fprintf(stderr, "#Error. cudaMalloc() failed in consructor %s().\n", __func__);
	    exit(1);
	}
#if defined(__DSCUDA__)
	//dscudaRecordHistOn();
	//BKUPMEM.removeRegion(d_Nfault);
#endif
    }
#if 0
  /*
   * Destructor
   */
    ~FaultConf() {
	cudaError_t err;
	cudaSetDevice(0); /* temporary */
	err = cudaFree(p_Nfault);
	if (err != cudaSuccess) {
	    fprintf(stderr, "#Error. cudaFree() failed in destructor %s().\n", __func__);
	    exit(1);
	}
    }
#endif
} FaultConf_t;
   
#endif // DSCUDA_H
