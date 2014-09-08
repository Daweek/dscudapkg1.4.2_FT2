//                             -*- Mode: C++ -*-
// Filename         : dscudaverb.cu
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-09 00:49:08
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "dscuda.h"
#include "dscudarpc.h"
#include "libdscuda.h"
#include "dscudaverb.h"

#define DEBUG

typedef enum {
    DSCVMethodNone = 0,
    DSCVMethodSetDevice,
    //DSCVMethodGetDeviceProperties,
    DSCVMethodMalloc,
    DSCVMethodMemcpyH2D,
    DSCVMethodMemcpyD2D,
    DSCVMethodMemcpyD2H,
    DSCVMethodMemcpyToSymbolH2D,
    DSCVMethodMemcpyToSymbolD2D,
    DSCVMethodFree,
    //DSCVMethodLoadModule,
    DSCVMethodRpcLaunchKernel,
    DSCVMethodIbvLaunchKernel,
    DSCVMethodEnd
} DSCVMethod;


static int
checkSum(void *targ, int size) {
    int sum=0, *ptr = (int *)targ;
    
    for (int s=0; s<size; s+=sizeof(int)) {
	//printf("ptr[%d]= %d\n", s, *ptr);
	sum += *ptr;
	ptr++;
    }
    return sum;
}

//stubs for store/release args, and recall functions.
static void *(*storeArgsStub[DSCVMethodEnd])(void *);
static void (*releaseArgsStub[DSCVMethodEnd])(void *);
static void (*recallStub[DSCVMethodEnd])(void *);

#define DSCUDAVERB_SET_STUBS(mthd) \
  storeArgsStub[DSCVMethod ## mthd] = store ## mthd; \
  releaseArgsStub[DSCVMethod ## mthd] = release ## mthd; \
  recallStub[DSCVMethod ## mthd] = recall ## mthd;

#define DSCUDAVERB_SET_ARGS(mthd) \
  cuda ## mthd ## Args *argsrc; \
  argsrc = (cuda ## mthd ## Args *)argp;

#define DSCUDAVERB_STORE_ARGS(mthd) \
  DSCUDAVERB_SET_ARGS(mthd); \
  cuda ## mthd ## Args *argdst; \
  argdst = (cuda ## mthd ## Args *)malloc(sizeof(cuda ## mthd ## Args)); \
  *argdst = *(cuda ## mthd ## Args *)argp;


//mapping RPCfunctionID to DSCUDAVerbMethodID
static DSCVMethod funcID2DSCVMethod(int funcID) {
    switch (funcID) {
      case dscudaSetDeviceId:
	return DSCVMethodSetDevice;
      case dscudaMallocId:
	return DSCVMethodMalloc;
      case dscudaMemcpyH2DId:
	return DSCVMethodMemcpyH2D;
      case dscudaMemcpyD2DId:
	return DSCVMethodMemcpyD2D;
      case dscudaMemcpyD2HId:
	return DSCVMethodMemcpyD2H;
      case dscudaMemcpyToSymbolH2DId:
	return DSCVMethodMemcpyToSymbolH2D;
      case dscudaMemcpyToSymbolD2DId:
	return DSCVMethodMemcpyToSymbolD2D;
      case dscudaFreeId:
	return DSCVMethodFree;
	/*    
	      case dscudaLoadModuleId:
	      return DSCVMethodLoadModule;
	*/
      case dscudaLaunchKernelId:
	if (dscudaRemoteCallType() == RC_REMOTECALL_TYPE_IBV) {
	    return DSCVMethodIbvLaunchKernel;
	} else {
	    return DSCVMethodRpcLaunchKernel;
	}
      default:
	return DSCVMethodNone;
    }
}

//stubs for store args
static void *
storeSetDevice(void *argp) {
    WARN(3, "add hist cudaSetDevice\n");
    DSCUDAVERB_STORE_ARGS(SetDevice); 
    return argdst;
}

static void *
storeMalloc(void *argp) {
    //nothing to do
    return NULL;
}

static void *
storeMemcpyH2D(void *argp) {
    WARN(3, "add hist cudaMemcpyH2D\n");
    DSCUDAVERB_STORE_ARGS(Memcpy);
    argdst->src = malloc(argsrc->count + 1);
    memcpy(argdst->src, (const void *)argsrc->src, argsrc->count);
    return argdst;
}

static void *
storeMemcpyD2D(void *argp) {
    WARN(3, "add hist cudaMemcpyD2D\n");
    DSCUDAVERB_STORE_ARGS(Memcpy);
    return argdst;
}

static void *
storeMemcpyD2H(void *argp) {
    WARN(3, "add hist cudaMemcpyD2H\n");
    DSCUDAVERB_STORE_ARGS(Memcpy);
    return argdst;
}

static void *
storeMemcpyToSymbolH2D(void *argp) {
    WARN(3, "add hist cudaMemcpyToSymbolH2D\n");
    DSCUDAVERB_STORE_ARGS(MemcpyToSymbol);
    
    int nredundancy = dscudaNredundancy();
    argdst->moduleid = (int *)malloc(sizeof(int) * nredundancy);
    memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);
  
    argdst->symbol = (char *)malloc(sizeof(char) * (strlen(argsrc->symbol) + 1));
    argdst->src = malloc(argsrc->count);
    
    strcpy(argdst->symbol, argsrc->symbol);
    memcpy(argdst->src, argsrc->src, argsrc->count);

    return argdst;
}

static void *
storeMemcpyToSymbolD2D(void *argp) {
    WARN(3, "add hist cudaMemcpyToSymbolD2D\n");
    DSCUDAVERB_STORE_ARGS(MemcpyToSymbol);

    int nredundancy = dscudaNredundancy();
    argdst->moduleid = (int *)malloc(sizeof(int) * nredundancy);
    memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);

    argdst->symbol = (char *)malloc(sizeof(char) * (strlen(argsrc->symbol) + 1));
    strcpy(argdst->symbol, argsrc->symbol);
    
    return argdst;
}

static void *storeFree(void *argp) {
    //nothing to do
    return NULL;
}

static void *storeLoadModule(void *argp) {
    DSCUDAVERB_STORE_ARGS(LoadModule);
    argdst->name = (char *)malloc(sizeof(char) * (strlen(argsrc->name) + 1));
    argdst->strdata = (char *)malloc(sizeof(char) * (strlen(argsrc->strdata) + 1));
    strcpy(argdst->name, argsrc->name);
    strcpy(argdst->strdata, argsrc->strdata);
    return argdst;
}

static void *storeRpcLaunchKernel(void *argp) {
    WARN(3, "add hist RpcLaunchKernel\n");
    DSCUDAVERB_STORE_ARGS(RpcLaunchKernel);

    int nredundancy = dscudaNredundancy();
    argdst->moduleid = (int *)malloc(sizeof(int) * nredundancy);
    memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);
    
    argdst->kname = (char *)malloc(sizeof(char) * strlen(argsrc->kname) + 1);
    strcpy(argdst->kname, argsrc->kname);
    
    int narg = argsrc->args.RCargs_len;
    RCarg *rpcargbuf = (RCarg *)malloc(sizeof(RCarg) * narg);
    memcpy(rpcargbuf, argsrc->args.RCargs_val, sizeof(RCarg) * narg);
    argdst->args.RCargs_val = rpcargbuf;

    return argdst;
}

#if !defined(RPC_ONLY)
static void *
storeIbvLaunchKernel(void *argp) {
    WARN(3, "add hist IbvLaunchKernel\n");
    DSCUDAVERB_STORE_ARGS(IbvLaunchKernel);

    int nredundancy = dscudaNredundancy();
    argdst->moduleid = (int *)malloc(sizeof(int) * nredundancy);
    memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);

    argdst->kname = (char *)malloc(sizeof(char) * strlen(argsrc->kname) + 1);
    strcpy(argdst->kname, argsrc->kname);

    argdst->gdim = (int *)malloc(sizeof(dim3));
    argdst->bdim = (int *)malloc(sizeof(dim3));
    memcpy(argdst->gdim, argsrc->gdim, sizeof(dim3));
    memcpy(argdst->bdim, argsrc->bdim, sizeof(dim3));
    
    int narg = argsrc->narg;
    IbvArg *ibvargbuf = (IbvArg *)malloc(sizeof(IbvArg) * narg);
    memcpy(ibvargbuf, argsrc->arg, sizeof(IbvArg) * narg);
    argdst->arg = ibvargbuf;
    
    return argdst;
}
#endif

//stubs for release args
static void
releaseSetDevice(void *argp) {
    DSCUDAVERB_SET_ARGS(SetDevice);
    free(argsrc);
}

static void
releaseMalloc(void *argp) {
    //nothing to do
}

static void
releaseMemcpyH2D(void *argp) {
    DSCUDAVERB_SET_ARGS(Memcpy);
    free(argsrc->src);
    free(argsrc);
}

static void
releaseMemcpyD2D(void *argp) {
    DSCUDAVERB_SET_ARGS(Memcpy);
    free(argsrc);
}

static void
releaseMemcpyD2H(void *argp) {
    DSCUDAVERB_SET_ARGS(Memcpy);
    free(argsrc);
}

static void
releaseMemcpyToSymbolH2D(void *argp) {
    DSCUDAVERB_SET_ARGS(MemcpyToSymbol);
    free(argsrc->moduleid);
    free(argsrc->symbol);
    free(argsrc->src);
    free(argsrc);
}

static void
releaseMemcpyToSymbolD2D(void *argp) {
    DSCUDAVERB_SET_ARGS(MemcpyToSymbol);
    free(argsrc->moduleid);
    free(argsrc->symbol);
    free(argsrc);

}

static void
releaseFree(void *argp) {
    //nothing to do
}

static void
releaseLoadModule(void *argp) {
    DSCUDAVERB_SET_ARGS(LoadModule);
    free(argsrc->name);
    free(argsrc->strdata);
    free(argsrc);
}

static void
releaseRpcLaunchKernel(void *argp) {
    DSCUDAVERB_SET_ARGS(RpcLaunchKernel);
    free(argsrc->moduleid);
    free(argsrc->kname);
    free(argsrc->args.RCargs_val);
    free(argsrc);
}

#if !defined(RPC_ONLY)
static void
releaseIbvLaunchKernel(void *argp) {
    DSCUDAVERB_SET_ARGS(IbvLaunchKernel);
    free(argsrc->moduleid);
    free(argsrc->kname);
    free(argsrc->gdim);
    free(argsrc->bdim);
    free(argsrc->arg);
    free(argsrc);
}
#endif

//stubs for recall
static
void recallSetDevice(void *argp) {
    DSCUDAVERB_SET_ARGS(SetDevice);
    WARN(3, "Recall cudaSetDevice()...\n");
    cudaSetDevice(argsrc->device);
}

static
void recallMalloc(void *argp) {
    //nothing to do
}

static
void recallMemcpyH2D(void *argp) {
    DSCUDAVERB_SET_ARGS(Memcpy);
    WARN(3, "Recall cudaMemcpyH2D()...\n");
    cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyHostToDevice);
}

static
void recallMemcpyD2D(void *argp) {
    DSCUDAVERB_SET_ARGS(Memcpy);
    WARN(3, "Recall cudaMemcpyD2D()...\n");
    cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyDeviceToDevice);
}

static void
recallMemcpyD2H(void *argp) {
    DSCUDAVERB_SET_ARGS(Memcpy);
    WARN(3, "Recall cudaMemcpyD2H()...\n");
    cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyDeviceToHost);
}

static void
recallMemcpyToSymbolH2D(void *argp) {
    DSCUDAVERB_SET_ARGS(MemcpyToSymbol);
    WARN(3, "recall cudaMemcpyToSymbolH2D\n");
    dscudaMemcpyToSymbolWrapper(argsrc->moduleid, argsrc->symbol, argsrc->src, argsrc->count, argsrc->offset, cudaMemcpyHostToDevice);
}

static void
recallMemcpyToSymbolD2D(void *argp) {
    DSCUDAVERB_SET_ARGS(MemcpyToSymbol);
    WARN(3, "recall cudaMemcpyToSymbolD2D\n");
    dscudaMemcpyToSymbolWrapper(argsrc->moduleid, argsrc->symbol, argsrc->src, argsrc->count, argsrc->offset, cudaMemcpyDeviceToDevice);
}

static void
recallFree(void *argp) {
    //nothing to do
}

static void
recallLoadModule(void *argp) {
    DSCUDAVERB_SET_ARGS(LoadModule);
}

#if !defined(RPC_ONLY)
static void
recallIbvLaunchKernel(void *argp) {
    DSCUDAVERB_SET_ARGS(IbvLaunchKernel);
    WARN(3, "recall IbvLaunchKernel\n");
    ibvDscudaLaunchKernelWrapper(argsrc->moduleid, argsrc->kid, argsrc->kname, argsrc->gdim, argsrc->bdim, argsrc->smemsize, argsrc->stream, argsrc->narg, argsrc->arg);
}
#endif

static void
recallRpcLaunchKernel(void *argp) {
    DSCUDAVERB_SET_ARGS(RpcLaunchKernel);
    WARN(3, "Recall RpcLaunchKernel((int*)moduleid=%p, (int)kid=%d, (char*)kname=%s, ...)...\n",
	 argsrc->moduleid, argsrc->kid, argsrc->kname);
    rpcDscudaLaunchKernelWrapper(argsrc->moduleid, argsrc->kid, argsrc->kname, argsrc->gdim, argsrc->bdim, argsrc->smemsize, argsrc->stream, argsrc->args);
}

//initialize redundant unit
void dscudaVerbInit(void) {
    memset(storeArgsStub,   0, sizeof(DSCVMethod) * DSCVMethodEnd);
    memset(releaseArgsStub, 0, sizeof(DSCVMethod) * DSCVMethodEnd);
    memset(recallStub,      0, sizeof(DSCVMethod) * DSCVMethodEnd);
  
    DSCUDAVERB_SET_STUBS(SetDevice);
    DSCUDAVERB_SET_STUBS(Malloc);
    DSCUDAVERB_SET_STUBS(MemcpyH2D);
    DSCUDAVERB_SET_STUBS(MemcpyD2D);
    DSCUDAVERB_SET_STUBS(MemcpyD2H);
    DSCUDAVERB_SET_STUBS(MemcpyToSymbolH2D);
    DSCUDAVERB_SET_STUBS(MemcpyToSymbolD2D);
    DSCUDAVERB_SET_STUBS(Free);
    //DSCUDAVERB_SET_STUBS(LoadModule);
    DSCUDAVERB_SET_STUBS(RpcLaunchKernel);
    //DSCUDAVERB_SET_STUBS(IbvLaunchKernel); // in kaust debug, 17Aug2014

    for ( int i=1; i<DSCVMethodEnd; i++ ) {
	if (!storeArgsStub[i]) {
	    fprintf(stderr, "dscudaVerbInit: storeArgsStub[%d] is not initialized.\n", i);
	    exit(1);
	}
	if (!releaseArgsStub[i]) {
	    fprintf(stderr, "dscudaVerbInit: releaseArgsStub[%d] is not initialized.\n", i);
	    exit(1);
	}
	if (!recallStub[i]) {
	    fprintf(stderr, "dscudaVerbInit: recallStub[%d] is not initialized.\n", i);
	    exit(1);
	}
    }
    HISTREC.on();
}


