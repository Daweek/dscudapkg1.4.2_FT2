//                             -*- Mode: C++ -*-
// Filename         : dscudaverb.h
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-08-25 17:13:45
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef __DSCUDAVERB_H__
#define __DSCUDAVERB_H__
#include <pthread.h>
#include "libdscuda.h"
#include "libdscuda_bkupmem.h"

typedef struct { /* cudaSetDevice() */
    int device;
} cudaSetDeviceArgs;

typedef struct CudaMallocArgs_t { /* cudaMalloc() */
    void *devPtr;
    size_t size;
    CudaMallocArgs_t( void ) { devPtr = NULL, size = 0; }
    CudaMallocArgs_t( void *ptr, size_t sz ) { devPtr = ptr; size = sz; }
} cudaMallocArgs;

typedef struct CudaMemcpyArgs_t {                        /* cudaMemcpy() */
    void *dst;
    void *src;
    size_t count;
    enum cudaMemcpyKind kind;
    CudaMemcpyArgs_t( void ) { dst = src = NULL; count = 0; }
    CudaMemcpyArgs_t( void *d, void *s, size_t c, enum cudaMemcpyKind k )
    {
	dst = d; src = s; count = c; kind = k;
    }
} cudaMemcpyArgs;

typedef struct {                        /* cudaMemcpyToSymbol */
    int *moduleid;
    char *symbol;
    void *src;
    size_t count;
    size_t offset;
    enum cudaMemcpyKind kind;
} cudaMemcpyToSymbolArgs;

typedef struct {                        /* cudaFree() */
    void *devPtr;
} cudaFreeArgs;

typedef struct {                        /* cudaLoadModule() */
    char *name;
    char *strdata;
} cudaLoadModuleArgs;

typedef struct {                        /* cudaRpcLaunchKernel() */
    int     *moduleid;
    int      kid;
    char    *kname;
    RCdim3   gdim;
    RCdim3   bdim;
    RCsize   smemsize;
    RCstream stream;
    RCargs   args;
} cudaRpcLaunchKernelArgs;

#if 0 // RPC_ONLY
typedef struct {
    int     *moduleid;
    int      kid;
    char    *kname;
    int     *gdim;
    int     *bdim;
    RCsize   smemsize;
    RCstream stream;
    int      narg;
    IbvArg  *arg;
} cudaIbvLaunchKernelArgs;
#endif

void dscudaVerbInit(void);                /* Initializer    */
void dscudaVerbMigrateDevice(RCServer_t *svr_from, RCServer_t *svr_to);
void dscudaClearHist(void);
void printRegionalCheckSum(void);

#endif // __DSCUDAVERB_H__
