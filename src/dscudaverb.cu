//                             -*- Mode: C++ -*-
// Filename         : dscudaverb.cu
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-02-12 20:57:57
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

HistRecord  HISTREC; /* CUDA calling history. */
BkupMemList BKUPMEM; /* Backup memory regions against all GPU devices. */

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

int BkupMemList_t::isEmpty( void ) {
    if      ( head==NULL && tail==NULL ) return 1;
    else if ( head!=NULL && tail!=NULL ) return 0;
    else {
	fprintf(stderr, "Unexpected error in %s().\n", __func__);
	exit(1);
    }
}

int BkupMemList_t::countRegion( void ) {
    BkupMem *mem = head;
    int count = 0;
    while ( mem != NULL ) {
	mem = mem->next;
	count++;
    }
    return count;
}

int BkupMemList_t::checkSumRegion( void *targ, int size ) {
    int sum=0;
    int  *ptr = (int *)targ;
    
    for (int s=0; s < size; s+=sizeof(int)) {
	sum += *ptr;
	ptr++;
    }
    return sum;
}
/* Class: "BkupMemList_t"
 * Method: queryRegion()
 *
 */
BkupMem* BkupMemList_t::queryRegion( void *dst ) {
    BkupMem *mem = head;
    int i = 0;
    while ( mem != NULL ) { /* Search */
	if ( mem->dst == dst ) { /* tagged by its address on GPU */
	    WARN(10, "---> %s(%p): return %p\n", __func__, dst, mem);
	    return mem;
	}
	WARN(10, "%s(): search %p, check[%d]= %p\n", __func__, dst, i, mem->dst);
	mem = mem->next;
	i++;
    }
    return NULL;
}
/*********************************************************************
 *
 */
void BkupMemList_t::addRegion( void *dst, int size ) {
    BkupMem *mem;
    
    mem = (BkupMem *)malloc(sizeof(BkupMem));
    if ( mem==NULL ) {
	perror( "addRegion()" );
    }
    mem->dst  = dst;
    mem->size = size;
    mem->src  = (void *)malloc( size );
    if ( mem->src == NULL ) {
	perror( "addRegion()" );
    }
    mem->next = NULL;

    if ( isEmpty() ) {
	head = mem;
	mem->prev = NULL;
    } else {
	tail->next = mem;
	mem->prev = tail;
    }
    tail = mem;
    length++;
    total_size += size;

    WARN( 5, "(WARN-5) +--- add BkupMemList[%d]: p_dev=%p, size=%d\n", length - 1, dst, size );
    if ( getLen() < 0 ) {
	fprintf( stderr, "(+_+) Unexpected error in %s()\n", __func__ );
	exit(1);
    }
}
/*====================================================================
 * Class: "BkupMemlist_t"
 * Method: removeRegion()
 */
void BkupMemList_t::removeRegion(void *dst) {
    BkupMem *mem = queryRegion(dst);
    BkupMem *p_list = head;
    int i;
    if ( mem == NULL ) {
	WARN(0, "%s(): not found requested memory region.\n", __func__);
	WARN(0, "mem. list length= %d \n", countRegion());
	i = 0;
	while ( p_list != NULL ) {
	    WARN(0, "mem-list[%d] = %p\n", i, p_list->dst);
	    p_list = p_list->next;
	    i++;
	}
	return;
    } else if ( mem->isHead() ) { // remove head, begin with 2nd.
	head = mem->next;
	if ( head != NULL ) {
	    head->prev = NULL;
	}
    } else if ( mem->isTail() ) {
	tail = mem->prev;
    } else {
	mem->prev->next = mem->next;
    }
    
    total_size -= mem->size;    
    free(mem->src);
    free(mem);
    length--;
    if ( getLen() < 0 ) {
	fprintf( stderr, "(+_+) Unexpected error in %s()\n", __func__ );
	exit(1);
    }
}

void* BkupMemList_t::searchUpdateRegion(void *dst) {
    BkupMem *mem = head;
    char *d_targ  = (char *)dst;
    char *d_begin;
    char *h_begin;
    char *h_p     = NULL;
    int   i = 0;
    
    while (mem) { /* Search */
	d_begin = (char *)mem->dst;
	h_begin = (char *)mem->src;
	
	if (d_targ >= d_begin &&
	    d_targ < (d_begin + mem->size)) {
	    h_p = h_begin + (d_targ - d_begin);
	    break;
	}
	mem = mem->next;
	i++;
    }
    return (void *)h_p;
}

void BkupMemList_t::updateRegion( void *dst, void *src, int size ) {
// dst : GPU device memory region
// src : HOST memory region
    BkupMem *mem;
    void             *src_mirrored;
    
    if ( src == NULL ) {
	WARN(0, "(WARN-0) (+_+) not found backup target memory region (%p).\n", dst);
	exit(1);
    } else {
	//mem = BkupMem.queryRegion(dst);
	//src_mirrored = mem->src;
	src_mirrored = searchUpdateRegion(dst);
	memcpy(src_mirrored, src, size); // update historical memory region.
	WARN(3, "(WARN-3) +--- Also copied to backup region (%p), checksum=%d.\n",
	     dst, checkSumRegion(src, size));
    }
}

/* 
 * Take the data backups of each virtualized GPU to client's host memory
 * after verifying between redundant physical GPUs every specified wall clock
 * time period. The period is defined in second.
 */
void*
BkupMemList_t::periodicCheckpoint( void *arg ) {
    int devid, j;
    int errcheck = 1;
    cudaError_t cuerr;
    int pmem_devid;
    BkupMem *pmem;
    void *lsrc;
    void *ldst;
    RCServer_t *sp;

    void *dst_cand[RC_NREDUNDANCYMAX];
    dscudaMemcpyD2HResult *rp;
    for (;;) { /* infinite loop */
	fprintf(stderr, "%s\n", __func__);
	for ( devid = 0; devid < Nvdev; devid++ ) { /* All virtual GPUs */
	    pmem = BKUPMEM.head;
	    while ( pmem != NULL ) { /* sweep all registered regions */
		pmem_devid = dscudaDevidOfUva( pmem->dst );
		if ( devid == pmem_devid ) {
		    cudaSetDevice_clnt( devid, errcheck );
		    sp = vdev->server;
		    for ( int redun=0; redun < Vdev[devid].nredundancy; redun++ ) {
			
			rp = dscudamemcpyd2hid_1((RCadr)pmem->dst, count, clnt[sp->id]);
			checkResult(rp, sp);
			err = (cudaError_t)rp->err;
			if (rp->err != cudaSuccess) {
			    err = (cudaError_t)rp->err;
			}
			dst_cand[redun] = malloc( pmem->size );
			if (dst_cand[redun] == NULL) {
			    fprintf( stderr, "malloc() failed.\n");
			    exit(1);
			}
			memcpy( dst_cand[redun], rp->buf.RCbuf_val, rp->buf.RCbuf_len);

			xdr_free((xdrproc_t)xdr_dscudaMemcpyD2HResult, (char *)rp);
		    }
		    for ( int redun=0; redun < Vdev[devid].nredundancy; redun++ ) {
			free( dst_cand[redun] );
		    }
		} // if (
		pmem = pmem->next;
	    }
	}
	
	sleep(2);
	pthread_testcancel();/* cancelation available */
    }
}

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

void printRegionalCheckSum(void) {
    BkupMem *pMem = BKUPMEM.head;
    int length = 0;
    while (pMem != NULL) {
	printf("Region[%d](dp=%p, size=%d): checksum=0x%08x\n",
	       length, pMem->dst, pMem->size, checkSum(pMem->src, pMem->size));
	fflush(stdout);
	pMem = pMem->next;
	length++;
    }
}

static cudaError_t
dscudaVerbMalloc(void **devAdrPtr, size_t size, RCServer_t *pSvr) {
    int      vid = vdevidIndex();
    
    void *adrs;
    dscudaMallocResult *rp;
    cudaError_t err = cudaSuccess;
    
    WARN(3, "(WARN-3) %s(%p, %d, RCServer_t *pSvr{id=%d,cid=%d,uniq=%d})...",
	 __func__, devAdrPtr, size, pSvr->id, pSvr->cid, pSvr->uniq);
    //initClient();
    rp = dscudamallocid_1(size, Clnt[Vdevid[vid]][pSvr->id]);
    checkResult(rp, pSvr);
    if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
    }
    adrs = (void*)rp->devAdr;
    WARN(3, "device : devAdrPtr:%p\n", adrs);	
    xdr_free((xdrproc_t)xdr_dscudaMallocResult, (char *)rp);

    RCuvaRegister(Vdevid[vid], &adrs, size);
    *devAdrPtr = dscudaUvaOfAdr(adrs, Vdevid[vid]);
    WARN(3, "done. *devAdrPtr:%p, Length of Registered MemList: %d\n", *devAdrPtr, BKUPMEM.countRegion());

    return err;
}

void BkupMemList_t::reallocDeviceRegion(RCServer_t *svr) {
    BkupMem *mem = head;
    int     verb = St.isAutoVerb();
    int     copy_count = 0;
    int     i = 0;
    
    WARN(1, "#(WARN-1) %s(RCServer_t *sp).\n", __func__);
    WARN(1, "#(WARN-1) Num. of realloc region = %d\n", BKUPMEM.length );
    St.unsetAutoVerb();
    while ( mem != NULL ) {
	/* TODO: select migrateded virtual device, not all region. */
	WARN(5, "#(WARN-1) mem[%d]->dst = %p, size= %d\n", i, mem->dst, mem->size);
	dscudaVerbMalloc(&mem->dst, mem->size, svr);
	mem = mem->next;
	i++;
    }
    St.setAutoVerb(verb);
    WARN(1, "(WARN-1) +--- Done.\n");
}
/* 
 * Resore the all data of a GPU device with backup data on client node.
 */
void BkupMemList_t::restructDeviceRegion(void) {
    BkupMem *mem = head;
    int      verb = St.isAutoVerb();
    int      copy_count = 0;
    unsigned char    *mon;
    float            *fmon;
    int              *imon;

    WARN(2, "(WARN-2) %s(void).\n", __func__);
    St.unsetAutoVerb();
    while (mem != NULL) {
	WARN(1, "###   + region[%d] (dst=%p, src=%p, size=%d) . checksum=0x%08x\n",
	     copy_count++, mem->dst, mem->src, mem->size, checkSum(mem->src, mem->size));
	cudaMemcpy(mem->dst, mem->src, mem->size, cudaMemcpyHostToDevice);
	mem = mem->next;
    }
    St.setAutoVerb( verb );
    WARN(2, "(WARN-2) +--- done.\n");
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

static void *
storeFree(void *argp) {
    //nothing to do
    return NULL;
}

static void *
storeLoadModule(void *argp) {
    DSCUDAVERB_STORE_ARGS(LoadModule);
    argdst->name = (char *)malloc(sizeof(char) * (strlen(argsrc->name) + 1));
    argdst->strdata = (char *)malloc(sizeof(char) * (strlen(argsrc->strdata) + 1));
    strcpy(argdst->name, argsrc->name);
    strcpy(argdst->strdata, argsrc->strdata);
    return argdst;
}

static void *
storeRpcLaunchKernel(void *argp) {
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

static void
recallIbvLaunchKernel(void *argp) {
    DSCUDAVERB_SET_ARGS(IbvLaunchKernel);
    WARN(3, "recall IbvLaunchKernel\n");
    ibvDscudaLaunchKernelWrapper(argsrc->moduleid, argsrc->kid, argsrc->kname, argsrc->gdim, argsrc->bdim, argsrc->smemsize, argsrc->stream, argsrc->narg, argsrc->arg);
}

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
    DSCUDAVERB_SET_STUBS(IbvLaunchKernel);

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
    St.unsetRecordHist();
}

void HistRecord_t::add( int funcID, void *argp ) {
    int DSCVMethodId;

    if ( length == max_len ) { /* Extend the existing memory region. */
	max_len += DSCUDAVERB_HISTMAX_GROWSIZE;
	hist = (HistCell *)realloc( hist, sizeof(HistCell) * max_len );
    }

    DSCVMethodId = funcID2DSCVMethod(funcID);
    hist[length].args = (storeArgsStub[funcID2DSCVMethod(funcID)])(argp);
    hist[length].funcID = funcID;
    length++; /* Increment the count of cuda call */

    switch (funcID2DSCVMethod(funcID)) {
    case DSCVMethodMemcpyD2D: { /* cudaMemcpy(DevicetoDevice) */
	cudaMemcpyArgs *args = (cudaMemcpyArgs *)argp;
	BkupMem *mem = BKUPMEM.queryRegion(args->dst);
	if (!mem) {
	    break;
	}
	int verb = St.isAutoVerb();
	St.unsetAutoVerb();
	cudaMemcpy(mem->dst, args->src, args->count, cudaMemcpyDeviceToHost);
	St.setAutoVerb(verb);
	break;
      }
    }
    return;
}
/*
 *
 */
void HistRecord_t::clear( void ) {
   if ( hist != NULL ) {
      for (int i=0; i < length; i++) {
         (releaseArgsStub[funcID2DSCVMethod( hist[i].funcID)])(hist[i].args);
      }
      //free(hist);
      //hist = NULL;
   }
   length = 0;
}

void dscudaClearHist(void)
{
    HISTREC.clear();
}

void HistRecord_t::print(void)
{
    WARN(1, "%s(): *************************************************\n", __func__);
    if ( length == 0 ) {
	WARN(1, "%s(): Recall History[]> (Empty).\n", __func__);
	return;
    }
    for (int i=0; i < length; i++) { /* Print recall history. */
	WARN(1, "%s(): Recall History[%d]> ", __func__, i);
	switch (hist[i].funcID) { /* see "dscudarpc.h" */
	  case 305: WARN(1, "cudaSetDevice()\n");        break;
	  case 504: WARN(1, "cudaEventRecord()\n");      break;
	  case 505: WARN(1, "cudaEventSynchronize()\n"); break;
	  case 600: WARN(1, "kernel-call<<< >>>()\n");   break;
	  case 700: WARN(1, "cudaMalloc()\n");           break;
	  case 701: WARN(1, "cudaFree()\n");             break;
	  case 703: WARN(1, "cudaMemcpy(H2D)\n");        break;
	  case 704: WARN(1, "cudaMemcpy(D2H)\n");        break;
	  default:  WARN(1, "/* %d */()\n", hist[i].funcID);
	}
    }
    WARN(1, "%s(): *************************************************\n", __func__);
}
/*
 * Rerun the recorded history of cuda function series.
 */
int HistRecord_t::recall(void) {
   static int called_depth = 0;
   int result;
   int verb = St.isAutoVerb();

   St.unsetAutoVerb();
   
   WARN(1, "#<--- Entering (depth=%d) %d function(s)..., %s().\n", called_depth, length, __func__);
   WARN(1, "called_depth= %d.\n", called_depth);
   if (called_depth < 0) {       /* irregal error */
       WARN(1, "#**********************************************************************\n");
       WARN(1, "# (;_;) DS-CUDA gave up the redundant calculation.                    *\n"); 
       WARN(1, "#       Unexpected error occured. called_depth=%d in %s()             *\n", called_depth, __func__);
       WARN(1, "#**********************************************************************\n\n");
       exit(1);
   } else if (called_depth < RC_REDUNDANT_GIVEUP_COUNT) { /* redundant calculation.*/
       this->print();
       called_depth++;       
       for (int i=0; i< length; i++) { /* Do recall history */
	   (recallStub[funcID2DSCVMethod(HISTREC.hist[i].funcID)])(HISTREC.hist[i].args); /* partially recursive */
       }
       called_depth=0;
       result = 0;
   } else { /* try migraion or not. */
       WARN(1, "#**********************************************************************\n");
       WARN(1, "# (;_;) DS-CUDA gave up the redundant calculation.                    *\n"); 
       WARN(1, "#       I have tried %2d times but never matched.                    *\n", RC_REDUNDANT_GIVEUP_COUNT);
       WARN(1, "#**********************************************************************\n\n");
       called_depth=0;
       result = 1;
   }

   St.setAutoVerb( verb );
   WARN(1, "#---> Exiting (depth=%d) done, %s()\n", called_depth, __func__);
   return result;
}
/*
 *
 */
void dscudaVerbMigrateModule() {
    // module not found in the module list.
    // really need to send it to the server.
    int vi = vdevidIndex();
    Vdev_t *vdev = Vdev + Vdevid[vi];
    int i, mid;
    char *ptx_path, *ptx_data;

    ptx_path = CltModulelist[0].name;      // 0 is only for test
    ptx_data = CltModulelist[0].ptx_image; // 0 is only for test
    
    for (i=0; i<vdev->nredundancy; i++) { /* Reload to all redundant devices. */
	mid = dscudaLoadModuleLocal(St.getIpAddress(), getpid(), ptx_path, ptx_data, Vdevid[vi], i);
        WARN(3, "dscudaLoadModuleLocal returns %d\n", mid);
    }
    printModuleList();
}
/*
 *
 */
void dscudaVerbMigrateDevice(RCServer_t *from, RCServer_t *to) {
    WARN(1, "#**********************************************************************\n");
    WARN(1, "# (._.) DS-CUDA will try GPU device migration.\n");
    WARN(1, "#**********************************************************************\n\n");
    WARN(1, "#(WARN-1) Failed 1st= %s\n", from->ip);
    replaceBrokenServer(from, to);
    WARN(1, "#(WARN-1) Reconnecting to %s replacing %s\n", from->ip, to->ip);
    setupConnection(Vdevid[vdevidIndex()], from);

    BKUPMEM.reallocDeviceRegion(from);
    BKUPMEM.restructDeviceRegion(); /* */
    printModuleList();
    invalidateModuleCache(); /* Clear cache of kernel module to force send .ptx to new hoSt. */
    dscudaVerbMigrateModule(); // not good ;_;, or no need.
    HISTREC.recall();  /* ----- Do redundant calculation(recursive) ----- */
}
