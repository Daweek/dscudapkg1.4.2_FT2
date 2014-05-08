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
#include "dscuda.h"
#include "dscudarpc.h"
#include "libdscuda.h"
#include "dscudaverb.h"

#define DEBUG

static dscudaVerbHist   *verbHists = NULL;
static int               verbHistNum = 0; /* Number of recorded function calls to be recalled */
static int               verbHistMax = 0; /* Upper bound of "verbHistNum", extensible */
static verbAllocatedMem *verbAllocatedMemListTop = NULL;
static verbAllocatedMem *verbAllocatedMemListTail = NULL;

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

int
verbGetLengthOfMemList(void)
{
    verbAllocatedMem *pMem = verbAllocatedMemListTop;
    int length = 0;
    while (pMem != NULL) {
	pMem = pMem->next;
	length++;
    }
    return length;
}

verbAllocatedMem *
verbAllocatedMemQuery(void *dst)
{
    verbAllocatedMem *mem = verbAllocatedMemListTop;

    while (mem != NULL) { /* Search */
	if (mem->dst == dst) { /* tagged by its address on GPU */
	    WARN(10, "---> %s(%p): return %p\n", __func__, dst, mem);
	    return mem;
	}
	mem = mem->next;
    }
    return NULL;
}

void *
verbAllocatedMemUpdateQuery(void *dst)
{
    verbAllocatedMem *mem = verbAllocatedMemListTop;
    //WARN(2, "<--- %s(%p):\n", __func__, dst);
    char *d_targ  = (char *)dst;
    char *d_begin;
    char *h_begin;
    char *h_p     = NULL;
    int   i = 0;

    //WARN(2, "   + d_targ  = %p\n", d_targ);
    while (mem) { /* Search */
	d_begin = (char *)mem->dst;
	h_begin = (char *)mem->src;
	//WARN(2, "   + d_begin[%d] = %p\n", i, d_begin);
	//WARN(2, "   + h_begin[%d] = %p\n", i, h_begin);
	if (d_targ >= d_begin &&
	    d_targ < (d_begin + mem->size)) {
	    h_p = h_begin + (d_targ - d_begin);
	    break;
	}
	mem = mem->next;
	i++;
    }
    WARN(10, "---> %s(%p): return %p\n", __func__, dst, h_p);
    return (void *)h_p;
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
void
printRegionalCheckSum(void) {
    verbAllocatedMem *pMem = verbAllocatedMemListTop;
    int length = 0;
    while (pMem != NULL) {
	printf("Region[%d](dp=%p, size=%d): checksum=0x%08x\n",
	       length, pMem->dst, pMem->size, checkSum(pMem->src, pMem->size));
	fflush(stdout);
	pMem = pMem->next;
	length++;
    }
}
/*
 * Register cudaMalloc()
 */
void
verbAllocatedMemRegister(void *dst, int size)
{
    static int i=0;
    // WARN(10, "<--- %s(dst=%p, size=%d) [%d]\n", __func__, dst, size, i);
    verbAllocatedMem *mem = (verbAllocatedMem *)malloc(sizeof(verbAllocatedMem));
    if (!mem) {
	perror("verbAllocatedMemRegister");
    }
    mem->dst  = dst;
    mem->size = size;
    mem->src  = malloc(size);
    mem->next = NULL;
    
    if (verbAllocatedMemListTop == NULL) {
	verbAllocatedMemListTop = mem;
	mem->prev = NULL;
    }
    else {
	verbAllocatedMemListTail->next = mem;
	mem->prev = verbAllocatedMemListTail;
    }
    verbAllocatedMemListTail = mem;
    // WARN(10, "---> %s(dst=%p, size=%d: mem=%p, src=%p) [%d]\n", __func__, dst, size, mem, mem->src, i);
    i++;
}

void
verbAllocatedMemUnregister(void *dst)
{
    // WARN(10, "<--- %s(dst=%p)\n", __func__, dst);
    verbAllocatedMem *mem = verbAllocatedMemQuery(dst);
    verbAllocatedMem *p_list = verbAllocatedMemListTop;
    int i;
    if (!mem) {
	WARN(0, "%s(): not found requested memory region.\n", __func__);
	WARN(0, "mem. list length= %d \n", verbGetLengthOfMemList());
	i = 0;
	while (p_list != NULL) {
	    WARN(0, "mem-list[%d] = %p\n", i, p_list->dst);
	    p_list = p_list->next;
	    i++;
	}
	return;
    }

    if (mem->prev != NULL) { /* not Top */
	//WARN(2, "not TOP\n");
	mem->prev->next = mem->next;
    }
    else {
	//WARN(2, "is TOP\n");
	verbAllocatedMemListTop = mem->next;
	if (mem->next) {
	    mem->next->prev = NULL;
	}
    }

    if (!mem->next) {
	verbAllocatedMemListTail = mem->prev;
    }

    free(mem->src);
    free(mem);
    // WARN(10, "---> %s(dst=%p)\n", __func__, dst);
}

void
verbAllocatedMemUpdate(void *dst, void *src, int size)
// dst : GPU device memory region
// src : HOST memory region
{
    verbAllocatedMem *mem;
    void             *src_mirrored;

    WARN(10, "    <--- %s(dst=%p, src=%p, size=%d)\n", __func__, dst, src, size);

    if (src == NULL) {
	WARN(0, "(+_+) not found backup target memory region (%p).\n", dst);
	exit(1);
    }
    else {
	//mem = verbAllocatedMemQuery(dst);
	//src_mirrored = mem->src;
	src_mirrored = verbAllocatedMemUpdateQuery(dst);
	memcpy(src_mirrored, src, size); // update historical memory region.
	WARN(10, "        Also copied to backup region (%p), checksum=%d.\n",
	     dst, checkSum(src, size));
	printRegionalCheckSum();
    }
    WARN(10, "    ---> %s(dst=%p, src=%p, size=%d)\n", __func__, dst, src, size); 
}

void
dscudaVerbRealloc(void)
{
    verbAllocatedMem *mem = verbAllocatedMemListTop;
    int               verb = St.isAutoVerb();
    int               copy_count = 0;
    unsigned char    *mon;
    float            *fmon;
    int              *imon;

    WARN(1, "###============================================================\n");
    WARN(1, "### %s() called.\n", __func__);
    WARN(1, "###============================================================\n");

    while (mem != NULL) {
	//cudaMalloc(&mem->dst, mem->size); /* To migrate another GPU, you need to do cudaMalloc() before following cudaMemcpy(). */
	WARN(2, "Restore device memory region[%d] (dst=%p, src=%p, size=%d) . checksum=0x%08x\n",
	     copy_count++, mem->dst, mem->src, mem->size, checkSum(mem->src, mem->size));
	if (mem->size <= 16) {
	    fmon = (float *)mem->src;
	    imon = (int   *)mem->src;
	    for (int i=0; i < (mem->size / sizeof(float)); i++) {
		WARN(2, "  + float[%d]= %f, int[%d]= %d\n", i, *fmon, i, *imon);
		fmon++;
		imon++;
	    }
	}
	cudaMemcpy(mem->dst, mem->src, mem->size, cudaMemcpyHostToDevice);
	// mon = (unsigned char *)mem->src;
	// for (int i=0; i<16; i++) {
	//     printf("%04d: ", i);
	//     for (int j=0; j<16; j++) {
	// 	printf("%02x ", *mon);
	// 	mon++;
	//     }
	//     printf("\n"); fflush(stdout);
	// }
	mem = mem->next;
    }
    WARN(1, "###============================================================\n");
    WARN(1, "### %s() done.\n", __func__);
    WARN(1, "###============================================================\n");
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
storeRpcLaunchKernel(void *argp)
{
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
storeIbvLaunchKernel(void *argp)
{
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
static void
recallSetDevice(void *argp) {
    DSCUDAVERB_SET_ARGS(SetDevice);
    WARN(3, "Recall cudaSetDevice()...\n");
    cudaSetDevice(argsrc->device);
}

static void
recallMalloc(void *argp) {
    //nothing to do
}

static void
recallMemcpyH2D(void *argp) {
    DSCUDAVERB_SET_ARGS(Memcpy);
    WARN(3, "Recall cudaMemcpyH2D()...\n");
    cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyHostToDevice);
}

static void
recallMemcpyD2D(void *argp) {
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
    WARN(3, "Recall RpcLaunchKernel()...\n");
    rpcDscudaLaunchKernelWrapper(argsrc->moduleid, argsrc->kid, argsrc->kname, argsrc->gdim, argsrc->bdim, argsrc->smemsize, argsrc->stream, argsrc->args);
}

//initialize redundant unit
void
dscudaVerbInit(void) {
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

    for (int i=1; i<DSCVMethodEnd; i++) {
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

void
dscudaVerbAddHist(int funcID, void *argp)
{
    int DSCVMethodId;

    if (verbHistNum == verbHistMax) { /* Extend the existing memory region. */
	verbHistMax += DSCUDAVERB_HISTMAX_GROWSIZE;
	verbHists = (dscudaVerbHist *)realloc(verbHists, sizeof(dscudaVerbHist) * verbHistMax);
    }

    DSCVMethodId = funcID2DSCVMethod(funcID);
    verbHists[verbHistNum].args = (storeArgsStub[funcID2DSCVMethod(funcID)])(argp);
    verbHists[verbHistNum].funcID = funcID;
    verbHistNum++; /* Increment the count of cuda call */

    switch (funcID2DSCVMethod(funcID)) {
      case DSCVMethodMemcpyD2D: { /* cudaMemcpy(DevicetoDevice) */
	  cudaMemcpyArgs *args = (cudaMemcpyArgs *)argp;
	  verbAllocatedMem *mem = verbAllocatedMemQuery(args->dst);
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
void
dscudaVerbClearHist(void)
{
   if (verbHists) {
      for (int i=0; i<verbHistNum; i++) {
         (releaseArgsStub[funcID2DSCVMethod(verbHists[i].funcID)])(verbHists[i].args);
      }
      //free(verbHists);
      //verbHists = NULL;
   }
   verbHistNum = 0;
   
   WARN(3, "\"%s\":%s()> function history cleared.\n", __FILE__, __func__);
   return;
}

void
dscudaClearHist(void)
{
    dscudaVerbClearHist();
}

void
dscudaPrintHist(void)
{
    WARN(1, "%s(): *************************************************\n", __func__);
    if (verbHistNum==0) {
	WARN(1, "%s(): Recall History[]> (Empty).\n", __func__);
	return;
    }
    for (int i=0; i<verbHistNum; i++) { /* Print recall history. */
	WARN(1, "%s(): Recall History[%d]> ", __func__, i);
	switch (verbHists[i].funcID) { /* see "dscudarpc.h" */
	  case 305: WARN(1, "cudaSetDevice()\n");        break;
	  case 504: WARN(1, "cudaEventRecord()\n");      break;
	  case 505: WARN(1, "cudaEventSynchronize()\n"); break;
	  case 600: WARN(1, "kernel-call<<< >>>()\n");   break;
	  case 700: WARN(1, "cudaMalloc()\n");           break;
	  case 701: WARN(1, "cudaFree()\n");             break;
	  case 703: WARN(1, "cudaMemcpy(H2D)\n");        break;
	  case 704: WARN(1, "cudaMemcpy(D2H)\n");        break;
	  default:  WARN(1, "/* %d */()\n", verbHists[i].funcID);
	}
    }
    WARN(1, "%s(): *************************************************\n", __func__);
}

void
dscudaVerbRecallHist(void)
{
   char       func_name[256]; 
   static int called_depth=0;

   WARN(1, "#<--- Entering (depth=%d) %d function(s)..., %s().\n", called_depth, verbHistNum, __func__);
   WARN(1, "called_depth= %d.\n", called_depth);
   if (called_depth >= RC_REDUNDANT_GIVEUP_COUNT) {
       WARN(1, "#*****************************************************\n");
       WARN(1, "# (;_;) I give up redundant calculation.             \n"); 
       WARN(1, "#       I have tried %d times and all failed.        \n", RC_REDUNDANT_GIVEUP_COUNT);
       WARN(1, "#*****************************************************\n");
       exit(1);
       called_depth=0;
   }
   else {
       dscudaPrintHist();
       called_depth++;       
       for (int i=0; i<verbHistNum; i++) { /* Do recall history */
	   (recallStub[funcID2DSCVMethod(verbHists[i].funcID)])(verbHists[i].args); /* partially recursive */
       }
       called_depth=0;
   }
   WARN(1, "#---> Exiting (depth=%d) done, %s()\n", called_depth, __func__);
}
