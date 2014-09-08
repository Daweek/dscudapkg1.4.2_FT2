//                             -*- Mode: C++ -*-
// Filename         : libdscuda_histrec.cu
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-09 00:49:30
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//----------------------------------------------------------------------
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "dscuda.h"
#include "dscudarpc.h"
#include "libdscuda.h"

#define DEBUG

static int checkSum(void *targ, int size) {
    int sum=0, *ptr = (int *)targ;
    
    for (int s=0; s<size; s+=sizeof(int)) {
	//printf("ptr[%d]= %d\n", s, *ptr);
	sum += *ptr;
	ptr++;
    }
    return sum;
}

#define DSCUDAVERB_SET_STUBS(mthd) \
  storeArgsStub[DSCVMethod ## mthd] = store ## mthd; \
  releaseArgsStub[DSCVMethod ## mthd] = release ## mthd; \
  recallStub[DSCVMethod ## mthd] = recall ## mthd;

#define DSCUDAVERB_STORE_ARGS(mthd) \
  cuda ## mthd ## Args *argsrc;		\
  argsrc = (cuda ## mthd ## Args *)argp; \
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
	return DSCVMethodRpcLaunchKernel;
      default:
	return DSCVMethodNone;
    }
}

//stubs for store args
static void *storeSetDevice(void *argp) {
    WARN(3, "add hist cudaSetDevice\n");
    DSCUDAVERB_STORE_ARGS(SetDevice); 
    return argdst;
}

static void *storeMalloc(void *argp) {
    //nothing to do
    return NULL;
}

static void *storeMemcpyH2D(void *argp) {
    WARN(3, "add hist cudaMemcpyH2D\n");
    DSCUDAVERB_STORE_ARGS(Memcpy);
    argdst->src = malloc(argsrc->count + 1);
    memcpy(argdst->src, (const void *)argsrc->src, argsrc->count);
    return argdst;
}

static void *storeMemcpyD2D(void *argp) {
    WARN(3, "add hist cudaMemcpyD2D\n");
    DSCUDAVERB_STORE_ARGS(Memcpy);
    return argdst;
}

static void *storeMemcpyD2H(void *argp) {
    WARN(3, "add hist cudaMemcpyD2H\n");
    DSCUDAVERB_STORE_ARGS(Memcpy);
    return argdst;
}

static void *storeMemcpyToSymbolH2D(void *argp) {
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

static void *storeMemcpyToSymbolD2D(void *argp) {
    WARN(3, "add hist cudaMemcpyToSymbolD2D\n");
    DSCUDAVERB_STORE_ARGS(MemcpyToSymbol);

    int nredundancy = dscudaNredundancy();
    argdst->moduleid = (int *)malloc(sizeof(int) * nredundancy);
    if (argdst->moduleid == NULL) {
	WARN(0, "%s():malloc failed.\n", __func__);
	exit(1);
    }
    
    memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);

    argdst->symbol = (char *)malloc(sizeof(char) * (strlen(argsrc->symbol) + 1));
    if (argdst->symbol == NULL) {
	WARN(0, "%s():malloc failed.\n", __func__);
	exit(1);
    }
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
    //argdst->moduleid = (int *)malloc(sizeof(int) * nredundancy);
    //memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);
    argdst->moduleid = argsrc->moduleid;
    
    argdst->kname = (char *)malloc(sizeof(char) * strlen(argsrc->kname) + 1);
    strcpy(argdst->kname, argsrc->kname);
    
    int narg = argsrc->args.RCargs_len;
    RCarg *rpcargbuf = (RCarg *)malloc(sizeof(RCarg) * narg);
    memcpy(rpcargbuf, argsrc->args.RCargs_val, sizeof(RCarg) * narg);
    argdst->args.RCargs_val = rpcargbuf;

    return argdst;
}

//stubs for release args
static void releaseSetDevice(void *argp) {
    cudaSetDeviceArgs *argsrc;
    argsrc = (cudaSetDeviceArgs *)argp;
    free(argsrc);
}

static void releaseMalloc(void *argp) {
    //nothing to do
}

static void releaseMemcpyH2D(void *argp) {
    cudaMemcpyArgs *argsrc;
    argsrc = (cudaMemcpyArgs *)argp;
    free(argsrc->src);
    free(argsrc);
}

static void releaseMemcpyD2D(void *argp) {
    cudaMemcpyArgs *argsrc;
    argsrc = (cudaMemcpyArgs *)argp;
    free(argsrc);
}

static void releaseMemcpyD2H(void *argp) {
    cudaMemcpyArgs *argsrc;
    argsrc = (cudaMemcpyArgs *)argp;
    free(argsrc);
}

static void releaseMemcpyToSymbolH2D(void *argp) {
    cudaMemcpyToSymbolArgs *argsrc;
    argsrc = (cudaMemcpyToSymbolArgs *)argp;
    
    free(argsrc->moduleid);
    free(argsrc->symbol);
    free(argsrc->src);
    free(argsrc);
}

static void releaseMemcpyToSymbolD2D(void *argp) {
    cudaMemcpyToSymbolArgs *argsrc;
    argsrc = (cudaMemcpyToSymbolArgs *)argp;

    free(argsrc->moduleid);
    free(argsrc->symbol);
    free(argsrc);
}

static void releaseFree(void *argp) {
    //nothing to do
}

static void releaseLoadModule(void *argp) {
    cudaLoadModuleArgs *argsrc;
    argsrc = (cudaLoadModuleArgs *)argp;
    
    free(argsrc->name);
    free(argsrc->strdata);
    free(argsrc);
}

static void releaseRpcLaunchKernel(void *argp) {
    cudaRpcLaunchKernelArgs *argsrc;
    argsrc = (cudaRpcLaunchKernelArgs *)argp;
    
    //free(argsrc->moduleid);
    free(argsrc->kname);
    free(argsrc->args.RCargs_val);
    free(argsrc);
}

//stubs for recall
static void recallSetDevice(void *argp) {
    cudaSetDeviceArgs *argsrc;
    argsrc = (cudaSetDeviceArgs *)argp;

    WARN(3, "Recall cudaSetDevice()...\n");
    cudaSetDevice(argsrc->device);
}

static void recallMalloc(void *argp) {
    //nothing to do
}

static void recallMemcpyH2D(void *argp) {
    cudaMemcpyArgs *argsrc;
    argsrc = (cudaMemcpyArgs *)argp;

    WARN(3, "Recall cudaMemcpyH2D()...\n");
    cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyHostToDevice);
}

static void recallMemcpyD2D(void *argp) {
    cudaMemcpyArgs *argsrc;
    argsrc = (cudaMemcpyArgs *)argp;
    
    WARN(3, "Recall cudaMemcpyD2D()...\n");
    cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyDeviceToDevice);
}

static void recallMemcpyD2H(void *argp) {
    cudaMemcpyArgs *argsrc;
    argsrc = (cudaMemcpyArgs *)argp;
    WARN(3, "Recall cudaMemcpyD2H()...\n");
    cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyDeviceToHost);
}

static void recallMemcpyToSymbolH2D(void *argp) {
    cudaMemcpyToSymbolArgs *argsrc;
    argsrc = (cudaMemcpyToSymbolArgs *)argp;
    WARN(3, "recall cudaMemcpyToSymbolH2D\n");
    dscudaMemcpyToSymbolWrapper(argsrc->moduleid, argsrc->symbol, argsrc->src, argsrc->count, argsrc->offset, cudaMemcpyHostToDevice);
}

static void recallMemcpyToSymbolD2D(void *argp) {
    cudaMemcpyToSymbolArgs *argsrc;
    argsrc = (cudaMemcpyToSymbolArgs *)argp;
    WARN(3, "recall cudaMemcpyToSymbolD2D\n");
    dscudaMemcpyToSymbolWrapper(argsrc->moduleid, argsrc->symbol, argsrc->src, argsrc->count, argsrc->offset, cudaMemcpyDeviceToDevice);
}

static void recallFree(void *argp) {
    //nothing to do
}

static void recallLoadModule(void *argp) {
    cudaLoadModuleArgs *argsrc;
    argsrc = (cudaLoadModuleArgs *)argp;
}

static void recallRpcLaunchKernel(void *argp) {
    cudaRpcLaunchKernelArgs *argsrc;
    argsrc = (cudaRpcLaunchKernelArgs *)argp;
    WARN(3, "Recall RpcLaunchKernel((int*)moduleid=%p, (int)kid=%d, (char*)kname=%s, ...)...\n",
	 argsrc->moduleid, argsrc->kid, argsrc->kname);
    rpcDscudaLaunchKernelWrapper(argsrc->moduleid, argsrc->kid, argsrc->kname, argsrc->gdim, argsrc->bdim, argsrc->smemsize, argsrc->stream, argsrc->args);
}

/*
 *  CONSTRUCTOR
 */
HistRecList_t::HistRecList_t(void) {
    length    = 0;
    max_len   = 32;
    
    histrec = (HistRec_t *)malloc( sizeof(HistRec) * max_len );
    if (histrec == NULL) {
	WARN(0, "%s():malloc() failed.\n", __func__);
	exit(EXIT_FAILURE);
    }

    //<-- import from dscudaVerbInit()
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

    for (int i=1; i<DSCVMethodEnd; i++) {
	if (!storeArgsStub[i]) {
	    fprintf(stderr, "HistRecList_t(constructor): storeArgsStub[%d] is not initialized.\nexit.\n\n", i);
	    exit(1);
	}
	if (!releaseArgsStub[i]) {
	    fprintf(stderr, "HistRecList_t(constructor): releaseArgsStub[%d] is not initialized.\nexit.\n\n", i);
	    exit(1);
	}
	if (!recallStub[i]) {
	    fprintf(stderr, "HistRecList_t(constructor): recallStub[%d] is not initialized.\nexit.\n\n", i);
	    exit(1);
	}
    }
    //HISTREC.on();
    //--> import from dscudaVerbInit()
    //WARN( 5, "The constructor %s() called.\n", __func__ );
}

void
HistRecList_t::extendLen(void) {
    max_len += EXTEND_LEN;
    histrec = (HistRec *)realloc( histrec, sizeof(HistRec) * max_len );
    if (histrec == NULL) {
	WARN( 0, "%s():realloc() failed.\n", __func__ );
	exit(EXIT_FAILURE);
    }
    return;
}

/*
 *
 */
void HistRecList_t::add(int funcID, void *argp) {
    int DSCVMethodId;

    if (length == max_len) { /* Extend the existing memory region. */
	extendLen();
    }

    DSCVMethodId = funcID2DSCVMethod(funcID);
    histrec[length].args   = (storeArgsStub[funcID2DSCVMethod(funcID)])(argp);
    histrec[length].funcID = funcID;
    length++; /* Increment the count of cuda call */

#if 0
    switch (funcID2DSCVMethod(funcID)) {
    case DSCVMethodMemcpyD2D: /* cudaMemcpy(DevicetoDevice) */
	cudaMemcpyArgs *args = (cudaMemcpyArgs *)argp;
	BkupMem *mem = BKUPMEM.queryRegion( args->d_region );
	if (!mem) {
	    break;
	}
	int verb = St.isAutoVerb();
	St.unsetAutoVerb();
	cudaMemcpy(mem->dst, args->src, args->count, cudaMemcpyDeviceToHost);
	St.setAutoVerb(verb);
	break;
    default:
    }
#endif
    return;
}
/*
 * Clear all hisotry of calling cuda functions.
 */
void HistRecList_t::clear(void) {
   if (histrec != NULL) {
      for (int i=0; i < length; i++) {
         (releaseArgsStub[funcID2DSCVMethod( histrec[i].funcID)])(histrec[i].args);
      }
   }
   length = 0;
}

void HistRecList_t::setRecallFlag(void) {
    recall_flag = 1;
}

void HistRecList_t::clrRecallFlag(void) {
    recall_flag = 0;
}

void HistRecList_t::print(void) {
    WARN(1, "%s(): *************************************************\n", __func__);
    if ( length == 0 ) {
	WARN(1, "%s(): Recall History[]> (Empty).\n", __func__);
	return;
    }
    for (int i=0; i < length; i++) { /* Print recall history. */
	WARN(1, "%s(): Recall History[%d]> ", __func__, i);
	switch (histrec[i].funcID) { /* see "dscudarpc.h" */
	  case 305: WARN(1, "cudaSetDevice()\n");        break;
	  case 504: WARN(1, "cudaEventRecord()\n");      break;
	  case 505: WARN(1, "cudaEventSynchronize()\n"); break;
	  case 600: WARN(1, "kernel-call<<< >>>()\n");   break;
	  case 700: WARN(1, "cudaMalloc()\n");           break;
	  case 701: WARN(1, "cudaFree()\n");             break;
	  case 703: WARN(1, "cudaMemcpy(H2D)\n");        break;
	  case 704: WARN(1, "cudaMemcpy(D2H)\n");        break;
	  default:  WARN(1, "/* %d */()\n", histrec[i].funcID);
	}
    }
    WARN(1, "%s(): *************************************************\n", __func__);
}
/*
 * Rerun the recorded history of cuda function series.
 */
int HistRecList_t::recall(void) {
   static int called_depth = 0;
   int result;
   int verb_curr = St.autoverb;

   setRecallFlag();
   
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
	   (recallStub[funcID2DSCVMethod( histrec[i].funcID )])(histrec[i].args); /* partially recursive */
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

   WARN(1, "#---> Exiting (depth=%d) done, %s()\n", called_depth, __func__);
   St.autoverb = verb_curr;
   
   clrRecallFlag();
   
   return result;
}

