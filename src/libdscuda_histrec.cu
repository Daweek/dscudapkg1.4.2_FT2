//                             -*- Mode: C++ -*-
// Filename         : libdscuda_histrec.cu
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-17 11:50:29
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

static int
checkSum(void *targ, int size)
{
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
  Cuda ## mthd ## Args *argsrc;		\
  argsrc = (Cuda ## mthd ## Args *)argp; \
  Cuda ## mthd ## Args *argdst; \
  argdst = (Cuda ## mthd ## Args *)malloc(sizeof(Cuda ## mthd ## Args)); \
  *argdst = *(Cuda ## mthd ## Args *)argp;

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
static void*
storeSetDevice(void *argp)
{
    DSCUDAVERB_STORE_ARGS(SetDevice); 
    return argdst;
}

static void *
storeMalloc(void *argp)
{
    //nothing to do
    return NULL;
}

static void *
storeMemcpyH2D(void *argp)
{
    DSCUDAVERB_STORE_ARGS(Memcpy);
    argdst->src = malloc(argsrc->count + 1);
    memcpy(argdst->src, (const void *)argsrc->src, argsrc->count);
    return argdst;
}

static void*
storeMemcpyD2D(void *argp)
{
    DSCUDAVERB_STORE_ARGS(Memcpy);
    return argdst;
}

static void*
storeMemcpyD2H(void *argp)
{
    DSCUDAVERB_STORE_ARGS(Memcpy);
    return argdst;
}

static void*
storeMemcpyToSymbolH2D(void *argp)
{
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

static void*
storeMemcpyToSymbolD2D(void *argp)
{
    WARN(3, "add hist cudaMemcpyToSymbolD2D\n");
    DSCUDAVERB_STORE_ARGS(MemcpyToSymbol);

    int nredundancy = dscudaNredundancy();
    argdst->moduleid = (int *)dscuda::xmalloc(sizeof(int) * nredundancy);
    
    memcpy(argdst->moduleid, argsrc->moduleid, sizeof(int) * nredundancy);

    argdst->symbol = (char *)dscuda::xmalloc(sizeof(char) * (strlen(argsrc->symbol) + 1));

    strcpy(argdst->symbol, argsrc->symbol);
    
    return argdst;
}

static void*
storeFree(void *argp)
{
    //nothing to do
    return NULL;
}

static void*
storeLoadModule(void *argp)
{
    DSCUDAVERB_STORE_ARGS(LoadModule);
    argdst->name = (char *)malloc(sizeof(char) * (strlen(argsrc->name) + 1));
    argdst->strdata = (char *)malloc(sizeof(char) * (strlen(argsrc->strdata) + 1));
    strcpy(argdst->name, argsrc->name);
    strcpy(argdst->strdata, argsrc->strdata);
    return argdst;
}

static void*
storeRpcLaunchKernel(void *argp)
{
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
static void
releaseSetDevice(void *argp)
{
    CudaSetDeviceArgs *argsrc;
    argsrc = (CudaSetDeviceArgs *)argp;
    free(argsrc);
}

static void
releaseMalloc(void *argp)
{
    //nothing to do
}

static void
releaseMemcpyH2D(void *argp)
{
    CudaMemcpyArgs *argsrc;
    argsrc = (CudaMemcpyArgs *)argp;
    free(argsrc->src);
    free(argsrc);
}

static void
releaseMemcpyD2D(void *argp)
{
    CudaMemcpyArgs *argsrc;
    argsrc = (CudaMemcpyArgs *)argp;
    free(argsrc);
}

static void
releaseMemcpyD2H(void *argp)
{
    CudaMemcpyArgs *argsrc;
    argsrc = (CudaMemcpyArgs *)argp;
    free(argsrc);
}

static void
releaseMemcpyToSymbolH2D(void *argp)
{
    CudaMemcpyToSymbolArgs *argsrc;
    argsrc = (CudaMemcpyToSymbolArgs *)argp;
    
    free(argsrc->moduleid);
    free(argsrc->symbol);
    free(argsrc->src);
    free(argsrc);
}

static void
releaseMemcpyToSymbolD2D(void *argp)
{
    CudaMemcpyToSymbolArgs *argsrc;
    argsrc = (CudaMemcpyToSymbolArgs *)argp;

    free(argsrc->moduleid);
    free(argsrc->symbol);
    free(argsrc);
}

static void
releaseFree(void *argp)
{
    //nothing to do
}

static void
releaseLoadModule(void *argp)
{
    CudaLoadModuleArgs *argsrc;
    argsrc = (CudaLoadModuleArgs *)argp;
    
    free(argsrc->name);
    free(argsrc->strdata);
    free(argsrc);
}

static void
releaseRpcLaunchKernel(void *argp)
{
    CudaRpcLaunchKernelArgs *argsrc;
    argsrc = (CudaRpcLaunchKernelArgs *)argp;
    
    //free(argsrc->moduleid);
    free(argsrc->kname);
    free(argsrc->args.RCargs_val);
    free(argsrc);
}

//stubs for recall
static void
recallSetDevice(void *argp)
{
    CudaSetDeviceArgs *argsrc;
    argsrc = (CudaSetDeviceArgs *)argp;

    WARN(3, "Recall cudaSetDevice()...\n");
    cudaSetDevice(argsrc->device);
}

static void
recallMalloc(void *argp)
{
    //nothing to do
}

static void
recallMemcpyH2D(void *argp)
{
    // note: dont insert pthread_mutex_lock or unlock.
    CudaMemcpyArgs *argsrc;
    int         vdevid = Vdevid[ vdevidIndex() ];
    VirDev     *vdev   = St.Vdev + vdevid;
    int         rec_en_stack;
    
    argsrc = (CudaMemcpyArgs *)argp;
    WARN(3, "Recall cudaMemcpyH2D() \n");

    rec_en_stack = vdev->setRecord(0);
    vdev->cudaMemcpyH2D(argsrc->dst, argsrc->src, argsrc->count);
    vdev->setRecord(rec_en_stack);
    WARN(3, "\n");
}

static void
recallMemcpyD2D(void *argp)
{
    // note: dont insert pthread_mutex_lock or unlock.
    CudaMemcpyArgs *argsrc;
    argsrc = (CudaMemcpyArgs *)argp;
    
    WARN(3, "Recall cudaMemcpyD2D()...\n");
    cudaMemcpy(argsrc->dst, argsrc->src, argsrc->count, cudaMemcpyDeviceToDevice);
}

static void
recallMemcpyD2H(void *argp)
{
    // note: dont insert pthread_mutex_lock or unlock.
    CudaMemcpyArgs *argsrc;
    int         vdevid = Vdevid[ vdevidIndex() ];
    VirDev     *vdev   = St.Vdev + vdevid;
    int         rec_en_stack;

    argsrc = (CudaMemcpyArgs *)argp;
    WARN(3, "Recall cudaMemcpyD2H()...\n");

    rec_en_stack = vdev->setRecord(0);
    vdev->cudaMemcpyD2H(argsrc->dst, argsrc->src, argsrc->count);
    vdev->setRecord(rec_en_stack);
    WARN(3, "\n");
}

static void
recallMemcpyToSymbolH2D(void *argp)
{
    CudaMemcpyToSymbolArgs *argsrc;
    argsrc = (CudaMemcpyToSymbolArgs *)argp;
    WARN(3, "recall cudaMemcpyToSymbolH2D\n");
    dscudaMemcpyToSymbolWrapper(argsrc->moduleid, argsrc->symbol, argsrc->src, argsrc->count, argsrc->offset, cudaMemcpyHostToDevice);
}

static void
recallMemcpyToSymbolD2D(void *argp)
{
    CudaMemcpyToSymbolArgs *argsrc;
    argsrc = (CudaMemcpyToSymbolArgs *)argp;
    WARN(3, "recall cudaMemcpyToSymbolD2D\n");
    dscudaMemcpyToSymbolWrapper(argsrc->moduleid, argsrc->symbol, argsrc->src, argsrc->count, argsrc->offset, cudaMemcpyDeviceToDevice);
}

static void
recallFree(void *argp)
{
    //nothing to do
}

static void
recallLoadModule(void *argp)
{
    CudaLoadModuleArgs *argsrc;
    argsrc = (CudaLoadModuleArgs *)argp;
}

static void
recallRpcLaunchKernel(void *argp)
{
    // note: dont insert pthread_mutex_lock or unlock.
    CudaRpcLaunchKernelArgs *argsrc;
    argsrc = (CudaRpcLaunchKernelArgs *)argp;
    WARN(3, "Recall RpcLaunchKernel((int)moduleid=%d, (int)kid=%d, (char*)kname=%s, ...)...\n",
	 argsrc->moduleid, argsrc->kid, argsrc->kname);
#if 0
    rpcDscudaLaunchKernelWrapper(argsrc->moduleid, argsrc->kid, argsrc->kname, argsrc->gdim, argsrc->bdim, argsrc->smemsize, argsrc->stream, argsrc->args);
#else
    int         rec_en_stack;
    VirDev *vdev = St.Vdev + Vdevid[vdevidIndex()];
    rec_en_stack = vdev->setRecord(0);
    vdev->launchKernel(argsrc->moduleid, argsrc->kid, argsrc->kname, argsrc->gdim, argsrc->bdim, argsrc->smemsize, argsrc->stream, argsrc->args);
    vdev->setRecord(rec_en_stack);
#endif
}

/*
 *  CONSTRUCTOR
 */
HistList::HistList(void)
{
    add_count = 0;
    length    = 0;
    byte_size = 0;
    max_len   = 32;
    
    histrec = (HistCell*)dscuda::xmalloc( sizeof(HistCell) * max_len );

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
	    fprintf(stderr, "HistList(constructor): storeArgsStub[%d] is not initialized.\nexit.\n\n", i);
	    exit(1);
	}
	if (!releaseArgsStub[i]) {
	    fprintf(stderr, "HistList(constructor): releaseArgsStub[%d] is not initialized.\nexit.\n\n", i);
	    exit(1);
	}
	if (!recallStub[i]) {
	    fprintf(stderr, "HistList(constructor): recallStub[%d] is not initialized.\nexit.\n\n", i);
	    exit(1);
	}
    }
    //HISTREC.on();
    //--> import from dscudaVerbInit()
    //WARN( 5, "The constructor %s() called.\n", __func__ );
} // HistList::HistList()

void
HistList::extendLen(void)
{
    max_len += EXTEND_LEN;
    histrec = (HistCell*)realloc( histrec, sizeof(HistCell) * max_len );
    if (histrec == NULL) {
	WARN( 0, "%s():realloc() failed.\n", __func__ );
	exit(EXIT_FAILURE);
    }
    return;
}

/*
 * Add one item to called histry of CUDA API. 
 */
void
HistList::add(int funcID, void *argp)
{
    int DSCVMethodId;

    if (length == max_len) { /* Extend the existing memory region. */
	extendLen();
    }

    DSCVMethodId = funcID2DSCVMethod(funcID);
    histrec[length].seq_num = add_count;
    histrec[length].args    = (storeArgsStub[funcID2DSCVMethod(funcID)])(argp);
    histrec[length].funcID  = funcID;
    
    length++; /* Increment the count of cuda call */
    byte_size += sizeof(funcID);
    byte_size += sizeof(int);// dev_id
    add_count++; // count up.
    
    switch (funcID) {
    case dscudaSetDeviceId:
	byte_size += sizeof( CudaSetDeviceArgs );
	break;
    case dscudaMallocId:
	byte_size += sizeof( CudaMallocArgs );
	break;
    case dscudaMemcpyH2DId: //thru
	byte_size += sizeof( CudaMemcpyArgs );
	break;
    case dscudaMemcpyD2HId: //thru
	byte_size += sizeof( CudaMemcpyArgs );
	break;
    case dscudaMemcpyD2DId:
	byte_size += sizeof( CudaMemcpyArgs );
	WARN(3, "add hist cudaMemcpyD2D\n");
	break;
    case dscudaMemcpyToSymbolH2DId: //thru
    case dscudaMemcpyToSymbolD2DId:
	byte_size += sizeof( CudaMemcpyToSymbolArgs );
	break;
    case dscudaFreeId:
	byte_size += sizeof( CudaFreeArgs );
	break;
	/*    
	      case dscudaLoadModuleId:
	      return DSCVMethodLoadModule;
	*/
    case dscudaLaunchKernelId:
	byte_size += sizeof( CudaRpcLaunchKernelArgs ) + 32;
	// 32 is pseudo length of *kname.
	break;
    default:
	WARN(0, "%s():unknown kind of cuda api.\n", __func__);
	exit(1);
    }
}
/*
 * Clear all hisotry of calling cuda functions.
 */
void
HistList::clear(void)
{
   if (histrec != NULL) {
      for (int i=0; i < length; i++) {
         (releaseArgsStub[funcID2DSCVMethod( histrec[i].funcID)])(histrec[i].args);
      }
   }
   length = 0;
   byte_size = 0;
}

void
HistList::setRecallFlag(void)
{
    recall_flag = 1;
}

void
HistList::clrRecallFlag(void)
{
    recall_flag = 0;
}

void
HistList::print(void)
{
    WARN0(1, "<--- Record of CUDA API history Stack  *******\n");
    if (this->length == 0) {
	WARN0(1, "%s(): RecList[]> (Empty).\n", __func__);
	return;
    }
    for (int i=0; i<length; i++) { /* Print recall history. */
	WARN0(1, "        [%d] = #%lld:", i, histrec[i].seq_num);
	switch (histrec[i].funcID) { /* see "dscudarpc.h" */
	case 305: WARN0(1, "cudaSetDevice()\n");        break;
	case 504: WARN0(1, "cudaEventRecord()\n");      break;
	case 505: WARN0(1, "cudaEventSynchronize()\n"); break;
	case 600: WARN0(1, "kernel-call<<< >>>()\n");   break;
	case 700: WARN0(1, "cudaMalloc()\n");           break;
	case 701: WARN0(1, "cudaFree()\n");             break;
	case 703: WARN0(1, "cudaMemcpy(H2D)\n");        break;
	case 704: WARN0(1, "cudaMemcpy(D2H)\n");        break;
	default:  WARN0(1, "/* %d */()\n", histrec[i].funcID);
	}
    }
    WARN0(1, "Occupied memory size is %d Byte.\n",  byte_size);
    WARN0(1, "---> Record of CUDA API history  *************\n");
}
/*
 * Rerun the recorded history of cuda function series.
 */
int
HistList::recall(void)
{
    WARN(9, "HistList::%s() {\n", __func__);
    static int called_depth = 0;
    int result;
    int verb_curr = St.autoverb;
   
    setRecallFlag();

    WARN(1, "called_depth= %d.\n", called_depth);
    if (called_depth < 0) {       /* irregal error */
	WARN(1, "#**********************************************************************\n");
	WARN(1, "# (;_;) DS-CUDA gave up the redundant calculation.                    *\n"); 
	WARN(1, "#       Unexpected error occured. called_depth=%d in %s()             *\n", called_depth, __func__);
	WARN(1, "#**********************************************************************\n\n");
	exit(1);
    } else if (called_depth < RC_REDUNDANT_GIVEUP_COUNT) { /* redundant calculation.*/
	called_depth++;       
	for (int i=0; i< length; i++) { /* Do recall history */
	    WARN(3, "(._.)Rollback API[%4d/%d]................................\n", i, length-1);
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

    WARN(9, "} HistList::%s()\n", __func__);
    St.autoverb = verb_curr;
    clrRecallFlag();
    
   return result;
} // HistList::recall(void)

