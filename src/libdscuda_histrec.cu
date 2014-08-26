//                             -*- Mode: C++ -*-
// Filename         : dscudaverb.cu
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-08-26 09:39:42
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include "dscuda.h"
#include "dscudarpc.h"
#include "libdscuda.h"
#include "libdscuda_histrec.h"

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

void printRegionalCheckSum(void) {
    BkupMem *pMem = BKUPMEM.head;
    int length = 0;
    while (pMem != NULL) {
	fprintf(stderr, "Region[%d](dp=%p, size=%d): checksum=0x%08x\n",
	       length, pMem->d_region, pMem->size, checkSum(pMem->h_region, pMem->size));
	pMem = pMem->next;
	length++;
    }
}

/*
 *  
 */
HistRecList_t::HistRecList_t(void) {
    length      = 0;
    recalling   = 0;
    
    max_len   = 32;
    hist = (HistRec_t *)malloc( sizeof(HistRec) * max_len );
    if ( hist == NULL ) {
	WARN( 0, "%s():malloc() failed.\n", __func__ );
	exit( EXIT_FAILURE );
    }
    WARN( 5, "The constructor %s() called.\n", __func__ );
}

void
HistRecList_t::extendLen(void) {
    max_len += EXTEND_LEN;
    hist = (HistRec *)realloc( hist, sizeof(HistRec) * max_len );
    if ( hist == NULL ) {
	WARN( 0, "%s():realloc() failed.\n", __func__ );
	exit( EXIT_FAILURE );
    }
    return;
}

/*
 *
 */
void HistRecList_t::add( int funcID, void *argp ) {
    int DSCVMethodId;

    if ( rec_en==0 || recalling==1 ) {
       /* record-enable-flag is disabled, or in recalling process. */
       return;
    }
    if ( length == max_len ) { /* Extend the existing memory region. */
	extendLen();
    }

    DSCVMethodId = funcID2DSCVMethod(funcID);
    hist[length].args   = (storeArgsStub[funcID2DSCVMethod(funcID)])(argp);
    hist[length].funcID = funcID;
    length++; /* Increment the count of cuda call */

    switch (funcID2DSCVMethod(funcID)) {
    case DSCVMethodMemcpyD2D: { /* cudaMemcpy(DevicetoDevice) */
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
      }
    }
    return;
}
/*
 * Clear all hisotry of calling cuda functions.
 */
void HistRecList_t::clear( void ) {
   if ( hist != NULL ) {
      for (int i=0; i < length; i++) {
         (releaseArgsStub[funcID2DSCVMethod( hist[i].funcID)])(hist[i].args);
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

void dscudaClearHist(void) {
    HISTREC.clear();
}

void HistRecList_t::print(void) {
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
	   (recallStub[funcID2DSCVMethod( HISTREC.hist[i].funcID )])(HISTREC.hist[i].args); /* partially recursive */
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
/*
 *
 */
void dscudaVerbMigrateModule() {
    // module not found in the module list.
    // really need to send it to the server.
    int     vi   = vdevidIndex();
    Vdev_t *vdev = St.Vdev + Vdevid[vi];
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
    WARN(1, "Failed 1st= %s\n", from->ip);
    replaceBrokenServer(from, to);
    WARN(1, "Reconnecting to %s replacing %s\n", from->ip, to->ip);
    setupConnection(Vdevid[vdevidIndex()], from);

    BKUPMEM.reallocDeviceRegion(from);
    BKUPMEM.restructDeviceRegion(); /* */
    printModuleList();
    invalidateModuleCache(); /* Clear cache of kernel module to force send .ptx to new hoSt. */
    dscudaVerbMigrateModule(); // not good ;_;, or no need.
    HISTREC.recall();  /* ----- Do redundant calculation(recursive) ----- */
}
