//                             -*- Mode: C++ -*-
// Filename         : dscudaverb.h
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-08-17 09:07:17
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef __DSCUDAVERB_H__
#define __DSCUDAVERB_H__
#include <pthread.h>
#include "libdscuda.h"
#define DSCUDAVERB_HISTMAX_GROWSIZE (10)
/*** ==========================================================================
 *** Backup memory region of devices allocated by cudaMemcpy().
 ***/
typedef struct BkupMem_t {
    void *dst;        /* server device momeory space (UVA)*/
    void *src;        /* client host memory space */
    void *src_golden;
    int   size;       /* in Byte */
    int   update_rdy; /* 1:"*dst" has valid data, 0:invalid */
    struct BkupMem_t *next;
    struct BkupMem_t *prev;
    //--- methods
    int isHead( void ) {
	if ( prev==NULL ) return 1;
	else              return 0;
    }
    int isTail( void ) {
	if ( next==NULL ) return 1;
	else              return 0;
    }
    void init(void *idst, int isize ) {
	dst = idst;
	size = isize;
	update_rdy = 0;
	src = (void *)malloc( isize );
	src_golden = (void *)malloc( isize );
	if ( src == NULL || src_golden == NULL) {
	    perror("BkupMem_t.init()");
	}
	prev = next = NULL;
    }
    void updateGolden(void) {
	memcpy( src_golden, src, size );
    }
    void restoreGolden(void) {
	cudaError_t cuerr = cudaSuccess;
        cuerr = cudaMemcpy( dst, src_golden, size, cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess) {
	    fprintf(stderr, "%s():cudaMemcpy(H2D) failed.\n", __func__);
	    exit(1);
	}
    }
    BkupMem_t(void) {
	dst = src = NULL;
	size = update_rdy = 0;
    }
} BkupMem;

typedef struct BkupMemList_t {
private:
    pthread_t tid;        /* thread ID of Checkpointing */
    static void* periodicCheckpoint( void *arg );
public:
    int     bkup_en;
    BkupMem *head;        /* pointer to 1st  BkupMem */
    BkupMem *tail;        /* pointer to last BkupMem */
    int     length;       /* Counts of allocated memory region */
    long    total_size;   /* Total size of backuped memory in Byte */
    //--- methods --------------------------------------------------------------
    int      isEmpty( void );
    int      getLen( void ) { return length; }
    long     getTotalSize( void ) { return total_size; }
    int      countRegion( void );
    int      checkSumRegion( void *targ, int size );
    BkupMem* queryRegion( void *dst );
    void     addRegion( void *dst, int size ); // verbAllocatedMemRegister()
    void     removeRegion( void *dst );        // verbAllocatedMemUnregister()
    void*    searchUpdateRegion( void *dst );
    void     updateRegion( void *dst, void *src, int size );
    void     reallocDeviceRegion(RCServer_t *svr);             /* ReLoad backups */
    void     restructDeviceRegion(void);              /* ReLoad backups */
    //---
    BkupMemList_t( void );
    ~BkupMemList_t( void );
} BkupMemList;

/*** ==========================================================================
 *** Each argument types and lists for historical recall.
 *** If you need to memorize another function into history, add new one.
 ***/
typedef struct HistCell_t {
   int  funcID;   // Recorded cuda*() function.
   void *args;    // And its arguments.
   int  dev_id;   // The Device ID, set by last cudaSetDevice().
} HistCell;

typedef struct HistRecord_t {
    HistCell *hist;
    int rec_en;    /* enable */
    int recalling;
    int length;    /* # of recorded function calls to be recalled */
    int max_len;   /* Upper bound of "verbHistNum", extensible */
    //
    void on(void) { rec_en = 1; }
    void off(void) { rec_en = 0; }
    void add( int funcID, void *argp ); /* Add */
    void clear( void );           /* Clear */
    void print( void );           /* Print to stdout */
    int  recall( void );          /* Recall */
    //
    HistRecord_t( void );
} HistRecord;

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
    int *moduleid;
    int kid;
    char *kname;
    RCdim3 gdim;
    RCdim3 bdim;
    RCsize smemsize;
    RCstream stream;
    RCargs args;
} cudaRpcLaunchKernelArgs;

#if 0 // RPC_ONLY
typedef struct {
    int      *moduleid;
    int      kid;
    char     *kname;
    int      *gdim;
    int      *bdim;
    RCsize   smemsize;
    RCstream stream;
    int      narg;
    IbvArg   *arg;
} cudaIbvLaunchKernelArgs;
#endif

void dscudaVerbInit(void);                /* Initializer    */
void dscudaVerbMigrateDevice(RCServer_t *svr_from, RCServer_t *svr_to);
void dscudaClearHist(void);
void printRegionalCheckSum(void);

extern BkupMemList BKUPMEM;
extern HistRecord  HISTREC;

#endif // __DSCUDAVERB_H__
