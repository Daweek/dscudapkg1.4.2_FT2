//                             -*- Mode: C++ -*-
// Filename         : dscudaverb.h
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-02-12 20:57:57
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef __DSCUDAVERB_H__
#define __DSCUDAVERB_H__
#include "libdscuda.h"
#define DSCUDAVERB_HISTMAX_GROWSIZE (10)
/*** ==========================================================================
 *** Backup memory region of devices allocated by cudaMemcpy().
 ***/
typedef struct BkupMem_t {
    void *dst; /* device momeory space */
    void *src; /* client memory space */
    int   size; /* Byte */
    struct BkupMem_t *next;
    struct BkupMem_t *prev;
    //--- methods
    int isHead( void );
    int isTail( void );
} BkupMem;

typedef struct BkupMemList_t {
    BkupMem *head;        /* pointer to 1st  BkupMem */
    BkupMem *tail;        /* pointer to last BkupMem */
    int     length;       /* Counts of allocated memory region */
    long    total_size;   /* Total size of backuped memory in Byte */
    //--- methods
    int      isEmpty( void );
    int      getLen( void ) { return length; }
    long     getTotalSize( void ) { return total_size; }
    int      countRegion( void );
    int      checkSumRegion( void *targ, int size );
    BkupMem* queryRegion( void *dst );
    void     addRegion( void *dst, int size );
    void     removeRegion( void *dst );
    void*    searchUpdateRegion( void *dst );
    void     updateRegion( void *dst, void *src, int size );
    void     reallocDeviceRegion(RCServer_t *svr);             /* ReLoad backups */
    void     restructDeviceRegion(void);              /* ReLoad backups */
    //---
    BkupMemList_t( void ) { head = tail = NULL; length = 0; total_size = 0; }
} BkupMemList;
/*** ==========================================================================
 *** Each argument types and lists for historical recall.
 *** If you need to memorize another function into history, add new one.
 ***/
typedef struct HistCell_t {
    int   funcID;
    void *args;
} HistCell;
typedef struct HistRecord_t {
    HistCell *hist;
    int length;    /* # of recorded function calls to be recalled */
    int max_len;   /* Upper bound of "verbHistNum", extensible */
    //
    void add( int funcID, void *argp ); /* Add */
    void clear( void );           /* Clear */
    void print( void );           /* Print to stdout */
    int  recall( void );          /* Recall */
    //
    HistRecord_t( void ) { hist = NULL; length = max_len = 0; }
} HistRecord;

typedef struct {                        /* cudaSetDevice() */
    int device;
} cudaSetDeviceArgs;

typedef struct CudaMallocArgs_t {                        /* cudaMalloc() */
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


void dscudaVerbInit(void);                /* Initializer    */


void dscudaVerbMigrateDevice(RCServer_t *svr_from, RCServer_t *svr_to);
void dscudaClearHist(void);
void printRegionalCheckSum(void);

extern BkupMemList BKUPMEM;
extern HistRecord  HISTREC;

#endif // __DSCUDAVERB_H__
