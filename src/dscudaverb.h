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
    BkupMem *head;
    BkupMem *tail;
    int     length;
    //--- methods
    int      isEmpty( void );
    int      countRegion( void );
    int      checkSumRegion( void *targ, int size );
    BkupMem* queryRegion( void *dst );
    void     registerRegion( void *dst, int size );
    void     unregisterRegion( void *dst );
    void*    searchUpdateRegion( void *dst );
    void     updateRegion( void *dst, void *src, int size );
    BkupMemList_t( void ) { head = tail = NULL; length = 0; }
} BkupMemList;
/*** ==========================================================================
 *** Each argument types and lists for historical recall.
 *** If you need to memorize another function into history, add new one.
 ***/
typedef struct {                        /* cudaSetDevice() */
    int device;
} cudaSetDeviceArgs;

typedef struct {                        /* cudaMalloc() */
    void *devPtr;
    size_t size;
} cudaMallocArgs;

typedef struct {                        /* cudaMemcpy() */
    void *dst;
    void *src;
    size_t count;
    enum cudaMemcpyKind kind;  
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

typedef struct {
    int   funcID;
    void *args;
} dscudaVerbHist;

void dscudaVerbInit(void);                /* Initializer    */
void dscudaVerbAddHist(int, void *);      /* Add            */
void dscudaVerbClearHist(void);           /* Clear          */
int  dscudaVerbRecallHist(void);          /* Recall         */
void dscudaVerbRealloc(void);             /* ReLoad backups */
void dscudaVerbMemDup(void);              /* ReLoad backups */

void dscudaVerbMigrateDevice(RCServer_t *svr_from, RCServer_t *svr_to);

void dscudaClearHist(void);
void dscudaPrintHist(void);
void printRegionalCheckSum(void);

extern BkupMemList BKUPMEM;

#endif // __DSCUDAVERB_H__
