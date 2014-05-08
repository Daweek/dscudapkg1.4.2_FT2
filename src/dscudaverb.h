#ifndef __DSCUDAVERB_H__
#define __DSCUDAVERB_H__
#define DSCUDAVERB_HISTMAX_GROWSIZE (10)
/*** ==========================================================================
 *** Backup memory region of devices allocated by cudaMemcpy().
 ***/
typedef struct verbAllocatedMem_t {
    void *dst;
    void *src;
    int   size; /* Byte */
    struct verbAllocatedMem_t *next;
    struct verbAllocatedMem_t *prev;
} verbAllocatedMem;
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
void dscudaVerbRecallHist(void);          /* Recall         */
void dscudaVerbRealloc(void);             /* ReLoad backups */

void dscudaClearHist(void);
void dscudaPrintHist(void);
void printRegionalCheckSum(void);

verbAllocatedMem *verbAllocatedMemQuery(void *dst);
void verbAllocatedMemRegister(void *dst, int size);
void verbAllocatedMemUnregister(void *dst);
void verbAllocatedMemUpdate(void *dst, void *src, int size);
int  verbGetLengthOfMemList(void);

#endif // __DSCUDAVERB_H__
