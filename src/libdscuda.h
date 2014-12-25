//                             -*- Mode: C++ -*-
// Filename         : libdscuda.h
// Description      : DS-CUDA client node common(IBV and RPC) library header.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-17 10:55:33
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef __LIBDSCUDA_H__
#define __LIBDSCUDA_H__
#include "dscudadefs.h"
#include "dscudarpc.h"
#include "sockutil.h"

//**************************************************************************
//*** "Cuda*Args_t"
//*** CUDA API arguments packing list.
//***
//**************************************************************************
struct CudaSetDeviceArgs
{
    int      device;
};
struct CudaMallocArgs
{
    void    *devPtr;
    size_t   size;
    CudaMallocArgs(void) { devPtr = NULL, size = 0; }
    CudaMallocArgs( void *ptr, size_t sz ) { devPtr = ptr; size = sz; }
};
struct CudaMemcpyArgs
{
    void    *dst;
    void    *src;
    size_t   count;
    enum cudaMemcpyKind kind;
    CudaMemcpyArgs( void ) { dst = src = NULL; count = 0; }
    CudaMemcpyArgs( void *d, void *s, size_t c, enum cudaMemcpyKind k ) {
	dst = d; src = s; count = c; kind = k;
    }
};
struct CudaMemcpyToSymbolArgs
{
    int     *moduleid;
    char    *symbol;
    void    *src;
    size_t   count;
    size_t   offset;
    enum cudaMemcpyKind kind;
};
struct CudaFreeArgs
{
    void    *devPtr;
};
struct CudaLoadModuleArgs
{
    char    *name;
    char    *strdata;
};
struct CudaRpcLaunchKernelArgs
{
    int      moduleid;
    int      kid;
    char    *kname;
    RCdim3   gdim;
    RCdim3   bdim;
    RCsize   smemsize;
    RCstream stream;
    RCargs   args;
};
//************************************************************************
//***  Class Name: "BkupMem_t"
//***  Description:
//***    - Backup memory region of devices allocated by cudaMemcpy().
//***    - mirroring of a global memory region to client memory region.
//***    - In case when device memory region was corrupted, restore with
//***    - clean data to device memory.
//***    - In case when using device don't response from client request,
//***    - migrate to another device and restore with clean data.
//************************************************************************
struct BkupMem
{
    void  *v_region;        // UVA, Search index, and also Virtual device address.
    void  *d_region;        //!UVA, server device memory space.
    void  *h_region;        //
    int    size;            // in Byte.
    int    update_rdy;      // 1:"*dst" has valid data, 0:invalid.
    BkupMem *next; // For double-linked-list prev.
    BkupMem *prev; // For double-linked-list next.
    /*constructor/destructor.*/
    BkupMem(void);
    //--- methods
    void   init(void *uva_ptr, void *d_ptr, int isize);
    int    isHead(void);
    int    isTail(void);
    int    calcSum(void);
    void  *translateAddrVtoD(const void *v_ptr);
    void  *translateAddrVtoH(const void *v_ptr);
};

//********************************************************************
//***  Class Name: "BkupMemList"
//***  Description:
//***      - 
//********************************************************************
struct BkupMemList
{
public:
    BkupMem *head;        /* pointer to 1st  BkupMem */
    BkupMem *tail;        /* pointer to last BkupMem */
    int      length;       /* Counts of allocated memory region */
    long     total_size;   /* Total size of backuped memory in Byte */
    //--- construct/destruct
    BkupMemList(void);
    ~BkupMemList(void);
    //--- methods ---------------
    void     print(void);
    BkupMem* query(void *uva_ptr);
    void     add(void *uva_ptr, void *dst, int size);
    void     remove(void *uva_ptr);        // verbAllocatedMemUnregister()
    int      isEmpty(void);
    int      getLen(void);
    long     getTotalSize(void); // get total size of allocated memory.
    int      countRegion(void);
    int      checkSumRegion(void *targ, int size);
    void*    queryHostPtr(const void *v_ptr);
    void*    queryDevicePtr(const void *v_ptr);
    void     restructDeviceRegion(void);              /* ReLoad backups */
};

//********************************************************************
//***
//*** CUDA API call record stub.
//***
//********************************************************************
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
    //DSCVMethodIbvLaunchKernel,
    DSCVMethodEnd
} DSCVMethod;
//********************************************************************
//***  Class Name: "HistCell"
//***  Description:
//***      - Recording the sequential CUDA-call.
//********************************************************************
struct HistCell
{
    int64_t seq_num;  // unique serial ID.
    int     funcID;   // Recorded cuda*() function.
    void   *args;     // And its arguments.
    int     dev_id;   // The Device ID, set by last cudaSetDevice().
};
//********************************************************************
//***  Class Name: "HistList"
//***  Description:
//***      - 
//********************************************************************
struct HistList
{
    HistCell *histrec;
    int      length;    /* # of recorded function calls to be recalled */
    int      byte_size; // Total size of this history
    int      max_len;   /* Upper bound of "verbHistNum", extensible */
    int64_t  add_count; // incremented by method add().

    // stubs for store/release args, and recall functions.
    // pointer array to arbitary functions.
    void *(*storeArgsStub[DSCVMethodEnd])(void *);
    void (*releaseArgsStub[DSCVMethodEnd])(void *);
    void (*recallStub[DSCVMethodEnd])(void *);
    
    /*CONSTRUCTOR*/
    HistList(void);
    //
    void add(int funcID, void *argp);
    void clear(void);           /* Clear */
    void print(void);           /* Print to stdout */
    int  recall(void);          /* Recall */
private:
    static const int EXTEND_LEN = 32; // Size of growing of "max_len"
    int      recall_flag;
    void     setRecallFlag(void);
    void     clrRecallFlag(void);
    void     extendLen(void);
};
//********************************************************************
//***  Class Name: "PtxRecord"
//***  Description:
//***      - CUDA Kernel function module management for Client.
//********************************************************************
struct PtxRecord
{
    int  valid;   //1:valid, 0:invalid.
    char name[RC_KMODULENAMELEN];
    char ptx_image[RC_KMODULEIMAGELEN]; //!Caution; may be Large size.
    /*CONSTRUCTOR*/
    PtxRecord(void);
    /*METHODS*/
    void invalidate(void);
    void set(char *name0, char *ptx_image0);
};
struct PtxStore
{
    PtxRecord ptx_record[RC_NKMODULEMAX];
    int used_count;
    /*CONSTRUCTOR*/
    PtxStore(void);
    /*METHODS*/
    PtxRecord *add(char *name0, char *ptx_image0);
    PtxRecord *query(char *name0);
    void         print(int n);
    // Never remove items.
};
//********************************************************************
//***  Class Name: "ClientModule"
//***  Description:
//********************************************************************
struct ClientModule
{
    int    index;  
    int    id;     /*  that consists of the virtual one, returned from server. */
    int    valid;  /* 1=>alive, 0=>cache out, -1=>init val. */
    PtxRecord *ptx_data;
    time_t sent_time;
    /*CONSTRUCTOR*/
    ClientModule(void);
    /*METHODS*/
    void validate() { valid = 1; }
    void invalidate() { valid = 0; }
    
    int  isValid(void);
    int  isInvalid(void);
    int  isAlive() {
	if( (time(NULL) - sent_time) < RC_CLIENT_CACHE_LIFETIME ) {
	    return 1;
	} else {
	    return 0;
	}
    }
};

//********************************************************************
//***  Class Name: "RCmappedMem"
//***  Description:
//********************************************************************
struct RCmappedMem
{
    void        *pHost;
    void        *pDevice;
    int          size;
    RCmappedMem *prev;
    RCmappedMem *next;
};

//********************************************************************
//***  Class Name: "RCstreamArray"
//***  Description:
//********************************************************************
struct RCstreamArray
{
    cudaStream_t s[RC_NREDUNDANCYMAX];
    RCstreamArray *prev;
    RCstreamArray *next;
};
//********************************************************************
//***  Class Name: "RCeventArray"
//***  Description:
//********************************************************************
struct RCeventArray
{
    cudaEvent_t e[RC_NREDUNDANCYMAX];
    RCeventArray *prev;
    RCeventArray *next;
};
//********************************************************************
//***  Class Name: "RCcuarrayArray"
//***  Description:
//********************************************************************
struct RCcuarrayArray
{
    cudaArray *ap[RC_NREDUNDANCYMAX];
    RCcuarrayArray *prev;
    RCcuarrayArray *next;
};
//********************************************************************
//***  Class Name: "RCuva_t"
//***  Description:
//********************************************************************
struct RCuva
{
    void  *adr[RC_NREDUNDANCYMAX];
    int    devid;
    int    size;
    RCuva *prev;
    RCuva *next;
};

//**
//** define FT mode for ClientState, VirtualDevice, and Physical one.
//**
enum FTmode { FT_NONE    = 0,   //-> No any Redundant or fault toleant behavior.
	      FT_ERRSTAT = 1,   //-> count errors only, not corrected.
	      FT_BYCPY   = 2,   //-> redundant data is verified every cudaMemcpyD2H.
	      FT_BYTIMER = 3,   //-> redundant data is verified specified period.
	      //;
	      FT_SPARE   = 4,   //-> spare device
	      FT_BROKEN  = 5,   //-> broken and replaced device
	      FT_IGNORE  = 6,
	      //;
	      FT_UNDEF   = 999 };  //-> (Initial value, actually unused.)
//********************************************************************
//***  Class Name: "RCServer"
//***  Description:
//***      - Physical GPU Device Class.
//********************************************************************
struct RCServer
{
    int         id;   // index for each redundant server.
    int         cid;  // id of a server given by -c option to dscudasvr.
                      // clients specify the server using this num preceded
                      // by an IP address & colon, e.g.,
                      // export DSCUDA_SERVER="192.168.1.123:2"
    char        ip[512];      // IP address. ex. "192.168.0.92"
    char        hostname[64]; // Hostname.   ex. "titan01"
    int         uniq;         // unique in all RCServer including svrCand[].
    
    BkupMemList memlist;      // GPU global memory mirroring region.
    HistList    reclist;      // GPU CUDA function called history.
    int         rec_en;
    
    int        *d_faultconf;  //

    enum FTmode ft_mode;      // Fault Tolerant mode.
    int         stat_error;   // Error  statics in redundant calculation.
    int         stat_correct; // Corrct statics in redundant calculation.
    
    CLIENT     *Clnt;         // RPC client pointer.

    /*CONSTRUCTOR*/
    RCServer();

    /*CUDA KERNEL MODULES MANAGEMENT*/
    ClientModule modulelist[RC_NKMODULEMAX];
    int         loadModule(unsigned int ipaddr, pid_t pid, char *modulename,
			   char *modulebuf, int module_index);
    int         findModuleOpen(void);
    int         queryModuleID(int module_index);
    void        invalidateModuleCache(void);
    /*RECORDING HISTORY*/
    int         isRecordOn(void);
    int         setRecord(int rec_en0); // return current rec_en.

    /*SETTER*/
    void setIP(const char *ip0);
    void setID(int id0);
    void setCID(int cid0);
    void setCID(char *cir_sz);
    void setUNIQ(int uniq0);
    void setFTMODE(enum FTmode ft_mode0);
    /*METHODS*/
    int  setupConnection(void); // 0:success, -1:fail.
    void dupServer(RCServer *dup);

    cudaError_t cudaMalloc(void **d_ptr, size_t, struct rpc_err *);
    cudaError_t cudaFree(void *d_ptr, struct rpc_err *);
    cudaError_t cudaMemcpyH2D(void *v_ptr, const void *h_ptr, size_t, struct rpc_err *);
    cudaError_t cudaMemcpyD2H(void *h_ptr, const void *v_ptr, size_t, struct rpc_err *);
    cudaError_t cudaThreadSynchronize( struct rpc_err *);

    void        launchKernel(int moduleid, int kid, char *kname, RCdim3 gdim,
			     RCdim3 bdim, RCsize smemsize, RCstream stream,
			     RCargs args, struct rpc_err *);

    //<--- Migration series
    void rpcErrorHook(struct rpc_err *err);
    void migrateServer(RCServer *spare);
    void migrateReallocAllRegions(void);
    void migrateDeliverAllRegions(void);
    void migrateDeliverAllModules(void);
    void migrateRebuildModulelist(void);

    void collectEntireRegions(void);
};  /* "RC" means "Remote Cuda" which is old name of DS-CUDA  */

//*************************************************
//***  Class Name: "ServerArray"
//***  Description:
//***      - The Group of Physical GPU Device Class.
//*************************************************
struct ServerArray
{
    int num;                      /* # of server candidates.         */
    RCServer svr[RC_NVDEVMAX];  /* a list of candidates of server. */
    /*CONSTRUCTOR*/
    ServerArray(void);
    //~ServerArray(void);
    /*METHODS*/
    int add(const char *ip, int ndev, const char *hname);
    int add(RCServer *svrptr);
//    void      removeArray(ServerArray *sub);
    RCServer *findSpareOne(void);
    RCServer *findBrokenOne(void);
    void      captureEnv(char *env, enum FTmode ft_mode0);
    void      print(void);
};


typedef enum VdevConf_e {
    VDEV_MONO = 0, //VirDev.nredundancy == 1
    VDEV_POLY = 1, //                   >= 2
    VDEV_INVALID = 8,
    VDEV_UNKNOWN = 9
} VdevConf;

//*************************************************
//***  Class Name: "VirDev"
//***  Description:
//***      - Virtualized GPU Device class.
//*************************************************
struct VirDev
{
    int         id;
    RCServer    server[RC_NREDUNDANCYMAX]; //Physical Device array.
    int         nredundancy;               //Redundant count

    enum FTmode ft_mode;
    VdevConf    conf;                      //{VDEV_MONO, VDEV_POLY}
    char        info[16];                  //{MONO, POLY(nredundancy)}
                                           /*** CHECKPOINTING ***/
    BkupMemList memlist;              //part of Checkpoint data.
    HistList    reclist;
    int         rec_en;

    /*CONSTRUCTOR*/
    VirDev(void);
    /*CUDA kernel modules management*/
    ClientModule modulelist[RC_NKMODULEMAX];
    int         loadModule(char *name, char *strdata);
    int         findModuleOpen(void);
    void        invalidateAllModuleCache(void);
    void        printModuleList(void);
    /*HISTORY RECORDING*/
    int         isRecordOn(void);
    int         setRecord(int rec_en0); // return current rec_en.
    
    void        setFaultMode(enum FTmode fault_mode);
    void        setConfInfo(int redun);
    cudaError_t cudaMalloc(void **h_ptr, size_t size);
    cudaError_t cudaFree(void *d_ptr);
    cudaError_t cudaMemcpyH2D(void *d_ptr, const void *h_ptr, size_t size);
    cudaError_t cudaMemcpyD2H(void *h_ptr, const void *d_ptr, size_t size);
    cudaError_t cudaThreadSynchronize(void);

    void  launchKernel(int module_index, int kid, char *kname, RCdim3 gdim,
		       RCdim3 bdim, RCsize smemsize, RCstream stream, RCargs args);
    /*CP*/
    void  remallocRegionsGPU(int num_svr); //cudaMemcpyD2H-all
    void  collectEntireRegions(void);
    int   verifyEntireRegions(void);
    void  updateMemlist(int svr_id);
    void  restoreMemlist(void);
    void  clearReclist(void);
}; // struct VirDev

//*******************************************************************************
//***  Class Name: "ClientState_t"                                              *
//***  Description:                                                             *
//***      - DS-CUDA Client Status class.                                       *
//*******************************************************************************
void *periodicCheckpoint(void *arg);

struct ClientState
{
private:
    pthread_t tid;        /* thread ID of Checkpointing */
    //static void *periodicCheckpoint(void *arg);
    void setFaultTolerantMode(void);
public:
    //static int    Nvdev;             // # of virtual devices available.
    //static VirDev Vdev[RC_NVDEVMAX]; // list of virtual devices.
    int    Nvdev;             // # of virtual devices available.
    VirDev Vdev[RC_NVDEVMAX]; // list of virtual devices.

    enum FTmode  ft_mode;
                              /*** Static Information ***/
    unsigned int ip_addr;     // Client IP address.
    time_t       start_time;  // Clinet start time.
    time_t       stop_time;   // Client stop time.

    /* Mode */
    int use_ibv;             /* 1:IBV, 0:RPC   */
    int autoverb;           /* {0, 1, 2, 3} Redundant calculation level */
    int migration;
    int cp_period;          // Period of checkpoint [sec] defined by DSCUDA_CP_PERIOD
    int daemon;
    int historical_calling;

    //** <-- DS-CUDA client log/err output files.
    FILE *dscuda_stdout;     // log-file descriptor.
    FILE *dscuda_stderr;     // err-file descriptor.
    char dslog_filename[80]; // ex.) "c20141224_235901.dslog", 'c' means clnt.
    char dserr_filename[80]; // ex.) "c20141224_235901.dserr", 'c' means clnt.
    //** --> DS-CUDA client log/err output files.
    
    /*CONSRUCTOR, DESTRUCTOR*/
    ClientState(void);
    ~ClientState(void);
    
    /*METHODS*/
    void setIpAddress(unsigned int val) { ip_addr = val; }
    unsigned int getIpAddress() { return ip_addr; }
    void initVirtualDeviceList(void); // Update the list of virtual devices
    
    void useIbv() { use_ibv = 1; }
    void useRpc() { use_ibv = 0; }
    int  isIbv()  { return use_ibv;       }
    int  isRpc()  { return (1 - use_ibv); }

    void   setAutoVerb(int val=1)  { autoverb = val; }
    void unsetAutoVerb() { autoverb = 0; }
    int  isAutoVerb(void)    { return autoverb; }

     void   setHistoCalling() { historical_calling = 1; }
    void unsetHistoCalling() { historical_calling = 0; }
    int  isHistoCalling()   { return historical_calling; }

    void setMigrateDevice(int val=1) { migration = val; }
    void unsetMigrateDevice() { migration = 0; }
    int  getMigrateDevice() { return migration; }
    /*CHECKPOINT*/
    void  collectEntireRegions(void);
    int   verifyEntireRegions(void);
    void  udpateMemlist(void);
}; // ClientState

extern ClientState St;
extern PtxStore    Ptx;

extern const char *DEFAULT_SVRIP;

extern ServerArray SvrSpare;   // Alternative GPU Device Servers.
extern ServerArray SvrIgnore;  // Forbidden GPU Device Servers.

extern int    Vdevid[RC_NPTHREADMAX];
//extern struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
extern void (*errorHandler)(void *arg);

extern void *errorHandlerArg;
//extern ClientModule  CltModulelist[RC_NKMODULEMAX];
extern RCmappedMem    *RCmappedMemListTop;
extern RCmappedMem    *RCmappedMemListTail;

void printModuleList(void);
void printVirtualDeviceList();
int  requestDaemonForDevice(char *ipaddr, int devid, int useibv);
int  vdevidIndex(void);

int  dscudaLoadModule(char *srcname, char *strdata);
void
checkResult(void *rp, RCServer &sp);
cudaError_t cudaSetDevice_clnt( int device, int errcheck );

cudaError_t
dscudaMemcpyToSymbolH2D(int moduleid, char *symbol, const void *src,
			size_t count, size_t offset, int vdevid, int raidid);
cudaError_t
dscudaMemcpyFromSymbolD2H(int moduleid, void **dstbuf, char *symbol,
			  size_t count, size_t offset, int vdevid, int raidid);
cudaError_t
dscudaMemcpyToSymbolD2D(int moduleid, char *symbol, const void *src,
			size_t count, size_t offset, int vdevid, int raidid);
cudaError_t
dscudaMemcpyFromSymbolD2D(int moduleid, void *dstadr, char *symbol,
			  size_t count, size_t offset, int vdevid, int raidid);
cudaError_t
dscudaMemcpyToSymbolAsyncH2D(int moduleid, char *symbol, const void *src,
			     size_t count, size_t offset, RCstream stream, int vdevid, int raidid);
cudaError_t
dscudaMemcpyToSymbolAsyncD2D(int moduleid, char *symbol, const void *src,
			     size_t count, size_t offset, RCstream stream, int vdevid, int raidid);
cudaError_t
dscudaMemcpyFromSymbolAsyncD2H(int moduleid, void **dstbuf, char *symbol,
			       size_t count, size_t offset, RCstream stream, int vdevid, int raidid);
cudaError_t
dscudaMemcpyFromSymbolAsyncD2D(int moduleid, void *dstadr, char *symbol,
			       size_t count, size_t offset, RCstream stream, int vdevid, int raidid);
RCstreamArray *RCstreamArrayQuery(cudaStream_t stream0);

void RCcuarrayArrayRegister(cudaArray **cuarrays);
void RCcuarrayArrayUnregister(cudaArray *cuarray0);
RCcuarrayArray *RCcuarrayArrayQuery(cudaArray *cuarray0);
void RCmappedMemRegister(void *pHost, void* pDevice, size_t size);
RCmappedMem *RCmappedMemQuery(void *pHost);
void RCmappedMemUnregister(void *pHost);
RCeventArray *RCeventArrayQuery(cudaEvent_t event0);
void RCeventArrayRegister(cudaEvent_t *events);
void RCeventArrayUnregister(cudaEvent_t event0);

void *dscudaUvaOfAdr(void *adr, int devid);
void *dscudaAdrOfUva(void *adr);
int   dscudaDevidOfUva(void *adr);
void
replaceBrokenServer(RCServer *broken, RCServer *spare);

cudaError_t
dscudaBindTextureWrapper(int *moduleid, char *texname,
			 size_t *offset,
			 const struct textureReference *tex,
			 const void *devPtr,
			 const struct cudaChannelFormatDesc *desc,
			 size_t size = UINT_MAX);

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureWrapper(int *moduleid, char *texname,
                                    size_t *offset,
                                    const struct texture<T, dim, readMode> &tex,
                                    const void *devPtr,
                                    const struct cudaChannelFormatDesc &desc,
                                    size_t size = UINT_MAX)
{
  return cudaBindTexture(offset, &tex, devPtr, &desc, size);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureWrapper(int *moduleid, char *texname,
                                    size_t *offset,
                                    const struct texture<T, dim, readMode> &tex,
                                    const void *devPtr,
                                    size_t size = UINT_MAX)
{
  return cudaBindTexture(offset, tex, devPtr, tex.channelDesc, size);
}
cudaError_t
dscudaMemcpyToSymbolWrapper(int *moduleid, const char *symbol, const void *src,
                                       size_t count, size_t offset = 0,
                                       enum cudaMemcpyKind kind = cudaMemcpyHostToDevice);
cudaError_t dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                                      size_t *offset,
                                      const struct textureReference *tex,
                                      const void *devPtr,
                                      const struct cudaChannelFormatDesc *desc,
                                      size_t width, size_t height, size_t pitch);

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                                      size_t *offset,
                                      const struct texture<T, dim, readMode> &tex,
                                      const void *devPtr,
                                      const struct cudaChannelFormatDesc &desc,
                                      size_t width, size_t height, size_t pitch)
{
    return dscudaBindTexture2DWrapper(moduleid, texname,
                                     offset, &tex, devPtr, &desc, width, height, pitch);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                                      size_t *offset,
                                      const struct texture<T, dim, readMode> &tex,
                                      const void *devPtr,
                                      size_t width, size_t height, size_t pitch)
{
    return dscudaBindTexture2DWrapper(moduleid, texname,
                                     offset, &tex, devPtr, &tex.channelDesc, width, height, pitch);
}
cudaError_t dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                                           const struct textureReference *tex,
                                           const struct cudaArray * array,
                                           const struct cudaChannelFormatDesc *desc);

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                                           const struct texture<T, dim, readMode> &tex,
                                           const struct cudaArray * array,
                                           const struct cudaChannelFormatDesc & desc)
{
    return dscudaBindTextureToArrayWrapper(moduleid, texname, &tex, array, &desc);
}

template<class T, int dim, enum cudaTextureReadMode readMode>
cudaError_t dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                                           const struct texture<T, dim, readMode> &tex,
                                           const struct cudaArray * array)
{
    struct cudaChannelFormatDesc desc;
    cudaError_t err = cudaGetChannelDesc(&desc, array);
    return err == cudaSuccess ? dscudaBindTextureToArrayWrapper(moduleid, texname, &tex, array, &desc) : err;
}
int dscudaNredundancy(void);
int dscudaRemoteCallType(void);
void dscudaSetErrorHandler(void (*handler)(void *), void *handler_arg);
void
dscudaGetMangledFunctionName(char *name, const char *funcif, const char *ptxdata);
cudaError_t
dscudaMemcpyFromSymbolWrapper(int *moduleid, void *dst, const char *symbol,
			      size_t count, size_t offset = 0,
			      enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
cudaError_t
dscudaFuncGetAttributesWrapper(int *moduleid, struct cudaFuncAttributes *attr, const char *func);
cudaError_t
dscudaMemcpyToSymbolAsyncWrapper(int *moduleid, const char *symbol, const void *src,
				 size_t count, size_t offset = 0,
				 enum cudaMemcpyKind kind = cudaMemcpyHostToDevice, cudaStream_t stream = 0);
cudaError_t
dscudaMemcpyFromSymbolAsyncWrapper(int *moduleid, void *dst, const char *symbol,
				   size_t count, size_t offset = 0,
				   enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost, cudaStream_t stream = 0);
void
rpcDscudaLaunchKernelWrapper(int moduleid, int kid, char *kname,
			     RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
			     RCargs args);

extern pthread_mutex_t cudaMemcpyD2H_mutex;
extern pthread_mutex_t cudaMemcpyH2D_mutex;
extern pthread_mutex_t cudaKernelRun_mutex;

#endif //__LIBDSCUDA_H__
