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
struct CudaSetDeviceArgs {
    int      device;
};
struct CudaMallocArgs {
    void*    devPtr;
    size_t   size;
    CudaMallocArgs(void) { devPtr = NULL, size = 0; }
    CudaMallocArgs( void *ptr, size_t sz ) { devPtr = ptr; size = sz; }
};
struct CudaMemcpyArgs {
    void*    dst;
    void*    src;
    size_t   count;
    cudaMemcpyKind kind;
    CudaMemcpyArgs( void ) { dst = src = NULL; count = 0; }
    CudaMemcpyArgs( void *d, void *s, size_t c, cudaMemcpyKind k ) {
	dst = d; src = s; count = c; kind = k;
    }
};
struct CudaMemcpyToSymbolArgs {
    int     *moduleid;
    char    *symbol;
    void    *src;
    size_t   count;
    size_t   offset;
    cudaMemcpyKind kind;
};
struct CudaFreeArgs {
    void    *devPtr;
};
struct CudaLoadModuleArgs {
    char    *name;
    char    *strdata;
};
struct CudaRpcLaunchKernelArgs {
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
struct BkupMem {
public:
    BkupMem* prev;            // For double-linked-list next.
    BkupMem* next;            // For double-linked-list prev.
    void*    v_region;        // UVA, Search index, and also Virtual device address.
    void*    d_region;        //!UVA, server device memory space.
    void*    h_region;        //
    size_t   size;            // in Byte.
    int      update_rdy;      // 1:"*dst" has valid data, 0:invalid.
    /*constructor/destructor.*/
             BkupMem           (void);
    //--- methods
    bool     isHead            (void);
    bool     isTail            (void);
    void     init              (void *uva_ptr, void *d_ptr, int isize);
    uint32_t calcChecksum      (void);
    void*    translateAddrVtoD (const void *v_ptr);
    void*    translateAddrVtoH (const void *v_ptr);
    //
    cudaError_t memcpyD2H( const void*, size_t, struct rpc_err*, int, CLIENT* );
private:
};
//********************************************************************
//***  Class Name: "BkupMemList"
//***  Description:
//***      - 
//********************************************************************
struct BkupMemList {
public:
    //--- construct/destruct
             BkupMemList(void);
             ~BkupMemList(void);
    //--- methods ---------------
    void     print         (void);
    BkupMem* headPtr       (void);
    BkupMem* query         (const void *v_ptr);
    void*    queryHostPtr  (const void *v_ptr);
    void*    queryDevicePtr(const void *v_ptr);
    void     append        (void *uva_ptr, void *dst, int size);
    void     remove        (void *uva_ptr);        // verbAllocatedMemUnregister()
    int      getLen        (void);
    long     getTotalSize  (void); // get total size of allocated memory.
    int      countRegion   (void);
    void     restructDeviceRegion(void);              /* ReLoad backups */
    void     incrAge(void);
    int      getAge(void);
private:
    BkupMem* head;        /* pointer to 1st  BkupMem */
    BkupMem* tail;        /* pointer to last BkupMem */
    int      length;       /* Counts of allocated memory region */    
    long     total_size;   /* Total size of backuped memory in Byte */
    int      age;
    bool     isEmpty(void);
};
//********************************************************************
//***
//*** CUDA API call record stub.
//***
//********************************************************************
enum DSCVMethod { DSCVMethodNone = 0,
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
		  DSCVMethodEnd };
//********************************************************************
//***  Class Name: "HistCell"
//***  Description:
//***      - Recording the sequential CUDA-call.
//********************************************************************
typedef int64_t HistID;
struct HistCell {
    HistID  seq_num;  // unique serial ID.
    int     funcID;   // Recorded cuda*() function.
    void*   args;     // And its arguments.
    int     dev_id;   // The Device ID, set by last cudaSetDevice().
};
//********************************************************************
//***  Class Name: "HistList"
//***  Description: List structure of previously defined "HistCell"
//********************************************************************
struct HistList {
    /*CONSTRUCTOR*/
    HistList(void);
    
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
    //
    //void add(int funcID, void *argp);
    void append(int funcID, void *argp);
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
struct PtxRecord {
    int  valid;   //1:valid, 0:invalid.
    char name[RC_KMODULENAMELEN];
    char ptx_image[RC_KMODULEIMAGELEN]; //!Caution; may be Large size.
    /*CONSTRUCTOR*/
    PtxRecord(void);
    /*METHODS*/
    void invalidate(void);
    void set(char *name0, char *ptx_image0);
};
struct PtxStore {
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
struct ClientModule {
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
struct RCmappedMem {
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
struct RCstreamArray {
    cudaStream_t s[RC_NREDUNDANCYMAX];
    RCstreamArray *prev;
    RCstreamArray *next;
};
//********************************************************************
//***  Class Name: "RCeventArray"
//***  Description:
//********************************************************************
struct RCeventArray {
    cudaEvent_t e[RC_NREDUNDANCYMAX];
    RCeventArray *prev;
    RCeventArray *next;
};
//********************************************************************
//***  Class Name: "RCcuarrayArray"
//***  Description:
//********************************************************************
struct RCcuarrayArray {
    cudaArray *ap[RC_NREDUNDANCYMAX];
    RCcuarrayArray *prev;
    RCcuarrayArray *next;
};
//********************************************************************
//***  Class Name: "RCuva_t"
//***  Description:
//********************************************************************
struct RCuva {
    void  *adr[RC_NREDUNDANCYMAX];
    int    devid;
    int    size;
    RCuva *prev;
    RCuva *next;
};

enum FThealth { hl_INIT    , // <= Default.
		hl_GOOD    , // <= Clean and usable.
		hl_BAD     , // <= Faulted and not to use.
		hl_RECYCLED  // <= Recoverd from "hl_BAD"
};
		 
//**
//** define FT mode for ClientState, VirtualDevice, and Physical one.
//**
enum FTmode { FT_NONE    =0,   //-> No any Redundant or fault toleant behavior.
	      FT_ERRSTAT =1,   //-> count errors only, not corrected.
	      FT_BYCPY   =2,   //-> redundant data is verified every cudaMemcpyD2H.
	      FT_BYTIMER =3,   //-> redundant data is verified specified period.
	      FT_OPTION  =256,
	      FT_UNDEF   =999  //-> (Initial value, actually unused.)
};
struct FToption {   //-*- Static configuration -*-
    bool d2h_simple;   //[0] If "true" then disable all redundant func, lower latency.
    bool d2h_reduncpy; //[1] 
    bool d2h_compare;  //[2] "true": compare data between redundant recieved.
    bool d2h_statics;  //[3] "true": count unmatched or matched.
    bool d2h_rollback; //[4] "true": enable rollback.
    //[5]
    //[6]
    //[7]
    bool cp_periodic;  //[8]
    bool cp_reduncpy;  //[9]
    bool cp_compare;   //[10] "true": compare data between redundant recieved.
    bool cp_statics;   //[11]
    bool cp_rollback;  //[12]
    //[13]
    //[14]
    //[15]
    bool rec_en;       //[16]
    //
    bool gpu_migrate;  //[24]
    //[17...]
};

//********************************************************************
//***  Class Name: "PhyDev"
//***  Description:
//***      - Physical GPU Device Class.
//********************************************************************
struct PhyDev {
public:
    int         id;   // index for each redundant server.
    int         cid;  // id of a server given by -c option to dscudasvr.
                      // clients specify the server using this num preceded
                      // by an IP address & colon, e.g.,
                      // export DSCUDA_SERVER="192.168.1.123:2"
    char        ip[512];      // IP address. ex. "192.168.0.92"
    char        hostname[64]; // Hostname.   ex. "titan01"
    int         uniq;         // unique in all PhyDev including svrCand[].
    
    BkupMemList memlist;      // GPU global memory mirroring region.
    int        *d_faultconf;  //

    //<-- Fault tolerant static configurations.
    FToption    ft;
    FTmode      ft_mode;      // Fault Tolerant mode.
    FThealth    ft_health;
    //--> 
    int         stat_error;   // Error  statics in redundant calculation.
    int         stat_correct; // Corrct statics in redundant calculation.
    
    CLIENT     *Clnt;         // RPC client pointer.

    /*CONSTRUCTOR*/
    PhyDev();

    /*CUDA KERNEL MODULES MANAGEMENT*/
    ClientModule modulelist[RC_NKMODULEMAX];
    int         loadModule(unsigned int ipaddr, pid_t pid, char *modulename,
			   char *modulebuf, int module_index);
    int         findModuleOpen(void);
    int         queryModuleID(int module_index);
    void        invalidateModuleCache(void);
    /*RECORDING HISTORY*/

    /*SETTER*/
    void setIP(const char *ip0);
    void setID(int id0);
    void setCID(int cid0);
    void setCID(char *cir_sz);
    void setUNIQ(int uniq0);
    void setFTMODE(FTmode ft_mode0);
    void setHealth(FThealth cond);
    /*METHODS*/
    int  setupConnection(void); // 0:success, -1:fail.
    void dupServer(PhyDev *dup);

    cudaError_t cudaMalloc(void **d_ptr, size_t, struct rpc_err *);
    cudaError_t cudaFree(void *d_ptr, struct rpc_err *);
    cudaError_t cudaMemcpyH2D(const void *v_ptr, const void *h_ptr, size_t,
			      struct rpc_err *);
    cudaError_t cudaMemcpyD2H(const void *h_ptr, const void *v_ptr, size_t,
			      int flag/*FT*/, struct rpc_err *);
    cudaError_t cudaThreadSynchronize( struct rpc_err *);

    void        launchKernel(int moduleid, int kid, char *kname, RCdim3 gdim,
			     RCdim3 bdim, RCsize smemsize, RCstream stream,
			     RCargs, struct rpc_err *, int );
    //<--- Migration series
    void rpcErrorHook(struct rpc_err *err);
    void migrateServer(PhyDev *spare);
    void migrateReallocAllRegions(void);
    void migrateDeliverAllRegions(void);
    void migrateDeliverAllModules(void);
    void migrateRebuildModulelist(void);
    //--->
    void collectEntireRegions(int flag/*FT*/);
};  /* "RC" means "Remote Cuda" which is old name of DS-CUDA  */

//*************************************************
//***  Class Name: "ServerArray"
//***  Description:
//***      - The Group of Physical GPU Device Class.
//*************************************************
struct ServerArray {
    int      num;               /* # of server candidates.         */
    PhyDev   svr[RC_NVDEVMAX];  /* a list of candidates of server. */
    /*CONSTRUCTOR*/
    ServerArray(void);
    /*METHODS*/
    int      append(const char *ip, int ndev, const char *hname);
    int      append(PhyDev *svrptr);
    PhyDev  *findSpareOne(void);
    PhyDev  *findBrokenOne(void);
    void     captureEnv(char *env, FThealth cond);
    void     print(void);
};

enum VdevConf {  VDEV_MONO    = 0, //VirDev.nredundancy == 1
		 VDEV_POLY    = 1, //                   >= 2
		 VDEV_INVALID = 8,
		 VDEV_UNKNOWN = 9   };
//*************************************************
//***  Class Name: "VirDev"
//***  Description:
//***      - Virtualized GPU Device class.
//*************************************************
struct VirDev {
public:
    int         id;
    PhyDev      server[RC_NREDUNDANCYMAX]; //Physical Device array.
    int         nredundancy;               //Actual redundant devcount.
    
    //<-- Fault tolerant function control 
    FTmode      ft_mode;
    FToption    ft;
    //--> Fault tolerant function control
    
    VdevConf    conf;                      //{VDEV_MONO, VDEV_POLY}
    char        info[16];                  //{MONO, POLY(nredundancy)}
                                           /*** CHECKPOINTING ***/
    BkupMemList memlist;              //part of Checkpoint data.
    HistList    reclist;
    //<--- CUDA API recording ON/OFF dynamically.
public:
    void        recordON(void); 
    void        recordOFF(void);
    bool        isRecording(void);
    void        appendRecord(int funcID, void *argp);
private:
    bool        history_recording;
    //---> CUDA API recording ON/OFF dynamically.
public:
    /*CONSTRUCTOR*/
    VirDev(void);
    /*CUDA kernel modules management*/
    ClientModule modulelist[RC_NKMODULEMAX];
    int         loadModule(char *name, char *strdata);
    int         findModuleOpen(void);
    void        invalidateAllModuleCache(void);
    void        printModuleList(void);
    /*HISTORY RECORDING*/

    
    void        setFaultMode(FTmode fault_mode);
    void        setConfInfo(int redun);
    cudaError_t cudaMalloc(void **h_ptr, size_t size);
    cudaError_t cudaFree(void *d_ptr);
    cudaError_t cudaMemcpyH2D(void *d_ptr, const void *h_ptr, size_t size);
    cudaError_t cudaMemcpyD2H(void *h_ptr, const void *d_ptr, size_t size);
    cudaError_t cudaThreadSynchronize(void);

    void  launchKernel(int, int, char*, RCdim3, RCdim3, RCsize, RCstream, RCargs, int);
    /*CP*/
    void  remallocRegionsGPU(int num_svr); //cudaMemcpyD2H-all
    void  collectEntireRegions(void);
    bool  verifyEntireRegions(void);
    void  updateMemlist(void);
    void  restoreMemlist(void);
    void  clearReclist(void);
}; // struct VirDev

//*******************************************************************************
//***  Class Name: "ClientState_t"                                              *
//***  Description:                                                             *
//***      - DS-CUDA Client Status class.                                       *
//*******************************************************************************
void *periodicCheckpoint(void *arg);

struct ClientState {
public:
    char      dscuda_path[512];
    int       Nvdev;             // # of virtual devices available.
    VirDev    Vdev[RC_NVDEVMAX]; // list of virtual devices.
    FTmode    ft_mode;
    FToption  ft;
    /* Mode */
    bool      use_ibv;             /* true:IBV, false:RPC   */
    void      useIbv(void);
    void      useRpc(void);
    bool      isIbv(void);
    bool      isRpc(void);
    
    int       autoverb;           /* {0, 1, 2, 3} Redundant calculation level */
    int       cp_period;          // Period of checkpoint [sec] defined by DSCUDA_CP_PERIOD
    int       daemon;
    
    bool      rollback_calling;
    void      setRollbackCalling(void);
    void      unsetRollbackCalling(void);
    bool      isRollbackCalling(void);
    
    //<-- Error/Fault Static Information
    time_t    start_time;  // Clinet start time.
    time_t    stop_time;   // Client stop time.

    //** <-- DS-CUDA client log/err output files.
    FILE     *dscuda_stdout;     // log-file descriptor.
    FILE     *dscuda_stderr;     // err-file descriptor.
    FILE     *dscuda_chkpnt;     // err-file descriptor.
    //** --> DS-CUDA client log/err output files.
    
    /*CONSRUCTOR, DESTRUCTOR*/
    ClientState(void);
    ~ClientState(void);
    
    /*METHODS*/
    unsigned  getIpAddress(void);
    void      initVirtualDevice(void); // Update the list of virtual devices

    /*CHECKPOINT*/
    void      collectEntireRegions(void);
    bool      verifyEntireRegions(void);
    void      udpateMemlist(void);
private:
    pthread_t tid;        /* thread ID of Checkpointing */
    unsigned  ip_addr;     // Client IP address.
    //<-- DSCUDA log filename.
    char      dslog_filename[80]; // ex.) "c20141224_235901.dslog", 'c' means clnt.
    char      dserr_filename[80]; // ex.) "c20141224_235901.dserr", 'c' means clnt.
    char      dschp_filename[80]; // ex.) "c20141224_235901.dschp", 'c' means clnt.
    void      setMyIPAddr(unsigned);
    void      configFT(void);
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
int  requestDaemonForDevice(char *ipaddr, int devid, bool useibv);
int  vdevidIndex(void);

int  dscudaLoadModule(char *srcname, char *strdata);
void
checkResult(void *rp, PhyDev &sp);
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
replaceBrokenServer(PhyDev *broken, PhyDev *spare);

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
                                       cudaMemcpyKind kind = cudaMemcpyHostToDevice);
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
			      cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
cudaError_t
dscudaFuncGetAttributesWrapper(int *moduleid, struct cudaFuncAttributes *attr, const char *func);
cudaError_t
dscudaMemcpyToSymbolAsyncWrapper(int *moduleid, const char *symbol, const void *src,
				 size_t count, size_t offset = 0,
				 cudaMemcpyKind kind = cudaMemcpyHostToDevice, cudaStream_t stream = 0);
cudaError_t
dscudaMemcpyFromSymbolAsyncWrapper(int *moduleid, void *dst, const char *symbol,
				   size_t count, size_t offset = 0,
				   cudaMemcpyKind kind = cudaMemcpyDeviceToHost, cudaStream_t stream = 0);
void
rpcDscudaLaunchKernelWrapper(int moduleid, int kid, char *kname,
			     RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
			     RCargs args);

extern pthread_mutex_t cudaMemcpyD2H_mutex;
extern pthread_mutex_t cudaMemcpyH2D_mutex;
extern pthread_mutex_t cudaKernelRun_mutex;
extern pthread_mutex_t cudaElse_mutex;

#endif //__LIBDSCUDA_H__
