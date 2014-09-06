//                             -*- Mode: C++ -*-
// Filename         : libdscuda.h
// Description      : DS-CUDA client node common(IBV and RPC) library header.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-06 12:06:32
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef __LIBDSCUDA_H__
#define __LIBDSCUDA_H__
#include "sockutil.h"

struct RCServer;
typedef RCServer RCServer_t;
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
typedef struct BkupMem_t {
    void  *v_region;        // UVA, Search index, and also Virtual device address.
    void  *d_region;        //!UVA, server device memory space.
    void  *h_region;        //
    int    size;            // in Byte.
    int    update_rdy;      // 1:"*dst" has valid data, 0:invalid.
    struct BkupMem_t *next; // For double-linked-list prev.
    struct BkupMem_t *prev; // For double-linked-list next.
    /*constructor/destructor.*/
    BkupMem_t( void );
    //--- methods
    void   init(void *uva_ptr, void *d_ptr, int isize);
    int    isHead(void);
    int    isTail(void);
    int    calcSum(void);
    void  *translateAddrVtoD(const void *v_ptr);
    void  *translateAddrVtoH(const void *v_ptr);
} BkupMem;

//********************************************************************
//***  Class Name: "BkupMemList_t"
//***  Description:
//***      - 
//********************************************************************
typedef struct BkupMemList_t {
public:
    BkupMem *head;        /* pointer to 1st  BkupMem */
    BkupMem *tail;        /* pointer to last BkupMem */
    int      length;       /* Counts of allocated memory region */
    long     total_size;   /* Total size of backuped memory in Byte */
    //--- construct/destruct
    BkupMemList_t(void);
    ~BkupMemList_t(void);
    //--- methods ---------------
    void     print(void);
    BkupMem* query(void *uva_ptr);
    void     add(void *uva_ptr, void *dst, int size);
    void     remove( void *dst );        // verbAllocatedMemUnregister()
    int      isEmpty(void);
    int      getLen(void);
    long     getTotalSize(void); // get total size of allocated memory.
    int      countRegion(void);
    int      checkSumRegion(void *targ, int size);
    void*    queryHostPtr(const void *v_ptr);
    void*    queryDevicePtr(const void *v_ptr);
    void     restructDeviceRegion(void);              /* ReLoad backups */
} BkupMemList;

//********************************************************************
//***  Class Name: "HistRec_t"
//***  Description:
//***      - Recording the sequential CUDA-call.
//********************************************************************
typedef struct HistRec_t {
    int      funcID;   // Recorded cuda*() function.
    void    *args;     // And its arguments.
    int      dev_id;   // The Device ID, set by last cudaSetDevice().
} HistRec;

//********************************************************************
//***  Class Name: "HistRecList_t"
//***  Description:
//***      - 
//********************************************************************
typedef struct HistRecList_t {
    HistRec *histrec;
    int      length;    /* # of recorded function calls to be recalled */
    int      max_len;   /* Upper bound of "verbHistNum", extensible */
    // Constructor.
    HistRecList_t(void);
    //
    void add(int funcID, void *argp); /* Add */
    void clear(void);           /* Clear */
    void print(void);           /* Print to stdout */
    int  recall(void);          /* Recall */
private:
    static const int EXTEND_LEN = 32; // Size of growing of "max_len"
    int      recall_flag;
    void     setRecallFlag(void);
    void     clrRecallFlag(void);
    void     extendLen(void);

} HistRecList;

//********************************************************************
//***  Class Name: "ClientModule_t"
//***  Description:
//***      - 
//********************************************************************
typedef struct ClientModule_t {
    int    valid;   /* 1=>alive, 0=>cache out, -1=>init val. */
    int    vdevid;  /* the virtual device the module is loaded into. */
    int    id[RC_NREDUNDANCYMAX]; /*  that consists of the virtual one, returned from server. */
    char   name[RC_KMODULENAMELEN];
    char   ptx_image[RC_KMODULEIMAGELEN]; /* needed for RecallHist(). */
    time_t sent_time;

    void validate() { valid = 1; }
    void invalidate() { valid = 0; }
    int  isValid()  {
	if (valid<-1 || valid>1) {
	    fprintf(stderr, "Unexpected error. %s:%d\n", __FILE__, __LINE__);
	    exit(1);
	}
	else if (valid==1) { return 1; }
	else { return 0; }
    }
    int  isInvalid()  {
	if (valid<-1 || valid>1) {
	    fprintf(stderr, "Unexpected error. %s:%d\n", __FILE__, __LINE__);
	    exit(1);
	}
	else if (valid==1) { return 0;}
	else { return 1; }
    }
    void setPtxPath(char *ptxpath) {
	strncpy(name, ptxpath, RC_KMODULENAMELEN);
    }
    void setPtxImage(char *ptxstr) {
	strncpy(ptx_image, ptxstr, RC_KMODULEIMAGELEN);
    }
    int isAlive() {
	if( (time(NULL) - sent_time) < RC_CLIENT_CACHE_LIFETIME ) {
	    return 1;
	} else {
	    return 0;
	}
    }
    /*
    int isTimeout() {
	
    }
    */
    ClientModule_t(void);
} ClientModule;

//********************************************************************
//***  Class Name: "RCmappedMem_t"
//***  Description:
//********************************************************************
typedef struct RCmappedMem_t {
    void *pHost;
    void *pDevice;
    int   size;
    RCmappedMem_t *prev;
    RCmappedMem_t *next;
} RCmappedMem;

//********************************************************************
//***  Class Name: "RCstreamArray_t"
//***  Description:
//********************************************************************
typedef struct RCstreamArray_t {
    cudaStream_t s[RC_NREDUNDANCYMAX];
    RCstreamArray_t *prev;
    RCstreamArray_t *next;
} RCstreamArray;

//********************************************************************
//***  Class Name: "RCeventArray_t"
//***  Description:
//********************************************************************
typedef struct RCeventArray_t {
    cudaEvent_t e[RC_NREDUNDANCYMAX];
    RCeventArray_t *prev;
    RCeventArray_t *next;
} RCeventArray;

//********************************************************************
//***  Class Name: "RCcuarrayArray_t"
//***  Description:
//********************************************************************
typedef struct RCcuarrayArray_t {
    cudaArray *ap[RC_NREDUNDANCYMAX];
    RCcuarrayArray_t *prev;
    RCcuarrayArray_t *next;
} RCcuarrayArray;

//********************************************************************
//***  Class Name: "RCuva_t"
//***  Description:
//********************************************************************
typedef struct RCuva_t {
    void    *adr[RC_NREDUNDANCYMAX];
    int      devid;
    int      size;
    RCuva_t *prev;
    RCuva_t *next;
} RCuva;

//********************************************************************
//***  Class Name: "RCServer"
//***  Description:
//***      - Physical GPU Device Class.
//********************************************************************
struct RCServer {
    int         id;   // index for each redundant server.
    int         cid;  // id of a server given by -c option to dscudasvr.
                      // clients specify the server using this num preceded
                      // by an IP address & colon, e.g.,
                      // export DSCUDA_SERVER="192.168.1.123:2"
    char        ip[512];      // ex. "192.168.0.92"
    char        hostname[64]; // ex. "titan01"
    int         uniq;         // unique in all RCServer_t including svrCand[].
    
    BkupMemList memlist;  // GPU global memory mirroring region.
    HistRecList reclist;  // GPU CUDA function called history.
    
    int        *d_faultconf;  //

    int         stat_error; // Error and Fault statics results.
    int         stat_correct;

    CLIENT     *Clnt;         // RPC client

    void setupConnection(void);
    void dupServer(RCServer_t *dup);


    cudaError_t cudaMalloc(void **d_ptr, size_t size);
    cudaError_t cudaMemcpyH2D(void *d_ptr, const void *h_ptr, size_t size);
    cudaError_t cudaMemcpyD2H(void *h_ptr, const void *v_ptr, size_t size);
    cudaError_t cudaFree(void *d_ptr);
    //<--- Migration series
    void migrateServer(RCServer_t *newone, RCServer_t *broken);
    void migrateReallocAllRegions(void);
    /*CONSTRUCTOR*/
    RCServer();
};  /* "RC" means "Remote Cuda" which is old name of DS-CUDA  */

//*************************************************
//***  Class Name: "SvrList"
//***  Description:
//***      - The Group of Physical GPU Device Class.
//*************************************************
typedef struct SvrList {
    int num;                      /* # of server candidates.         */
    RCServer_t svr[RC_NVDEVMAX];  /* a list of candidates of server. */
    /* methods */
    int cat( const char *ipaddr, int ndev, const char *hname ) {
	if ( num >= (RC_NVDEVMAX - 1) ) {
	    fprintf(stderr, "(+_+) Too many DS-CUDA daemons, exceeds RC_NVDEVMAX(=%d)\n", RC_NVDEVMAX);
	    exit( EXIT_FAILURE );
	}
	strcpy( svr[num].ip, ipaddr );
	svr[num].cid  = ndev;
	svr[num].uniq = RC_UNIQ_CANDBASE + num;
	strcpy( svr[num].hostname, hname );
	
	num++;
	return 0;
    }
} SvrList_t;

typedef enum {
    //* Define basic behavior of Fault tolerant functions.
    FT_PLAIN = 0,  // DSCUDA_AUTOVERB=0, DSCUDA_MIGRATION=0.
    FT_REDUN = 1,  // DSCUDA_AUTOVERB=1, DSCUDA_MIGRATION=0.
    FT_MIGRA = 2,  // DSCUDA_AUTOVERB=0, DSCUDA_MIGRATION=1.
    FT_BOTH  = 3   // DSCUDA_AUTOVERB=1, DSCUDA_MIGRATION=1.
} FtMode;

typedef enum VdevConf_e{
    VDEV_MONO = 0, //Vdev_t.nredundancy == 1
    VDEV_POLY = 1, //                   >= 2
    VDEV_INVALID = 8,
    VDEV_UNKNOWN = 9
} VdevConf;

//*************************************************
//***  Class Name: "VirDev_t"
//***  Description:
//***      - Virtualized GPU Device class.
//*************************************************
typedef struct VirDev_t {
    int         id;
    RCServer_t  server[RC_NREDUNDANCYMAX]; //Physical Device array.
    int         nredundancy;               //Redundant count

    FtMode      ft_mode;
    VdevConf    conf;                      //{VDEV_MONO, VDEV_POLY}
    char        info[16];                  //{MONO, POLY(nredundancy)}
                                           /*** CHECKPOINTING ***/
    BkupMemList memlist;              //part of Checkpoint data.
    HistRecList reclist;

    cudaError_t cudaMalloc(void **h_ptr, size_t size);
    cudaError_t cudaMemcpyH2D(void *d_ptr, const void *h_ptr, size_t size);
    cudaError_t cudaMemcpyD2H(void *h_ptr, const void *d_ptr, size_t size);
    cudaError_t cudaFree(void *d_ptr);
    
    void remallocRegionsGPU(int num_svr);
} Vdev_t;

/******************************************/
/* Client Application Status/Information. */
/******************************************/

typedef enum {
    ORIGIN      = 0,
    INITIALIZED,
    CUDA_CALLED
} ClntInitStat;

//*************************************************
//***  Class Name: "ClientState_t"
//***  Description:
//***      - DS-CUDA Client Status class.
//*************************************************
void *periodicCheckpoint(void *arg);
struct ClientState_t {
private:
    pthread_t tid;        /* thread ID of Checkpointing */
    //static void *periodicCheckpoint(void *arg);
    void initEnv(void);
    void setFaultTolerantMode(void);
public:
    //static int    Nvdev;             // # of virtual devices available.
    //static Vdev_t Vdev[RC_NVDEVMAX]; // list of virtual devices.
    int    Nvdev;             // # of virtual devices available.
    Vdev_t Vdev[RC_NVDEVMAX]; // list of virtual devices.


    FtMode       ft_mode;
    ClntInitStat init_stat;
    char *getFtModeString(void);    
                              /*** Static Information ***/
    unsigned int ip_addr;     // Client IP address.
    time_t       start_time;  // Clinet start time.
    time_t       stop_time;   // Client stop time.

    /* Mode */
    int use_ibv;             /* 1:IBV, 0:RPC   */
    int autoverb;           /* 1:Redundant calculation */
    int migration;
    int daemon;
    int historical_calling;
    
    ClientState_t(void);
    ~ClientState_t(void);
    
    void setIpAddress(unsigned int val) { ip_addr = val; }
    unsigned int getIpAddress() { return ip_addr; }
    
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

//    void initProgress(ClntInitStat stat);
//    void cudaCalled(void) { initProgress( CUDA_CALLED ); }
};
extern struct ClientState_t St;
//extern int    ClientState_t::Nvdev;
//extern Vdev_t ClientState_t::Vdev[RC_NVDEVMAX];   // list of virtual devices.


extern const char *DEFAULT_SVRIP;
//extern RCServer_t svrCand[RC_NVDEVMAX];
//extern RCServer_t svrSpare[RC_NVDEVMAX];
extern SvrList_t SvrSpare;
//extern RCServer_t svrBroken[RC_NVDEVMAX];
extern int    Vdevid[RC_NPTHREADMAX];
//extern struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
extern void (*errorHandler)(void *arg);

extern void *errorHandlerArg;
extern ClientModule  CltModulelist[RC_NKMODULEMAX]; /* is Singleton.*/
extern RCmappedMem    *RCmappedMemListTop;
extern RCmappedMem    *RCmappedMemListTail;

/* <-- Redundant APIs */
void dscudaRecordHistOn(void);  // add by oikawa
void dscudaRecordHistOff(void); // add by oikawa
void dscudaAutoVerbOff(void);
void dscudaAutoVerbOn(void);
/* --> Redundant APIs */
void printModuleList(void);
void printVirtualDeviceList();
void invalidateModuleCache(void);
int  requestDaemonForDevice(char *ipaddr, int devid, int useibv);
int  vdevidIndex(void);
void setupConnection(int idev, RCServer_t *sp);
int  dscudaLoadModuleLocal(unsigned int ipaddr, pid_t pid, char *modulename,
			   char *modulebuf, int vdevid, int raidid);
int *dscudaLoadModule(char *srcname, char *strdata);
void
checkResult(void *rp, RCServer_t &sp);
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
replaceBrokenServer(RCServer_t *broken, RCServer_t *spare);

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
rpcDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
			     RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
			     RCargs args);

extern pthread_mutex_t cudaMemcpyD2H_mutex;
extern pthread_mutex_t cudaMemcpyH2D_mutex;
extern pthread_mutex_t cudaKernelRun_mutex;

#endif //__LIBDSCUDA_H__
