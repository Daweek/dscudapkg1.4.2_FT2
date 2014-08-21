//                             -*- Mode: C++ -*-
// Filename         : libdscuda.h
// Description      : DS-CUDA client node common(IBV and RPC) library header.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-08-21 15:30:27
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef __LIBDSCUDA_H__
#define __LIBDSCUDA_H__
#include "sockutil.h"

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

typedef struct RCmappedMem_t {
    void *pHost;
    void *pDevice;
    int   size;
    RCmappedMem_t *prev;
    RCmappedMem_t *next;
} RCmappedMem;

typedef struct RCstreamArray_t {
    cudaStream_t s[RC_NREDUNDANCYMAX];
    RCstreamArray_t *prev;
    RCstreamArray_t *next;
} RCstreamArray;

typedef struct RCeventArray_t {
    cudaEvent_t e[RC_NREDUNDANCYMAX];
    RCeventArray_t *prev;
    RCeventArray_t *next;
} RCeventArray;

typedef struct RCcuarrayArray_t {
    cudaArray *ap[RC_NREDUNDANCYMAX];
    RCcuarrayArray_t *prev;
    RCcuarrayArray_t *next;
} RCcuarrayArray;

typedef struct RCuva_t {
    void    *adr[RC_NREDUNDANCYMAX];
    int      devid;
    int      size;
    RCuva_t *prev;
    RCuva_t *next;
} RCuva;

/**********************************/
/* Virtual GPU device Management. */
/**********************************/
typedef struct RCServer {
    int  id;   // index for each redundant server.
    int  cid;  // id of a server given by -c option to dscudasvr.
               // clients specify the server using this num preceded
               // by an IP address & colon, e.g.,
               // export DSCUDA_SERVER="192.168.1.123:2"
    char ip[512];      // ex. "192.168.0.92"
    char hostname[64]; // ex. "titan01"
    int  uniq; // unique number in all RCServer_t including svrCand[].
    int  *d_faultconf;
    int  errcount;
    RCServer() {
	id = cid = uniq = 0xffff;
	strcpy(ip, "empty");
	strcpy(hostname, "empty");
	errcount = 0;
    }
} RCServer_t;  /* "RC" means "Remote Cuda" which is old name of DS-CUDA  */

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
    VDEV_MONO = 0, //Vdev_t.nredundancy == 1
    VDEV_POLY = 1, //                   >= 2
    VDEV_INVALID = 8,
    VDEV_UNKNOWN = 9
} VdevConf;

typedef struct {
    int        nredundancy;
    RCServer_t server[RC_NREDUNDANCYMAX];
    VdevConf   conf;
    char       info[16]; //{MONO, POLY(nredundancy)}
} Vdev_t;

/******************************************/
/* Client Application Status/Information. */
/******************************************/
typedef enum {
    /*
     * Define basic behavior of Fault tolerant functions.
     */
    FT_PLAIN = 0,  // DSCUDA_AUTOVERB=0, DSCUDA_MIGRATION=0.
    FT_REDUN = 1,  // DSCUDA_AUTOVERB=1, DSCUDA_MIGRATION=0.
    FT_MIGRA = 2,  // DSCUDA_AUTOVERB=0, DSCUDA_MIGRATION=1.
    FT_BOTH  = 3   // DSCUDA_AUTOVERB=1, DSCUDA_MIGRATION=1.
} ClntFtMode;

typedef enum {
    ORIGIN      = 0,
    INITIALIZED,
    CUDA_CALLED
} ClntInitStat;

struct ClientState_t {
private:
    void initEnv( void );
    void setFaultTolerantMode( void );
public:
    ClntFtMode   ft_mode;
    char *getFtModeString(void);    

    ClntInitStat init_stat;
    int    Nvdev;               // # of virtual devices available.
    Vdev_t Vdev[RC_NVDEVMAX];   // a list of virtual devices.

    unsigned int ip_addr;    /* My IP address */
    /* Mode */
    int use_ibv;             /* 1:IBV, 0:RPC   */
    int autoverb;           /* 1:Redundant calculation */
    int migration;
    int daemon;
    int historical_calling;

    void setIpAddress(unsigned int val) { ip_addr = val; }
    unsigned int getIpAddress() { return ip_addr; }
    
    void useIbv() { use_ibv = 1; }
    void useRpc() { use_ibv = 0; }
    int  isIbv()  { return use_ibv;       }
    int  isRpc()  { return (1 - use_ibv); }
    
    void   setAutoVerb(int val=1)  { autoverb = val; }
    void unsetAutoVerb() { autoverb = 0; }
    int  isAutoVerb(void)    { return autoverb; }

    void   setUseDaemon(void) { daemon = 1; }
    void unsetUseDaemon() { daemon = 0; }
    int  isUseDaemon(void)   { return daemon; }

    void   setHistoCalling() { historical_calling = 1; }
    void unsetHistoCalling() { historical_calling = 0; }
    int  isHistoCalling()   { return historical_calling; }

    void setMigrateDevice(int val=1) { migration = val; }
    void unsetMigrateDevice() { migration = 0; }
    int  getMigrateDevice() { return migration; }

    void initProgress( ClntInitStat stat );
    void cudaCalled(void) { initProgress( CUDA_CALLED ); }
    ClientState_t(void);
};
extern struct ClientState_t St;

extern const char *DEFAULT_SVRIP;
//extern RCServer_t svrCand[RC_NVDEVMAX];
//extern RCServer_t svrSpare[RC_NVDEVMAX];
extern SvrList_t SvrSpare;
//extern RCServer_t svrBroken[RC_NVDEVMAX];
extern int    Vdevid[RC_NPTHREADMAX];
extern struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
extern void (*errorHandler)(void *arg);
extern CLIENT *Clnt[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
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
checkResult(void *rp, RCServer_t *sp);
cudaError_t cudaSetDevice_clnt( int device, int errcheck );
cudaError_t cudaMemcpyD2H_redundant( void *dst, void *src_uva, size_t count, int redundant );

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
void RCuvaRegister(int devid, void *adr[], size_t size);
void RCuvaUnregister(void *adr);
RCuva *RCuvaQuery(void *adr);

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
int
dscudaNredundancy(void);
int
dscudaRemoteCallType(void);
void
dscudaSetErrorHandler(void (*handler)(void *), void *handler_arg);
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
#if 0 //RPC_ONLY
void
ibvDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
			     int *gdim, int *bdim, RCsize smemsize, RCstream stream,
			     int narg, IbvArg *arg);
#endif

extern pthread_mutex_t InitClientMutex;
extern pthread_mutex_t cudaMemcpyD2H_mutex;
extern pthread_mutex_t cudaMemcpyH2D_mutex;
extern pthread_mutex_t cudaKernelRun_mutex;

#endif //__LIBDSCUDA_H__
