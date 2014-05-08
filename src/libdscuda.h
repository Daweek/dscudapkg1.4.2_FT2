//                             -*- Mode: C++ -*-
// Filename         : libdscuda.h
// Description      : DS-CUDA client node common(IBV and RPC) library header.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-02-12 20:57:57
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef __LIBDSCUDA_H__
#define __LIBDSCUDA_H__
#include "sockutil.h"

typedef struct {
    int  id;   // index for each redundant server.
    int  cid;  // id of a server given by -c option to dscudasvr.
               // clients specify the server using this num preceded
               // by an IP address & colon, e.g.,
               // export DSCUDA_SERVER="192.168.1.123:2"
    char ip[512];// ex.) 192.168.0.92
    int  uniq; // unique number in all RCServer_t including svrCand[]. 
} RCServer_t;

typedef struct {
    int nredundancy;
    RCServer_t server[RC_NREDUNDANCYMAX];
} Vdev_t;

typedef struct {
    int    valid;   /* "1" means valid, "0" means invalid, "(else)" not used. */
    int    vdevid;  /* the virtual device the module is loaded into. */
    int    id[RC_NREDUNDANCYMAX]; //  that consists of the virtual one.
    char   name[RC_KMODULENAMELEN];
    time_t sent_time;
    char  *ptx_image; //add
    void validate()   { valid=1; }
    void invalidate() { valid=0; }
    int  isValid()  {
	if (valid==0) return 0;
	else          return 1;
    }
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

/***
 *** Control client's behavior
 ***/
struct ClientState_t {
    unsigned int ip_addr;          /* My IP address */
    int use_ibv;             /* 1:IBV, 0:RPC   */
    int auto_verb;           /* 1:Redundant calculation */
    int record_hist;
    int use_daemon;
    int historical_calling;
    void setIpAddress(unsigned int val) { ip_addr = val; }
    unsigned int getIpAddress() { return ip_addr; }
    
    void useIbv() { use_ibv = 1; }
    void useRpc() { use_ibv = 0; }
    int  isIbv()  { return use_ibv;       }
    int  isRpc()  { return (1 - use_ibv); }
    
    void   setAutoVerb(int val=1)  { auto_verb = val; }
    void unsetAutoVerb() { auto_verb = 0; }
    int  isAutoVerb(void)    { return auto_verb; }

    void   setUseDaemon(void) { use_daemon = 1; }
    void unsetUseDaemon() { use_daemon = 0; }
    int  isUseDaemon(void)   { return use_daemon; }

    void   setRecordHist(void) { record_hist = 1; }
    void unsetRecordHist() { record_hist = 0; }
    int  isRecordHist(void)  { return record_hist; }

    void   setHistoCalling() { historical_calling = 1; }
    void unsetHistoCalling() { historical_calling = 0; }
    int  isHistoCalling()   { return historical_calling; }
    
    ClientState_t() {
	ip_addr = 0;
	use_ibv = 0;
	auto_verb = 0;
	record_hist = 0;
	use_daemon = 0;
	historical_calling = 0;
    }
};
extern struct ClientState_t St;

extern const char *DEFAULT_SVRIP;
extern Vdev_t Vdev[RC_NVDEVMAX];
extern int    Vdevid[RC_NPTHREADMAX];
extern struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
extern void (*errorHandler)(void *arg);
extern CLIENT *Clnt[RC_NVDEVMAX][RC_NREDUNDANCYMAX];
extern void *errorHandlerArg;

extern RCmappedMem    *RCmappedMemListTop;
extern RCmappedMem    *RCmappedMemListTail;

/* <-- Redundant APIs */
void dscudaRecordHistOn(void);  // add by oikawa
void dscudaRecordHistOff(void); // add by oikawa
void dscudaAutoVerbOff(void);
void dscudaAutoVerbOn(void);
/* --> Redundant APIs */

void invalidateModuleCache(void);
int  requestDaemonForDevice(char *ipaddr, int devid, int useibv);
int  vdevidIndex(void);
void setupConnection(int idev, RCServer_t *sp);
int  dscudaLoadModuleLocal(unsigned int ipaddr, pid_t pid, char *modulename,
			   char *modulebuf, int vdevid, int raidid);
int *dscudaLoadModule(char *srcname, char *strdata);
void
checkResult(void *rp, RCServer_t *sp);
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
void initClient(void);

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
void
ibvDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,
			     int *gdim, int *bdim, RCsize smemsize, RCstream stream,
			     int narg, IbvArg *arg);
#endif //__LIBDSCUDA_H__
