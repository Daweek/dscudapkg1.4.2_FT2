//                             -*- Mode: C++ -*-
// Filename         : libdscuda_rpc.cu
// Description      : DS-CUDA client node library for Remote Procedure Call.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-01 18:55:23
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>
#include <rpc/rpc.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <pthread.h>
#include "dscuda.h"
#include "libdscuda.h"
#include "dscudaverb.h"

#define DEBUG 1

int dscudaRemoteCallType(void) {
    return RC_REMOTECALL_TYPE_RPC;
}

//********************************************************************
//
// RCServer::
//
//********************************************************************
RCServer::RCServer(void) {
    id   = -1;
    cid  = -1;
    uniq = -1;
    strcpy(ip,       "empty");
    strcpy(hostname, "empty");
    stat_error   = 0;
    stat_correct = 0;
    d_faultconf = NULL;
    Clnt = NULL;
}

void RCServer_t::setupConnection(void) {  // Physical Device pointer.
    int  pgid = DSCUDA_PROG;
    char msg[256];

    struct sockaddr_in sockaddr;
    int ssock = RPC_ANYSOCK; // socket to the server for RPC communication.
                             // automatically created by clnttcp_create().
    int sport; // port number of the server. given by the daemon, or calculated from cid.

    St.useRpc();
    if ( St.daemon > 0 ) { // access to the server via daemon.
        sport = requestDaemonForDevice( ip, cid, St.isIbv() );
    } else { // directly access to the server.
        sport = RC_SERVER_IP_PORT + cid;
    }
    sockaddr = setupSockaddr( ip, sport );

    Clnt = clnttcp_create(&sockaddr,
			  pgid,
			  DSCUDA_VER,
			  &ssock,
			  RC_BUFSIZE, RC_BUFSIZE);

    sprintf( msg, "%s:%d (port %d) ", ip, cid, sport );

    if ( Clnt == NULL ) {
        clnt_pcreateerror( msg );
        if ( strcmp(ip, DEFAULT_SVRIP) == 0 ) {
            WARN( 0, "You may need to set an environment variable 'DSCUDA_SERVER'.\n" );
        } else {
            WARN( 0, "DSCUDA server (dscudasrv on %s:%d) may be down.\n", ip, id );
        }
        exit( EXIT_FAILURE );
    }
    WARN(2, "Established a socket connection to %s...\n", msg);
}

void RCServer_t::dupServer( RCServer_t *dup ) {
    dup->id   = this->id;
    dup->cid  = this->cid;
    dup->uniq = this->uniq;
    dup->stat_error   = this->stat_error;
    dup->stat_correct = this->stat_correct;
    strcpy( dup->ip, this->ip );
    strcpy( dup->hostname, this->hostname );
}

void RCServer_t::migrateServer(RCServer_t *newone, RCServer_t *broken) {
    dupServer(broken);
    newone->dupServer(this);

    WARN(3, "Reconnect to new physical device %s from old device %s\n",
	 this->ip, broken->ip); 
    setupConnection();

    return;
}

void
RCServer_t::migrateReallocAllRegions(void) {
    BkupMem *mem_ptr = memlist.head;
    int     verb = St.isAutoVerb();
    int     copy_count = 0;
    int     i = 0;
    
    WARN(1, "%s(RCServer_t *sp).\n", __func__);
    WARN(1, "Num. of realloc region = %d\n", memlist.length );
    
    while (mem_ptr != NULL) {
	/* TODO: select migrateded virtual device, not all region. */
	WARN(5, "mem[%d]->dst = %p, size= %d\n",
	     i, mem_ptr->d_region, mem_ptr->size);
	this->cudaMalloc(&mem_ptr->d_region, mem_ptr->size);
	mem_ptr = mem_ptr->next;
	i++;
    }
    WARN(1, "+--- Done.\n");
}

cudaError_t
RCServer::cudaMemcpyH2D(void *dst, void *src, size_t count) {
    dscudaResult *rp;
    RCbuf srcbuf;
    cudaError_t err = cudaSuccess;

    St.cudaCalled();
    srcbuf.RCbuf_len = count;
    srcbuf.RCbuf_val = (char *)src;

    rp = dscudamemcpyh2did_1((RCadr)dst, srcbuf, count, Clnt);
    if (rp == NULL) {
	WARN( 0, "NULL pointer returned, %s(). exit.\n", __func__ );
	clnt_perror(Clnt, ip);
	exit(EXIT_FAILURE);
    }
    if (rp->err != cudaSuccess) {
	err = (cudaError_t)rp->err;
    }
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    if (St.ft_mode==FT_REDUN || St.ft_mode==FT_MIGRA || St.ft_mode==FT_BOTH) {
	cudaMemcpyArgs args( dst, (void*)src, count, cudaMemcpyHostToDevice );
	reclist.add(dscudaMemcpyH2DId, (void *)&args);
    }

    return err;
}


void checkResult(void *rp, RCServer_t &sp) {
    if ( rp != NULL ) {
	return;
    } else {
	WARN( 0, "NULL pointer returned, %s(). exit.\n", __func__ );
//	clnt_perror( Clnt[Vdevid[vdevidIndex()]][sp->id], sp->ip );
	clnt_perror( sp.Clnt, sp.ip );
	exit( EXIT_FAILURE );
    }
}

static
void recoverClntError(RCServer_t *failed, RCServer_t *spare, struct rpc_err *err) {
    switch ( err->re_status ) {
	/* re_status is "clnt_stat" type.
	 * refer to /usr/include/rpc/clnt.h.
	 */
    case RPC_SUCCESS: //=0
	break;
    case RPC_CANTSEND: //=3
	break;
    case RPC_CANTRECV: //=4
	break;
    case RPC_TIMEDOUT: //=5
	WARN(1, "Detected RPC:Timed Out in  %s().\n", __func__);
	//dscudaVerbMigrateDevice( failed, spare );
	break;
    case RPC_UNKNOWNHOST: //=13
	break;
    case RPC_UNKNOWNPROTO: //=17
	break;
    case RPC_UNKNOWNADDR: //=19
	break;
    default:
	break;
    }
}

/*
 * Dscuda client-side counterpart for CUDA runtime API:
 */

/*
 * Thread Management
 */

cudaError_t cudaThreadExit(void) {
    dscudaResult *rp;
    int           vid = vdevidIndex();
    cudaError_t   err = cudaSuccess;

    WARN( 3, "cudaThreadExit() {\n");
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];  //Focused Vdev
    RCServer_t *sp   = vdev->server;           //Focused Server
//  for (int i = 0; i < vdev->nredundancy; i++, sp++) {
//      rp = dscudathreadexitid_1(Clnt[Vdevid[vid]][sp->id]);
    for ( int i = 0; i < vdev->nredundancy; i++ ) {
        rp = dscudathreadexitid_1( sp[i].Clnt );
        checkResult( rp, sp[i] );
        if ( rp->err != cudaSuccess ) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, " } cudaThreadExit()\n\n");

    return err;
}

cudaError_t cudaThreadSynchronize(void) {
    dscudaResult *rp;
    cudaError_t err = cudaSuccess;
    int vid = vdevidIndex();

    St.cudaCalled();
    WARN( 3, "cudaThreadSynchronize() {\n");
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp   = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudathreadsynchronizeid_1( sp[i].Clnt );
        checkResult( rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "} cudaThrreadSynchronize()\n");
    WARN(3, "\n");

    return err;
}

cudaError_t cudaThreadSetLimit(enum cudaLimit limit, size_t value)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    WARN(3, "cudaThreadSetLimit(%d, %zu)...", limit, value);
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp   = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudathreadsetlimitid_1(limit, value, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaThreadGetLimit(size_t *pValue, enum cudaLimit limit)
{
    cudaError_t err = cudaSuccess;
    dscudaThreadGetLimitResult *rp;
    int vid = vdevidIndex();

    WARN(3, "cudaThreadGetLimit(%p, %d)...", pValue, limit);
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp   = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudathreadgetlimitid_1(limit, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pValue = rp->value;
        }
        xdr_free((xdrproc_t)xdr_dscudaThreadGetLimitResult, (char *)rp);
    }
    WARN(3, "done.  *pValue: %zu\n", *pValue);

    return err;
}

cudaError_t
cudaThreadSetCacheConfig(enum cudaFuncCache cacheConfig) {
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    WARN(3, "cudaThreadSetCacheConfig(%d)...", cacheConfig);
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp   = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudathreadsetcacheconfigid_1(cacheConfig, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaThreadGetCacheConfig(enum cudaFuncCache *pCacheConfig) {
    cudaError_t err = cudaSuccess;
    dscudaThreadGetCacheConfigResult *rp;
    int vid = vdevidIndex();

    WARN(3, "cudaThreadGetCacheConfig(%p)...", pCacheConfig);
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp   = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudathreadgetcacheconfigid_1( sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pCacheConfig = (enum cudaFuncCache)rp->cacheConfig;
        }
        xdr_free((xdrproc_t)xdr_dscudaThreadGetCacheConfigResult, (char *)rp);
    }
    WARN(3, "done.  *pCacheConfig: %d\n", *pCacheConfig);

    return err;
}


/*
 * Error Handling
 */

cudaError_t cudaGetLastError(void) {
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    WARN(5, "cudaGetLastError()...");
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp   = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudagetlasterrorid_1( sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(5, "done.\n");

    return err;
}

cudaError_t cudaPeekAtLastError(void) {
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    WARN(5, "cudaPeekAtLastError()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudapeekatlasterrorid_1( sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(5, "done.\n");

    return err;
}

const char *cudaGetErrorString(cudaError_t error) {
    dscudaGetErrorStringResult *rp;
    static char str[4096];
    int vid = vdevidIndex();

    WARN(5, "cudaGetErrorString()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudageterrorstringid_1(error, sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (i == 0) {
            strcpy(str, rp->errmsg);
        }
        xdr_free((xdrproc_t)xdr_dscudaGetErrorStringResult, (char *)rp);
    }
    WARN(5, "done.\n");

    return str;
}

/*
 * Device Management
 */

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    WARN(3, "cudaSetDeviceFlags()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudasetdeviceflagsid_1(flags, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    invalidateModuleCache();

    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaDriverGetVersion (int *driverVersion) {
    cudaError_t err = cudaSuccess;
    dscudaDriverGetVersionResult *rp;
    int vid = vdevidIndex();

    WARN(3, "cudaDriverGetVersionCount(%p)...", driverVersion);
    Vdev_t *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudadrivergetversionid_1( sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
    }

    *driverVersion = rp->ver;
    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaRuntimeGetVersion(int *runtimeVersion) {
    cudaError_t err = cudaSuccess;
    dscudaRuntimeGetVersionResult *rp;
    int vid = vdevidIndex();

    WARN(3, "cudaRuntimeGetVersion(%p)...", runtimeVersion);
    Vdev_t *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudaruntimegetversionid_1( sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *runtimeVersion = rp->ver;
        }
        xdr_free((xdrproc_t)xdr_dscudaRuntimeGetVersionResult, (char *)rp);
    }

    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaDeviceSynchronize(void) {
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    WARN(3, "cudaDeviceSynchronize()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudadevicesynchronize_1( sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaDeviceReset(void) {
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    int vid = vdevidIndex();

    WARN(3, "cudaDeviceReset()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudadevicereset_1(sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

/*
 * Execution Control
 */

cudaError_t
cudaFuncSetCacheConfig(const char * func, enum cudaFuncCache cacheConfig) {
    cudaError_t err = cudaSuccess;
    WARN(1, "Current implementation of cudaFuncSetCacheConfig() does nothing "
         "but returning cudaSuccess.\n");
    err = cudaSuccess;
    return err;
}

/*
 * Memory Management
 */
cudaError_t
RCServer::cudaMalloc(void **d_ptr, size_t size) {
    dscudaMallocResult *rp;
    cudaError_t cuerr = cudaSuccess;
    int vid = vdevidIndex();
    void *adrs[RC_NREDUNDANCYMAX];

    rp = dscudamallocid_1(size, Clnt);
    if (rp == NULL) {
	WARN( 0, "NULL pointer returned, %s(). exit.\n", __func__ );
	clnt_perror(Clnt, ip);
	exit(EXIT_FAILURE);
    }
    if (rp->err != cudaSuccess) {
	cuerr = (cudaError_t)rp->err;
    }
    
    *d_ptr = (void*)rp->devAdr;
    xdr_free((xdrproc_t)xdr_dscudaMallocResult, (char *)rp);

    /*
     * Automatic Recoverly
     */
    if ( St.isAutoVerb() ) {
	cudaMallocArgs args( *d_ptr, size );
	//memlist.add( args.devPtr, args.size );
    }

    return cuerr;
}

cudaError_t
VirDev_t::cudaMalloc(void **devAdrPtr, size_t size) {
    dscudaMallocResult *rp;
    cudaError_t cuerr = cudaSuccess;
    int         vid = vdevidIndex();
    void       *adrs[RC_NREDUNDANCYMAX];

    WARN(3, "cudaMalloc( %p, %zu )...\n", devAdrPtr, size);
    St.cudaCalled();
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp   = vdev->server;
    for ( int i = 0; i < vdev->nredundancy; i++ ) {
	cuerr = server[i].cudaMalloc(&adrs[i], size);
	
        adrs[i] = (void*)rp->devAdr;
	WARN(3, "+--- redun[%d]: devAdrPtr=%p\n", i, adrs[i]);	
    }

    *devAdrPtr = dscudaUvaOfAdr(adrs[0], Vdevid[vid]);
    /*
     * Automatic Recoverly
     */
    if ( St.isAutoVerb() ) {
	cudaMallocArgs args( *devAdrPtr, size );
//	BKUPMEM.addRegion(args.devPtr, args.size);  /* Allocate mirroring memory */
    }
    //WARN(3, "+--- done. *devAdrPtr:%p, Length of Registered MemList: %d\n", *devAdrPtr, BKUPMEM.countRegion());

    return cuerr;
}

cudaError_t cudaMalloc(void **d_ptr, size_t size) {
    dscudaMallocResult *rp;
    cudaError_t cuerr;
    int         vid  = vdevidIndex();
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    void *adrs[RC_NREDUNDANCYMAX];

    WARN(3, "cudaMalloc(%p, %zu )...\n", d_ptr, size);

    cuerr = vdev->cudaMalloc(d_ptr, size);

    *d_ptr = dscudaUvaOfAdr(adrs[0], Vdevid[vid]);
    
    return cuerr;
}

cudaError_t cudaFree( void *mem ) {
    int          vid = vdevidIndex();
    cudaError_t  err = cudaSuccess;
    dscudaResult *rp;

    WARN(3, "cudaFree(%p) {", mem);
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    RCServer_t *sp   = vdev->server;
    for ( int i=0; i < vdev->nredundancy; i++ ) {
	rp = dscudafreeid_1( (RCadr)dscudaAdrOfUva(mem), sp[i].Clnt );
        checkResult( rp, sp[i] );
        if ( rp->err != cudaSuccess ) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    /*
     * Automatic Recoverly
     */
    if ( St.ft_mode==FT_REDUN || St.ft_mode==FT_MIGRA ||
	 St.ft_mode==FT_BOTH ) {
	//TODO: rewrite BKUPMEM.removeRegion(mem);
    }
    WARN(3, "} cudaFree(%p)\n", mem );
    WARN(3, "\n", mem );
    return err;
}

cudaError_t
VirDev_t::cudaMemcpyH2D(void *dst, void *src, size_t count) {
    WARN( 4, "   libdscuda:%s() called with \"%s(%s)\" {\n",
	  __func__, St.getFtModeString(), info );
    dscudaResult *rp;
    RCbuf srcbuf;
    cudaError_t err = cudaSuccess;

    St.cudaCalled();
    srcbuf.RCbuf_len = count;
    srcbuf.RCbuf_val = (char *)src;
    for (int i = 0; i<nredundancy; i++) {
	WARN( 4, "      + Physical[%d] dst=%p\n", i, dst);
        rp = dscudamemcpyh2did_1((RCadr)dst, srcbuf, count, server[i].Clnt);
        checkResult(rp, server[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    if ( St.ft_mode==FT_REDUN || St.ft_mode==FT_MIGRA || St.ft_mode==FT_BOTH ) {
	cudaMemcpyArgs args( dst, (void*)src, count, cudaMemcpyHostToDevice );
	reclist.add(dscudaMemcpyH2DId, (void *)&args);
    }
    WARN( 4, "   } libdscuda:%s().\n", __func__);
    return err;
}

/*
 * cudaMemcpy( DeviceToHost )
 */
cudaError_t
RCServer::cudaMemcpyD2H(void *dst, void *src, size_t count) {
    dscudaMemcpyD2HResult *rp;
    cudaMemcpyArgs args;
    cudaError_t err = cudaSuccess;

    switch ( St.ft_mode ) {
    case FT_PLAIN:
	break;
    case FT_REDUN: //thru
    case FT_MIGRA: //thru
    case FT_BOTH:
	args.dst   = (void *)dst;
	args.src   = (void *)src;
	args.count = count;
	args.kind  = cudaMemcpyDeviceToHost;
	reclist.add(dscudaMemcpyD2HId, (void *)&args);
	break;
    default:
	WARN( 0, "Unexpected failure.\n");
	exit( EXIT_FAILURE );
    }

    rp = dscudamemcpyd2hid_1((RCadr)src, count, Clnt);
    if (rp == NULL) {
	WARN( 0, "NULL pointer returned, %s(). exit.\n", __func__ );
	clnt_perror(Clnt, ip);
	exit(EXIT_FAILURE);
    }
    err = (cudaError_t)rp->err;
    if (rp->err != cudaSuccess) {
	err = (cudaError_t)rp->err;
    }
    memcpy(dst, rp->buf.RCbuf_val, rp->buf.RCbuf_len);
    xdr_free( (xdrproc_t)xdr_dscudaMemcpyD2HResult, (char *)rp );

    return err;
}

cudaError_t
VirDev_t::cudaMemcpyD2H(void *dst, void *src, size_t count) {
    WARN( 4, "   libdscuda:%s() called with \"%s(%s)\" {\n",
	  __func__, St.getFtModeString(), info );

    int matched_count   = 0;
    int unmatched_count = 0;
    int recall_result;

    cudaMemcpyArgs args;
    dscudaMemcpyD2HResult *rp;

    RCServer_t *failed_1st;
    //    int fail_flag[RC_NVDEVMAX]={0};
    cudaError_t err = cudaSuccess;

    St.cudaCalled();
    /*
     * Register called history.
     */
    switch ( St.ft_mode ) {
    case FT_PLAIN:
	break;
    case FT_REDUN: //thru
    case FT_MIGRA: //thru
    case FT_BOTH:
	args.dst   = (void *)dst;
	args.src   = (void *)src;
	args.count = count;
	args.kind  = cudaMemcpyDeviceToHost;
	reclist.add(dscudaMemcpyD2HId, (void *)&args); // not needed?
	break;
    default:
	WARN( 0, "Unexpected failure.\n");
	exit( EXIT_FAILURE );
    }

    /* Get the data from remote GPU(s), then verify */
    for (int i=0; i<nredundancy; i++) {
	WARN(4, "      + Physical[%d]:cudaMemcpy( dst=%p, src=%p, count=%zu )\n", i, dst, src, count);
	/*
	 * Access to Physical GPU Device.
	 */
	server[i].cudaMemcpyD2H(dst, src, count);
	
        if ( i==0 ) {
	    memcpy( dst, rp->buf.RCbuf_val, rp->buf.RCbuf_len );
        } else {
	    if ( bcmp( dst, rp->buf.RCbuf_val, rp->buf.RCbuf_len ) != 0 ) { // unmatched case
		server[i].stat_error++; //count up error.
		WARN( 0, "[ERRORSTATICS] Total Error Count: %d\n", server[i].stat_error);
		unmatched_count++;
		//fail_flag[i]=1;
		failed_1st = &server[i]; // temporary
		WARN(2, "   UNMATCHED redundant device %d/%d with device 0. %s()\n", i, nredundancy - 1, __func__);
	    } else { /* Matched case */
		matched_count++;
		//fail_flag[i]=0;
		WARN(3, "   Matched   reduncant device %d/%d with device 0. %s()\n", i, nredundancy - 1, __func__);
		memcpy(dst, rp->buf.RCbuf_val, rp->buf.RCbuf_len); // overwrite matched data
	    }
	}
	xdr_free( (xdrproc_t)xdr_dscudaMemcpyD2HResult, (char *)rp );
    }

    switch ( conf ) {
    case VDEV_MONO:
	if (( St.ft_mode==FT_REDUN || St.ft_mode==FT_MIGRA || St.ft_mode==FT_BOTH ) && (St.isHistoCalling()==0 )) {
	    //BKUPMEM.updateRegion( src, dst, count );
	}
	break;
    case VDEV_POLY:
	if ( unmatched_count==0 && matched_count==(nredundancy-1)) {
	    WARN(5, "   #\\(^_^)/ All %d Redundant device(s) matched. statics OK/NG = %d/%d.\n",
		 nredundancy-1, matched_count, unmatched_count);
	    /*
	     * Update backuped memory region.
	     */
	    if (( St.ft_mode==FT_REDUN || St.ft_mode==FT_MIGRA ||
		  St.ft_mode==FT_BOTH ) && (St.isHistoCalling()==0 )) {
		WARN( 5, "checkpoint-0\n");
		//TODO: rewrite BKUPMEM.updateRegion(src, dst, count); /* mirroring copy. !!!src and dst is swapped!!! */
		
		WARN( 5, "checkpoint-1\n");
	    }
	} else { /* redundant failed */
	    if ( unmatched_count>0 && matched_count<(nredundancy-1)) {
		WARN( 1, " #   #\n");
		WARN( 1, "  # #\n");
		WARN( 1, "   #  Detected Unmatched result. OK/NG= %d/%d.\n", matched_count, unmatched_count);
		WARN( 1, "  # #\n");
		WARN( 1, " #   #\n");
	    } else {
		WARN(1, "   #(;_;)   All %d Redundant device(s) unmathed. statics OK/NG = %d/%d.\n",
		     nredundancy-1, matched_count, unmatched_count);
	    }

	    if (( St.ft_mode==FT_REDUN || St.ft_mode==FT_MIGRA ||
		  St.ft_mode==FT_BOTH ) && (St.isHistoCalling()==0 )) {
		St.unsetAutoVerb();    // <=== Must be disabled autoVerb during Historical Call.
		//HISTREC.rec_en = 0; // <--- Must not record Historical call list.
	    
		//TODO: rewrite BKUPMEM.restructDeviceRegion();
		
		//recall_result = HISTREC.recall();
	    
		if (recall_result != 0) {
		    printModuleList();
		    printVirtualDeviceList();
		    //dscudaVerbMigrateDevice(failed_1st, &svrSpare[0]);
		    //dscudaVerbMigrateDevice(failed_1st, &(SvrSpare.svr[0]));
		}
		//HISTREC.on();  // ---> restore recordHist enable.
		St.setAutoVerb();    // ===> restore autoVerb enabled.
	    }
	}
	break;
    default: //irregular condition.
	WARN(1, "ERROR: # of redundancy is zero or minus value????. %s\n", __func__);
	exit( EXIT_FAILURE );
    }//switch

    WARN(4, "   } libdscuda:%s().\n", __func__);
    return err;
}

static cudaError_t
cudaMemcpyD2D(void *dst, const void *src, size_t count, Vdev_t *vdev ) {
    dscudaResult *rp;
    cudaError_t err = cudaSuccess;

    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudamemcpyd2did_1((RCadr)dst, (RCadr)src, count, sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    //<--- oikawa moved to here from cudaMemcpy();
    if (St.isAutoVerb() > 0) {
	cudaMemcpyArgs args( dst, (void *)src, count, cudaMemcpyDeviceToDevice );
	//HISTREC.add(dscudaMemcpyD2DId, (void *)&args);
    }
    //--->
    return err;
}

static cudaError_t
cudaMemcpyP2P(void *dst, int ddev, const void *src, int sdev, size_t count)
{
    cudaError_t err = cudaSuccess;
    int dev0;
    int pgsz = 4096;
    static int bufsize = 0;
    static char *buf = NULL;

    if (bufsize < count) {
        bufsize = ((count - 1) / pgsz + 1) * pgsz;
        buf = (char *)realloc(buf, bufsize);
        if (!buf) {
            perror("cudaMemcpyP2P");
            exit(1);
        }
    }

    cudaGetDevice(&dev0);

    if (sdev != dev0) {
        cudaSetDevice(sdev);
    }
    err = cudaMemcpy(buf, src, count, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        if (sdev != dev0) {
            cudaSetDevice(dev0);
        }
        return err;
    }
    if (ddev != sdev) {
        cudaSetDevice(ddev);
    }
    err = cudaMemcpy(dst, buf, count, cudaMemcpyHostToDevice);
    if (ddev != dev0) {
        cudaSetDevice(dev0);
    }
    return err;
}

/*
 * Replaced "cudaMemcpy()"
 */
cudaError_t cudaMemcpy( void *dst, const void *src,
			size_t count, enum cudaMemcpyKind kind ) {
    RCuva      *suva, *duva;
    int         dev0;
    cudaError_t err  = cudaSuccess;

    void *lsrc = dscudaAdrOfUva((void *)src);
    void *ldst = dscudaAdrOfUva(dst);

    int         vdevid = Vdevid[ vdevidIndex() ];
    Vdev_t     *vdev   = St.Vdev + vdevid;
    
    switch ( kind ) {
      case cudaMemcpyDeviceToHost:
	  WARN(3, "libdscuda:cudaMemcpy(%p, %p, %zu, DeviceToHost) called vdevid=%d...\n",
	       ldst, lsrc, count, vdevid);
	  // Avoid conflict between CheckPointing thread.
	  pthread_mutex_lock( &cudaMemcpyD2H_mutex );
	  err = vdev->cudaMemcpyD2H(dst, lsrc, count);
	  pthread_mutex_unlock( &cudaMemcpyD2H_mutex ); 
	  break;
      case cudaMemcpyHostToDevice:
	  WARN(3, "libdscuda:cudaMemcpy(%p, %p, %zu, HostToDevice) called\n", ldst, lsrc, count);
	  // Avoid conflict with CheckPointing thread.	
	  pthread_mutex_lock( &cudaMemcpyH2D_mutex );
	  err = vdev->cudaMemcpyH2D(ldst, lsrc, count);
	  pthread_mutex_unlock( &cudaMemcpyH2D_mutex );
	  break;
      case cudaMemcpyDeviceToDevice:
	  WARN(3, "libdscuda:cudaMemcpy(%p, %p, %zu, DeviceToDevice) called\n", ldst, lsrc, count);
	  err = cudaMemcpyD2D(ldst, lsrc, count, vdev );
	  break;
      case cudaMemcpyDefault: //thru
#if !__LP64__
	  WARN(0, "cudaMemcpy:In 32-bit environment, cudaMemcpyDefault cannot be given as arg4."
             "UVA is supported for 64-bit environment only.\n");
        exit(1);
#endif
      default:
	  WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
	  exit(1);
    }
    WARN(3, "} libdscuda:%s().\n", __func__);
    WARN(3, "\n");
    return err;
}

cudaError_t
cudaMemcpyPeer(void *dst, int ddev, const void *src, int sdev, size_t count) {
    WARN(3, "cudaMemcpyPeer(0x%08lx, %d, 0x%08lx, %d, %zu)...",
         (unsigned long)dst, ddev, (unsigned long)src, sdev, count);
    cudaError_t cuerr;

    cuerr = cudaMemcpyP2P(dst, ddev, src, sdev, count);

    WARN(3, "done.\n");
    return cuerr;
}

cudaError_t
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    cudaError_t err = cudaSuccess;
    dscudaGetDevicePropertiesResult *rp;

    WARN(3, "cudaGetDeviceProperties(0x%08lx, %d)...", (unsigned long)prop, device);
    Vdev_t     *vdev = St.Vdev + device;
    RCServer_t *sp   = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudagetdevicepropertiesid_1(device, sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            memcpy(prop, rp->prop.RCbuf_val, rp->prop.RCbuf_len);
        }
        xdr_free((xdrproc_t)xdr_dscudaGetDevicePropertiesResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

int
dscudaLoadModuleLocal(unsigned int ipaddr, pid_t pid, char *modulename, char *modulebuf, int vdevid, int raidid) {
    //WARN(10, "<---Entering %s()\n", __func__);
    //WARN(10, "ipaddr= %u, modulename= %s\n", ipaddr, modulename);
    
    int ret;
    //RCServer_t *sp = (St.Vdev + vdevid)->server + raidid;
    RCServer_t *sp = (St.Vdev + vdevid)->server;
    /* send to virtual GPU */
    dscudaLoadModuleResult *rp = dscudaloadmoduleid_1( St.getIpAddress(), getpid(), modulename, modulebuf, sp[raidid].Clnt );
    checkResult(rp, sp[raidid]);
    ret = rp->id;
    xdr_free((xdrproc_t)xdr_dscudaLoadModuleResult, (char *)rp);
    
    if (St.isAutoVerb() ) {
	/*Nop*/
    }

    //WARN(10, "--->Exiting  %s()\n", __func__);
    return ret;
}

/*
 * launch a kernel function of id 'kid', defined in a module of id 'moduleid'.
 * 'kid' must be unique inside a single module.
 */

void
rpcDscudaLaunchKernelWrapper(int *moduleid, int kid, char *kname,  /* moduleid is got by "dscudaLoadModule()" */
                             RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
                             RCargs args)
{
    WARN(5, "%s().\n", __func__)
    RCmappedMem *mem;
    RCstreamArray *st;
    
    pthread_mutex_lock( &cudaKernelRun_mutex ); // Avoid conflict with CheciPointing.p
    /*     
     * Automatic Recovery, Register to the called history.
     */
    if (St.isAutoVerb() ) {
        cudaRpcLaunchKernelArgs args2;
        args2.moduleid = moduleid;
        args2.kid      = kid;
        args2.kname    = kname;
        args2.gdim     = gdim;
        args2.bdim     = bdim;
        args2.smemsize = smemsize;
        args2.stream   = stream;
        args2.args     = args;
        //HISTREC.add( dscudaLaunchKernelId, (void *)&args2 );
    }

    st = RCstreamArrayQuery((cudaStream_t)stream);
    if (!st) {
        WARN(0, "invalid stream : %p\n", stream);
        exit(1);
    }

    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pDevice, mem->pHost, mem->size, cudaMemcpyHostToDevice);
        mem = mem->next;
    }

    Vdev_t     *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp   = vdev->server;
    struct rpc_err rpc_error;

    for ( int i = 0; i < vdev->nredundancy; i++ ) {
        void *rp = dscudalaunchkernelid_1(moduleid[i], kid, kname,
                                          gdim, bdim, smemsize, (RCstream)st->s[i],
                                          args, sp[i].Clnt );
	//<--- Timed Out
	clnt_geterr( sp[i].Clnt, &rpc_error );
	if ( rpc_error.re_status != RPC_SUCCESS ) {
	    break;
	}
	//--->
        checkResult(rp, sp[i]);
    }

    recoverClntError(sp, &(SvrSpare.svr[0]), &rpc_error );
    
    mem = RCmappedMemListTop;
    while (mem) {
        cudaMemcpy(mem->pHost, mem->pDevice, mem->size, cudaMemcpyDeviceToHost);
        mem = mem->next;
    }
    //---> Avoid conflict with CheckPointing.
    pthread_mutex_unlock( &cudaKernelRun_mutex ); // Avoid conflict with CheciPointing.
    WARN(5, "+--- done. %s().\n", __func__)
}

cudaError_t
cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
                size_t width, size_t height, unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaMallocArrayResult *rp;
    RCchanneldesc descbuf;
    cudaArray *ca[RC_NREDUNDANCYMAX];

    WARN(3, "cudaMallocArray(%p, %p, %zu, %zu, 0x%08x)...",
         array, desc, width, height, flags);

    descbuf.x = desc->x;
    descbuf.y = desc->y;
    descbuf.z = desc->z;
    descbuf.w = desc->w;
    descbuf.f = desc->f;

    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++ ) {
        rp = dscudamallocarrayid_1(descbuf, width, height, flags, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ca[i] = (cudaArray *)rp->array;
        xdr_free((xdrproc_t)xdr_dscudaMallocArrayResult, (char *)rp);
    }

    *array = ca[0];
    RCcuarrayArrayRegister(ca);
    WARN(3, "done. *array:%p\n", *array);

    return err;
}

cudaError_t
cudaFreeArray(struct cudaArray *array)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCcuarrayArray *ca;

    WARN(3, "cudaFreeArray(%p)...", array);
    ca = RCcuarrayArrayQuery(array);
    if (!ca) {
        WARN(0, "invalid cudaArray : %p\n", array);
        exit( EXIT_FAILURE );
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++ ) {
        rp = dscudafreearrayid_1((RCadr)ca->ap[i], sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    RCcuarrayArrayUnregister(ca->ap[0]);
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaMemcpyToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset, const void *src,
                  size_t count, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    RCcuarrayArray *ca;
    Vdev_t *vdev;
    RCServer_t *sp;

    WARN(3, "cudaMemcpyToArray(%p, %zu, %zu, %p, %zu, %s)...",
         dst, wOffset, hOffset, src, count, dscudaMemcpyKindName(kind));
    ca = RCcuarrayArrayQuery(dst);
    if (!ca) {
        WARN(0, "invalid cudaArray : %p\n", dst);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;

        vdev = St.Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            h2drp = dscudamemcpytoarrayh2did_1((RCadr)ca->ap[i], wOffset, hOffset, srcbuf, count, sp[i].Clnt);
            checkResult(h2drp, sp[i]);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)h2drp);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = St.Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            d2drp = dscudamemcpytoarrayd2did_1((RCadr)ca->ap[i], wOffset, hOffset, (RCadr)src, count, sp[i].Clnt );
            checkResult(d2drp, sp[i]);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)d2drp);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMemset(void *devPtr, int value, size_t count)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    WARN(3, "cudaMemset()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudamemsetid_1((RCadr)devPtr, value, count, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height)
{
    cudaError_t err = cudaSuccess;
    dscudaMallocPitchResult *rp;

    WARN(3, "cudaMallocPitch(%p, %p, %zu, %zu)...", devPtr, pitch, width, height);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudamallocpitchid_1(width, height, sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *devPtr = (void*)rp->devPtr;
            *pitch = rp->pitch;
        }
        xdr_free((xdrproc_t)xdr_dscudaMallocPitchResult, (char *)rp);
    }

    WARN(3, "done. *devPtr:%p  *pitch:%zu\n", *devPtr, *pitch);

    return err;
}

cudaError_t
cudaMemcpy2DToArray(struct cudaArray *dst, size_t wOffset, size_t hOffset,
                    const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpy2DToArrayD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    RCcuarrayArray *ca;
    Vdev_t *vdev;
    RCServer_t *sp;

    WARN(3, "cudaMemcpy2DToArray(%p, %zu, %zu, %p, %zu, %zu, %zu, %s)...",
         dst, wOffset, hOffset,
         src, spitch, width, height, dscudaMemcpyKindName(kind));
    ca = RCcuarrayArrayQuery(dst);
    if (!ca) {
        WARN(0, "invalid cudaArray : %p\n", dst);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = St.Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++ ) {
            d2hrp = dscudamemcpy2dtoarrayd2hid_1(wOffset, hOffset,
                                                 (RCadr)src, spitch, width, height, sp[i].Clnt);
            checkResult( d2hrp, sp[i] );
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            } else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            } else {
                WARN(3, "cudaMemcpy2DToArray() data copied from device%d matched with that from device0.\n", i);
            }
            xdr_free((xdrproc_t)xdr_dscudaMemcpy2DToArrayD2HResult, (char *)d2hrp);
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = spitch * height;
        srcbuf.RCbuf_val = (char *)src;
        vdev = St.Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            h2drp = dscudamemcpy2dtoarrayh2did_1((RCadr)ca->ap[i], wOffset, hOffset,
                                                 srcbuf, spitch, width, height, sp[i].Clnt );
            checkResult(h2drp, sp[i]);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)h2drp);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = St.Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            d2drp = dscudamemcpy2dtoarrayd2did_1((RCadr)ca->ap[i], wOffset, hOffset,
                                                 (RCadr)src, spitch, width, height, sp[i].Clnt );
            checkResult(d2drp, sp[i]);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)d2drp);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMemcpy2D(void *dst, size_t dpitch,
             const void *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    dscudaMemcpy2DD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    Vdev_t *vdev;
    RCServer_t *sp;

    WARN(3, "cudaMemcpy2D(%p, %zu, %p, %zu, %zu, %zu, %s)...",
         dst, dpitch,
         src, spitch, width, height, dscudaMemcpyKindName(kind));

    switch (kind) {
      case cudaMemcpyDeviceToHost:
        vdev = St.Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            d2hrp = dscudamemcpy2dd2hid_1(dpitch,
                                          (RCadr)src, spitch, width, height, sp[i].Clnt );
            checkResult(d2hrp, sp[i]);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            } else if (bcmp(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            } else {
                WARN(3, "cudaMemcpy() data copied from device%d matched with that from device0.\n", i);
            }
            xdr_free((xdrproc_t)xdr_dscudaMemcpy2DD2HResult, (char *)d2hrp);
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = spitch * height;
        srcbuf.RCbuf_val = (char *)src;
        vdev = St.Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            h2drp = dscudamemcpy2dh2did_1((RCadr)dst, dpitch,
                                          srcbuf, spitch, width, height, sp[i].Clnt );
            checkResult(h2drp, sp[i] );
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)h2drp);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        vdev = St.Vdev + Vdevid[vdevidIndex()];
        sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            d2drp = dscudamemcpy2dd2did_1((RCadr)dst, dpitch,
                                          (RCadr)src, spitch, width, height, sp[i].Clnt);
            checkResult(d2drp, sp[i]);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)d2drp);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    WARN(3, "cudaMemset2D(%p, %zu, %d, %zu, %zu)...",
         devPtr, pitch, value, width, height);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudamemset2did_1((RCadr)devPtr, pitch, value, width, height, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaMallocHost(void **ptr, size_t size)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaMallocHostResult *rp;

    WARN(3, "cudaMallocHost(%p, %d)...", ptr, size);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudamallochostid_1(size, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *ptr = (void*)rp->ptr;
        }
        xdr_free((xdrproc_t)xdr_dscudaMallocHostResult, (char *)rp);
    }

    WARN(3, "done. *ptr:%p\n", *ptr);
    return err;
#else
    // returned memory is not page locked.
    // it cannot be passed to cudaMemcpyAsync().
    *ptr = malloc(size);
    if (*ptr) {
        return cudaSuccess;
    } else {
        return cudaErrorMemoryAllocation;
    }
#endif
}

cudaError_t
cudaHostAlloc(void **pHost, size_t size, unsigned int flags)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaHostAllocResult *rp;

    WARN(3, "cudaHostAlloc(0x%08llx, %d, 0x%08x)...", (unsigned long)pHost, size, flags);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudahostallocid_1(size, flags, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pHost = (void*)rp->pHost;
        }
        xdr_free((xdrproc_t)xdr_dscudaHostAllocResult, (char *)rp);
    }

    WARN(3, "done. *pHost:0x%08llx\n", *pHost);

    return err;
#else
    // returned memory is not page locked.
    // it cannot be passed to cudaMemcpyAsync().

    cudaError_t err = cudaSuccess;
    void *devmem;

    WARN(3, "cudaHostAlloc(%p, %zu, 0x%08x)...", pHost, size, flags);

    *pHost = malloc(size);
    if (!*pHost) return cudaErrorMemoryAllocation;
    if (!(flags & cudaHostAllocMapped)) {
        WARN(3, "done. *pHost:%p\n", *pHost);
        return cudaSuccess;
    }

    // flags says the host memory must be mapped on to the device memory.
    err = cudaMalloc(&devmem, size);
    if (err == cudaSuccess) {
        RCmappedMemRegister(*pHost, devmem, size);
    }
    WARN(3, "done. host mem:%p  device mem:%p\n", *pHost, devmem);

    return err;
#endif
}

cudaError_t
cudaFreeHost(void *ptr)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;

    WARN(3, "cudaFreeHost(0x%08llx)...", (unsigned long)ptr);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++ ) {
        rp = dscudafreehostid_1((RCadr)ptr, sp[i].Clnt );
        checkResult(rp, sp[i] );
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");
    return err;
#else
    cudaError_t err = cudaSuccess;
    RCmappedMem *mem = RCmappedMemQuery(ptr);
    free(ptr);
    if (mem) { // ptr mapped on to a device memory.
        err = cudaFree(mem->pDevice);
        RCmappedMemUnregister(ptr);
        return err;
    } else {
        return cudaSuccess;
    }
#endif
}

// flags is not used for now in CUDA3.2. It should always be zero.
cudaError_t
cudaHostGetDevicePointer(void **pDevice, void*pHost, unsigned int flags) {
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaHostGetDevicePointerResult *rp;

    WARN(3, "cudaHostGetDevicePointer(0x%08llx, 0x%08llx, 0x%08x)...",
         (unsigned long)pDevice, (unsigned long)pHost, flags);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudahostgetdevicepointerid_1((RCadr)pHost, flags, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pDevice = (void *)rp->pDevice;
        }
        xdr_free((xdrproc_t)xdr_dscudaHostGetDevicePointerResult, (char *)rp);
    }

    WARN(3, "done. *pDevice:0x%08llx\n", *pDevice);
    return err;
#else
    RCmappedMem *mem = RCmappedMemQuery(pHost);
    if (!mem) return cudaErrorInvalidValue; // pHost is not registered as RCmappedMem.
    *pDevice = mem->pDevice;
    return cudaSuccess;
#endif
}

cudaError_t
cudaHostGetFlags(unsigned int *pFlags, void *pHost)
{
    cudaError_t err = cudaSuccess;
    dscudaHostGetFlagsResult *rp;

    WARN(3, "cudaHostGetFlags(%p %p)...", pFlags, pHost);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudahostgetflagsid_1((RCadr)pHost, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            *pFlags = rp->flags;
        }
        xdr_free((xdrproc_t)xdr_dscudaHostGetFlagsResult, (char *)rp);
    }
    WARN(3, "done. flags:0x%08x\n", *pFlags);
    return err;    
}

cudaError_t
cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream)
{
#if RC_SUPPORT_PAGELOCK
    cudaError_t err = cudaSuccess;
    dscudaMemcpyAsyncD2HResult *d2hrp;
    dscudaResult *h2drp, *d2drp;
    RCbuf srcbuf;
    RCstreamArray *st;
    Vdev_t *vdev;
    RCServer_t *sp;

    WARN(3, "cudaMemcpyAsync(0x%08llx, 0x%08llx, %d, %s, 0x%08llx)...",
         (unsigned long)dst, (unsigned long)src, count, dscudaMemcpyKindName(kind), st->s[0]);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            d2hrp = dscudamemcpyasyncd2hid_1((RCadr)src, count, (RCstream)st->s[i], sp[i].Clnt);
            checkResult(d2hrp, sp[i]);
            if (d2hrp->err != cudaSuccess) {
                err = (cudaError_t)d2hrp->err;
            }
            if (i == 0) {
                memcpy(dst, d2hrp->buf.RCbuf_val, d2hrp->buf.RCbuf_len);
            }
            xdr_free((xdrproc_t)xdr_dscudaMemcpyAsyncD2HResult, (char *)d2hrp);
        }
        break;
      case cudaMemcpyHostToDevice:
        srcbuf.RCbuf_len = count;
        srcbuf.RCbuf_val = (char *)src;
        Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            h2drp = dscudamemcpyasynch2did_1((RCadr)dst, srcbuf, count, (RCstream)st->s[i], sp[i].Clnt);
            checkResult(h2drp, sp[i]);
            if (h2drp->err != cudaSuccess) {
                err = (cudaError_t)h2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)h2drp);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
        RCServer_t *sp = vdev->server;
        for (int i = 0; i < vdev->nredundancy; i++) {
            d2drp = dscudamemcpyasyncd2did_1((RCadr)dst, (RCadr)src, count, (RCstream)st->s[i], sp[i].Clnt);
            checkResult(d2drp, sp[i]);
            if (d2drp->err != cudaSuccess) {
                err = (cudaError_t)d2drp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)d2drp);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;

#else
    // this DOES block.
    // this is only for use with a poor implementation of dscudaMallocHost().
    return cudaMemcpy(dst, src, count, kind);
#endif
}

cudaError_t
dscudaMemcpyToSymbolH2D(int moduleid, char *symbol, const void *src,
                        size_t count, size_t offset, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (St.Vdev + vdevid)->server;
    RCbuf srcbuf;
    cudaError_t err;

    srcbuf.RCbuf_len = count;
    srcbuf.RCbuf_val = (char *)src;
    rp = dscudamemcpytosymbolh2did_1(moduleid, symbol, srcbuf, count, offset, sp[raidid].Clnt);
    checkResult(rp, sp[raidid] );
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

cudaError_t
dscudaMemcpyToSymbolD2D(int moduleid, char *symbol, const void *src,
                        size_t count, size_t offset, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (St.Vdev + vdevid)->server;
    cudaError_t err;

    rp = dscudamemcpytosymbold2did_1(moduleid, symbol, (RCadr)src, count, offset, sp[raidid].Clnt );
    checkResult(rp, sp[raidid] );
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

cudaError_t
dscudaMemcpyFromSymbolD2H(int moduleid, void **dstbuf, char *symbol,
                          size_t count, size_t offset, int vdevid, int raidid)
{
    dscudaMemcpyFromSymbolD2HResult *rp;
    RCServer_t *sp = (St.Vdev + vdevid)->server;
    cudaError_t err;

    rp = dscudamemcpyfromsymbold2hid_1(moduleid, (char *)symbol, count, offset, sp[raidid].Clnt);
    *dstbuf = rp->buf.RCbuf_val;
    checkResult(rp, sp[raidid] );
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaMemcpyFromSymbolD2HResult, (char *)rp);

    return (cudaError_t)err;
}

cudaError_t
dscudaMemcpyFromSymbolD2D(int moduleid, void *dstadr, char *symbol,
                          size_t count, size_t offset, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (St.Vdev + vdevid)->server;
    cudaError_t err;

    rp = dscudamemcpyfromsymbold2did_1(moduleid, (RCadr)dstadr, (char *)symbol, count, offset, sp[raidid].Clnt );
    checkResult(rp, sp[raidid] );
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

cudaError_t
dscudaMemcpyToSymbolAsyncH2D(int moduleid, char *symbol, const void *src,
                             size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (St.Vdev + vdevid)->server;
    RCbuf srcbuf;
    cudaError_t err;

    srcbuf.RCbuf_len = count;
    srcbuf.RCbuf_val = (char *)src;
    rp = dscudamemcpytosymbolasynch2did_1(moduleid, symbol, srcbuf, count, offset, stream, sp[raidid].Clnt);
    checkResult(rp, sp[raidid]);
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

cudaError_t
dscudaMemcpyToSymbolAsyncD2D(int moduleid, char *symbol, const void *src,
                             size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (St.Vdev + vdevid)->server;
    cudaError_t err;

    rp = dscudamemcpytosymbolasyncd2did_1( moduleid, symbol, (RCadr)src, count, offset, stream, sp[raidid].Clnt);
    checkResult(rp, sp[raidid] );
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

cudaError_t
dscudaMemcpyFromSymbolAsyncD2H(int moduleid, void **dstbuf, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    dscudaMemcpyFromSymbolAsyncD2HResult *rp;
    RCServer_t *sp = (St.Vdev + vdevid)->server;
    cudaError_t err;

    rp = dscudamemcpyfromsymbolasyncd2hid_1(moduleid, (char *)symbol, count, offset,
                                            stream, sp[raidid].Clnt );
    *dstbuf = rp->buf.RCbuf_val;
    checkResult(rp, sp[raidid] );
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaMemcpyFromSymbolAsyncD2HResult, (char *)rp);

    return (cudaError_t)err;
}

cudaError_t
dscudaMemcpyFromSymbolAsyncD2D(int moduleid, void *dstadr, char *symbol,
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid)
{
    dscudaResult *rp;
    RCServer_t *sp = (St.Vdev + vdevid)->server;
    cudaError_t err;

    rp = dscudamemcpyfromsymbolasyncd2did_1(moduleid, (RCadr)dstadr, (char *)symbol, count, offset, stream, sp[raidid].Clnt);
    checkResult(rp, sp[raidid] );
    err = (cudaError_t)rp->err;
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return (cudaError_t)err;
}

/*
 * Stream Management
 */

cudaError_t
cudaStreamCreate(cudaStream_t *pStream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaStreamCreateResult *rp;
    cudaStream_t st[RC_NREDUNDANCYMAX];

    WARN(3, "cudaStreamCreate(0x%08llx)...", (unsigned long)pStream);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudastreamcreateid_1(sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        st[i] = (cudaStream_t)rp->stream;
        xdr_free((xdrproc_t)xdr_dscudaStreamCreateResult, (char *)rp);
    }

    *pStream = st[0];
    RCstreamArrayRegister(st);
    WARN(3, "done. *pStream:0x%08llx\n", *pStream);

    return err;
#else
    *pStream = 0;
    return cudaSuccess;
#endif
}

cudaError_t
cudaStreamDestroy(cudaStream_t stream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;

    WARN(3, "cudaStreamDestroy(0x%08llx)...", (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudastreamdestroyid_1((RCadr)st->s[i], sp[i].Clnt );
        checkResult(rp, sp[i] );
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    RCstreamArrayUnregister(st->s[0]);
    WARN(3, "done.\n");
    return err;
#else
    return cudaSuccess;
#endif
}

cudaError_t
cudaStreamSynchronize(cudaStream_t stream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;

    WARN(3, "cudaStreamSynchronize(0x%08llx)...", (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudastreamsynchronizeid_1((RCadr)st->s[i], sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");
    return err;
#else
    return cudaSuccess;
#endif
}

cudaError_t
cudaStreamQuery(cudaStream_t stream)
{
#if RC_SUPPORT_STREAM
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;

    WARN(3, "cudaStreamQuery(0x%08llx)...", (unsigned long)stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudastreamqueryid_1((RCadr)st->s[i], sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");
    return err;
#else
    return cudaSuccess;
#endif
}

/*
 * Event Management
 */

cudaError_t
cudaEventCreate(cudaEvent_t *event)
{
    cudaError_t err = cudaSuccess;
    dscudaEventCreateResult *rp;
    cudaEvent_t ev[RC_NREDUNDANCYMAX];

    WARN(3, "cudaEventCreate(%p)...", event);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudaeventcreateid_1(sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ev[i] = (cudaEvent_t)rp->event;
        xdr_free((xdrproc_t)xdr_dscudaEventCreateResult, (char *)rp);
    }
    *event = ev[0];
    RCeventArrayRegister(ev);
    WARN(3, "done. *event:%p\n", *event);

    return err;
}

cudaError_t
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaEventCreateResult *rp;
    cudaEvent_t ev[RC_NREDUNDANCYMAX];

    WARN(3, "cudaEventCreateWithFlags(%p, 0x%08x)...", event, flags);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudaeventcreatewithflagsid_1(flags, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        ev[i] = (cudaEvent_t)rp->event;
        xdr_free((xdrproc_t)xdr_dscudaEventCreateResult, (char *)rp);
    }
    *event = ev[0];
    RCeventArrayRegister(ev);
    WARN(3, "done. *event:%p\n", *event);

    return err;
}

cudaError_t cudaEventDestroy( cudaEvent_t event ) {
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCeventArray *ev;

    WARN(3, "cudaEventDestroy(%p)...", event);
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : %p\n", event);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++ ) {
        rp = dscudaeventdestroyid_1((RCadr)ev->e[i], sp[i].Clnt );
        checkResult(rp, sp[i] );
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    RCeventArrayUnregister(ev->e[0]);
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    cudaError_t err = cudaSuccess;
    dscudaEventElapsedTimeResult *rp;
    RCeventArray *es, *ee;

    WARN(3, "cudaEventElapsedTime(%p, %p, %p)...", ms, start, end);
    es = RCeventArrayQuery(start);
    if (!es) {
        WARN(0, "invalid start event : %p\n", start);
        exit(1);
    }
    ee = RCeventArrayQuery(end);
    if (!ee) {
        WARN(0, "invalid end event : %p\n", end);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudaeventelapsedtimeid_1((RCadr)es->e[i], (RCadr)ee->e[i], sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaEventElapsedTimeResult, (char *)rp);
    }

    *ms = rp->ms;
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;
    RCeventArray *ev;

    WARN(3, "cudaEventRecord(%p, %p)...", event, stream);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : %p\n", stream);
        exit(1);
    }
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : %p\n", event);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudaeventrecordid_1((RCadr)ev->e[i], (RCadr)st->s[i], sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaEventSynchronize(cudaEvent_t event)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCeventArray *ev;

    WARN(3, "cudaEventSynchronize(%p)...", event);
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : %p\n", event);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudaeventsynchronizeid_1((RCadr)ev->e[i], sp[i].Clnt );
        checkResult(rp, sp[i] );
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }
    WARN(3, "done.\n");
    return err;
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCeventArray *ev;

    WARN(3, "cudaEventQuery(%p)...", event);
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : %p\n", event);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudaeventqueryid_1((RCadr)ev->e[i], sp[i].Clnt );
        checkResult(rp, sp[i] );
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCstreamArray *st;
    RCeventArray *ev;

    WARN(3, "cudaStreamWaitEvent(%p, %p, 0x%08x)...", stream, event, flags);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : %p\n", stream);
        exit(1);
    }
    ev = RCeventArrayQuery(event);
    if (!ev) {
        WARN(0, "invalid event : %p\n", event);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudastreamwaiteventid_1((RCadr)st->s[i], (RCadr)ev->e[i], flags, sp[i].Clnt);
        checkResult(rp, sp[i]);
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    }

    WARN(3, "done.\n");
    return err;
}

/*
 * Texture Reference Management
 */

cudaChannelFormatDesc
cudaCreateChannelDesc(int x, int y, int z, int w, enum cudaChannelFormatKind f)
{
    dscudaCreateChannelDescResult *rp;
    cudaChannelFormatDesc desc;

    WARN(3, "cudaCreateChannelDesc()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudacreatechanneldescid_1(x, y, z, w, f, sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (i == 0) {
            desc.x = rp->x;
            desc.y = rp->y;
            desc.z = rp->z;
            desc.w = rp->w;
            desc.f = (enum cudaChannelFormatKind)rp->f;
        }
        xdr_free((xdrproc_t)xdr_dscudaCreateChannelDescResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return desc;
}

cudaError_t
cudaGetChannelDesc(struct cudaChannelFormatDesc *desc, const struct cudaArray *array)
{
    cudaError_t err = cudaSuccess;
    dscudaGetChannelDescResult *rp;
    RCcuarrayArray *ca;

    WARN(3, "cudaGetChannelDesc()...");
    ca = RCcuarrayArrayQuery((cudaArray *)array);
    if (!ca) {
        WARN(0, "invalid cudaArray : %p\n", array);
        exit(1);
    }
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscudagetchanneldescid_1( (RCadr)ca->ap[i], sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->err != cudaSuccess) {
            err = (cudaError_t)rp->err;
        }
        if (i == 0) {
            desc->x = rp->x;
            desc->y = rp->y;
            desc->z = rp->z;
            desc->w = rp->w;
            desc->f = (enum cudaChannelFormatKind)rp->f;
        }
        xdr_free((xdrproc_t)xdr_dscudaGetChannelDescResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return err;
}


cudaError_t
cudaUnbindTexture(const struct textureReference * texref)
{
    cudaError_t err = cudaSuccess;

    WARN(4, "Current implementation of cudaUnbindTexture() does nothing "
         "but returning cudaSuccess.\n");

    err = cudaSuccess;

    return err;
}

/*
 * CUFFT library
 */
cufftResult CUFFTAPI
cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType type)
{
    cufftResult res = CUFFT_SUCCESS;
    dscufftPlanResult *rp;

    WARN(3, "cufftPlan3d()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscufftplan3did_1(nx, ny, nz, (unsigned int)type, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
        if (i == 0) {
            *plan = rp->plan;
        }
        xdr_free((xdrproc_t)xdr_dscufftPlanResult, (char *)rp);
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftDestroy(cufftHandle plan) {
    cufftResult res = CUFFT_SUCCESS;
    dscufftResult *rp;

    WARN(3, "cufftDestroy()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++ ) {
        rp = dscufftdestroyid_1((unsigned int)plan, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if ( rp->err != CUFFT_SUCCESS ) {
            res = (cufftResult)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscufftResult, (char *)rp);
    }
    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecC2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction)
{
    cufftResult res = CUFFT_SUCCESS;
    dscufftResult *rp;

    WARN(3, "cufftExecC2C()...");
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        rp = dscufftexecc2cid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, direction, sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
        xdr_free((xdrproc_t)xdr_dscufftResult, (char *)rp);
    }

    WARN(3, "done.\n");

    return res;
}

#if 0

/*
 * Interface to CUFFT & CUBLAS written by Yoshikawa for old Remote CUDA.
 * some are already ported to DS-CUDA (see 'dscufftXXXid_1_svc' function defs above),
 * but some are not. Maybe someday, when I have time...
 */

cufftResult CUFFTAPI
cufftPlan1d(cufftHandle *plan, int nx, cufftType type, int batch)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftPlanResult *rp;

    WARN(3, "cufftPlan1d()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++) {
        rp = rcufftplan1did_1(nx, (unsigned int)type, batch, sp.Clnt );
        checkResult(rp, sp);
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }
    *plan = rp->plan;

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType type, int batch)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftPlanResult *rp;

    WARN(3, "cufftPlan2d()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++ ) {
        rp = rcufftplan2did_1(nx, ny, (unsigned int)type, sp[i].Clnt );
        checkResult(rp, sp[i] );
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }
    *plan = rp->plan;

    WARN(3, "done.\n");

    return res;
}

/*
  cufftResult CUFFTAPI
  cufftPlanMany(cufftHandle *plan, int nx, cufftType type, int batch)
  {
  cufftResult res = CUFFT_SUCCESS;
  rcufftPlanResult *rp;

  WARN(3, "cufftPlan1d()...");
  Server *sp = Serverlist;
  for (int i = 0; i < Nredundancy; i++, sp++) {
  rp = rcufftplan1did_1(nx, (unsigned int)type, Clnt[0][sp->id]);
  checkResult(rp, sp);
  if (rp->err != CUFFT_SUCCESS) {
  res = (cufftResult)rp->err;
  }
  }
  *plan = rp->plan;

  WARN(3, "done.\n");

  return res;
  }
*/

cufftResult CUFFTAPI
cufftExecR2C(cufftHandle plan, cufftComplex *idata, cufftComplex *odata)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    WARN(3, "cufftExecR2C()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++ ) {
        rp = rcufftexecr2cid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, sp[i].Clnt );
        checkResult(rp, sp[i] );
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftComplex *odata)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    WARN(3, "cufftExecC2R()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++ ) {
        rp = rcufftexecc2rid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecZ2Z(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    WARN(3, "cufftExecZ2Z()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++) {
        rp = rcufftexecz2zid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, direction, sp.Clnt );
        checkResult(rp, sp[i] );
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecD2Z(cufftHandle plan, cufftComplex *idata, cufftComplex *odata)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    WARN(3, "cufftExecD2Z()...");
    Server *sp = Serverlist;
    for ( int i = 0; i < Nredundancy; i++ ) {
        rp = rcufftexecd2zid_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, sp[i].Clnt);
        checkResult(rp, sp[i] );
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

cufftResult CUFFTAPI
cufftExecZ2D(cufftHandle plan, cufftComplex *idata, cufftComplex *odata)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    WARN(3, "cufftExecZ2D()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++ ) {
        rp = rcufftexecz2did_1((unsigned int)plan, (RCadr)idata, (RCadr)odata, sp[i].Clnt );
        checkResult(rp, sp[i] );
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}

/*
  cufftResult CUFFTAPI
  cufftSetStream(cufftHandle plan, cudaStream_t stream)
  {
  }
*/

cufftResult CUFFTAPI
cufftSetCompatibilityMode(cufftHandle plan, cufftCompatibility mode)
{
    cufftResult res = CUFFT_SUCCESS;
    rcufftResult *rp;

    WARN(3, "cufftSetCompatibilityMode()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++ ) {
        rp = rcufftsetcompatibilitymodeid_1((unsigned int)plan, (unsigned int)mode, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->err != CUFFT_SUCCESS) {
            res = (cufftResult)rp->err;
        }
    }

    WARN(3, "done.\n");

    return res;
}


/*
 * CUBLAS Library functions
 */
cublasStatus_t CUBLASAPI
cublasCreate_v2(cublasHandle_t *handle)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasCreateResult *rp;

    WARN(3, "cublasCreate()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++ ) {
        rp = rcublascreate_v2id_1( sp[i].Clnt );
        checkResult(rp, sp[i]);
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }
    *handle = (cublasHandle_t)rp->handle;

    WARN(3, "done.\n");

    return res;
}

cublasStatus_t CUBLASAPI
cublasDestroy_v2(cublasHandle_t handle)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasResult *rp;

    WARN(3, "cublasDestroy()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++ ) {
        rp = rcublasdestroy_v2id_1((RCadr)handle, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }
    WARN(3, "done.\n");

    return res;
}

cublasStatus_t CUBLASAPI
cublasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasResult *rp;

    RCbuf buf;
    buf.RCbuf_val = (char *)malloc(n * elemSize);
    buf.RCbuf_len = n;
    memcpy(buf.RCbuf_val, x, n);

    WARN(3, "cublasSetVector()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++ ) {
        rp = rcublassetvectorid_1(n, elemSize, buf, incx, (RCadr)devicePtr, incy, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }
    WARN(3, "done.\n");

    return res;
}

cublasStatus_t CUBLASAPI
cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasGetVectorResult *rp;

    WARN(3, "cublasGetVector()...");
    Server *sp = Serverlist;
    for (int i = 0; i < Nredundancy; i++) {
        rp = rcublasgetvectorid_1(n, elemSize, (RCadr)x, incx, incy, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }

    memcpy(y, rp->y.RCbuf_val, n * elemSize);
    WARN(3, "done.\n");

    return res;
}

cublasStatus_t CUBLASAPI
cublasSgemm_v2(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
               const float *alpha, const float *A, int lda,
               const float *B, int ldb, const float *beta, float *C, int ldc)
{
    cublasStatus_t res = CUBLAS_STATUS_SUCCESS;
    rcublasResult *rp;

    WARN(3, "cublasSgemm()...");
    Server *sp = Serverlist;
    for ( int i = 0; i < Nredundancy; i++ ) {
        rp = rcublassgemm_v2id_1((RCadr)handle, (unsigned int)transa, (unsigned int)transb, m, n, k,
                                 *alpha, (RCadr)A, lda, (RCadr)B, ldb, *beta, (RCadr)C, ldc, sp[i].Clnt );
        checkResult( rp, sp[i] );
        if (rp->stat != CUBLAS_STATUS_SUCCESS) {
            res = (cublasStatus_t)rp->stat;
        }
    }
    WARN(3, "done.\n");

    return res;
}
#endif // CUFFT

