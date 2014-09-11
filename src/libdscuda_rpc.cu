//                             -*- Mode: C++ -*-
// Filename         : libdscuda_rpc.cu
// Description      : DS-CUDA client node library for Remote Procedure Call.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-11 15:02:55
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
    uniq = RC_UNIQ_INVALID;
    ft_mode = FT_INIT;
    strcpy(ip,       "empty");
    strcpy(hostname, "empty");
    stat_error   = 0;
    stat_correct = 0;
    d_faultconf = NULL;
    Clnt = NULL;
}

int RCServer::setupConnection(void) {
    int  pgid = DSCUDA_PROG;
    char msg[256];

    struct sockaddr_in sockaddr;
    int ssock = RPC_ANYSOCK; // socket to the server for RPC communication.
                             // automatically created by clnttcp_create().
    int sport; // port number of the server. given by the daemon, or calculated from cid.

    St.useRpc();
    if ( St.daemon > 0 ) { // access to the server via daemon.
	WARN(1, "Access port number is informed by daemon.\n");
        sport = requestDaemonForDevice( ip, cid, St.isIbv() );
    } else { // directly access to the server.
	WARN(1, "Access port number is self-defined by client.\n");
        sport = RC_SERVER_IP_PORT + cid;
    }
    if ( sport < 0 ) { // means that maybe daemon program is down.
	return -1;
    }
    
    sockaddr = setupSockaddr( ip, sport );

    this->Clnt = clnttcp_create(&sockaddr,
				pgid,
				DSCUDA_VER,
				&ssock,
				RC_BUFSIZE, RC_BUFSIZE);

    sprintf( msg, "Clnt=%p, %s:%d (port %d) ", Clnt, ip, cid, sport );

    if ( Clnt == NULL ) {
        clnt_pcreateerror( msg );
        if ( strcmp(ip, DEFAULT_SVRIP) == 0 ) {
            WARN( 0, "You may need to set an environment variable 'DSCUDA_SERVER'.\n" );
        } else {
            WARN( 0, "DSCUDA server (dscudasrv on %s:%d) may be down.\n", ip, id );
        }
        exit( EXIT_FAILURE );
    }
    WARN(2, "Established a socket connection between %s...\n", msg);
    return 0;
}

void RCServer::dupServer(RCServer_t *dup) {
    dup->id   = this->id;
    dup->cid  = this->cid;
    dup->uniq = this->uniq;
    dup->ft_mode = this->ft_mode;
    dup->stat_error   = this->stat_error;
    dup->stat_correct = this->stat_correct;
    strcpy( dup->ip, this->ip );
    strcpy( dup->hostname, this->hostname );
}

void RCServer::migrateServer(RCServer_t *spare) {
    RCServer_t tmp;

    dupServer(&tmp);

    this->cid = spare->cid;
    this->stat_error = spare->stat_error;
    this->stat_correct = spare->stat_correct;
    strcpy(this->ip, spare->ip);
    strcpy(this->hostname, spare->hostname);

    spare->ft_mode = FT_BROKEN;
    spare->cid = tmp.cid;
    spare->stat_error = tmp.stat_error;
    spare->stat_correct = tmp.stat_correct;
    strcpy(spare->ip, tmp.ip);
    strcpy(spare->hostname, tmp.hostname);

    WARN(1, "***  Reconnect to new physical device\n");
    WARN(1, "***  Old physical device: ip=%s, port=%d\n", spare->ip, spare->cid);
    WARN(1, "***  New physical device: ip=%s, port=%d\n", this->ip,  this->cid); 

    return;
}

void RCServer::migrateReallocAllRegions(void) {
    BkupMem *memp = memlist.head;
    int     verb = St.isAutoVerb();
    int     i=0;
    
    WARN(1, "RCServer::%s(void) {\n", __func__);
    WARN(1, "   + # of realloc region = %d.\n", memlist.length );
    
    while (memp != NULL) {
	this->cudaMalloc(&memp->d_region, memp->size);
	WARN(5, "   + region[%d]: v_ptr=%p, d_ptr=%p(updated), size= %d\n",
	     i, memp->v_region, memp->d_region, memp->size);

	memp = memp->next;
	i++;
    }
    WARN(1, "}\n");
    WARN(1, "\n");
}

void RCServer::migrateDeliverAllRegions(void) {
    BkupMem *memp = memlist.head;
    int     verb = St.isAutoVerb();
    int     copy_count = 0;
    int     i = 0;
    
    WARN(1, "RCServer::%s(void) {\n", __func__);
    WARN(1, "   + # of deliverd region = %d.\n", memlist.length );
    
    while (memp != NULL) {
	WARN(5, "   + region[%d]: v_ptr=%p, d_ptr=%p, h_ptr=%p, size= %d\n",
	     i, memp->v_region, memp->d_region, memp->h_region, memp->size);
	this->cudaMemcpyH2D(memp->v_region, memp->h_region, memp->size);
	memp = memp->next;
	i++;
    }
    WARN(1, "}\n");
    WARN(1, "\n");
}

void RCServer::migrateDeliverAllModules(void) {
    WARN(1, "RCServer::%s(void) {\n", __func__);
    WARN(1, "   + # of deliverd modules = %d.\n", -1000);
    
    WARN(1, "}\n");
    WARN(1, "\n");
}

void RCServer::migrateRebuildModulelist(void) {
    WARN(5, "RCServer::%s(void) {\n", __func__);
    dscudaLoadModuleResult *rp;
    int module_id;
    struct rpc_err rpc_error;
    
    for (int i=0; i<RC_NKMODULEMAX; i++) {
	if (modulelist[i].valid != 1) {
	    continue;
	}
	WARN(0,"i=%d,checkpoint-0\n", i);
	rp = dscudaloadmoduleid_1(St.getIpAddress(),
				  getpid(),
				  modulelist[i].ptx_data->name,
				  modulelist[i].ptx_data->ptx_image,
				  Clnt);
	WARN(0,"i=%d,checkpoint-1\n",i);
	//<--- RPC Error Hook
	clnt_geterr(Clnt, &rpc_error);
	if (rpc_error.re_status == RPC_SUCCESS) {
	    if (rp == NULL) {
		WARN( 0, "NULL pointer returned, %s(). exit.\n", __func__ );
		clnt_perror(Clnt, ip);
		exit(EXIT_FAILURE);
	    }
	} else {
	    rpcErrorHook(&rpc_error);
	}
	//---> RPC Error Hook.
	
	module_id = rp->id;
	xdr_free((xdrproc_t)xdr_dscudaLoadModuleResult, (char *)rp);
	
	modulelist[i].id        = module_id;
	modulelist[i].sent_time = time(NULL);
    }
    WARN(5, "} //RCServer::%s(void).\n", __func__);
}

void VirDev_t::setFaultMode(enum FtMode_e fault_mode) {
    this->ft_mode = fault_mode;
    for (int i=0; i<RC_NREDUNDANCYMAX; i++) {
	server[i].ft_mode = fault_mode;
    }
    return;
}

void checkResult(void *rp, RCServer_t &sp) {
    if ( rp != NULL ) {
	return;
    } else {
	WARN( 0, "NULL pointer returned, %s(). exit.\n", __func__ );
	clnt_perror( sp.Clnt, sp.ip );
	exit(EXIT_FAILURE);
    }
}

//void rpcErrorHook(RCServer_t *failed, RCServer_t *spare, struct rpc_err *err) {
void RCServer::rpcErrorHook(struct rpc_err *err) {
    RCServer *sp;
    int retval;
    
    WARN(1, "********************************************************\n");
    WARN(1, "***  detected rpc communication error; ");
    switch (err->re_status) {// *refer to /usr/include/rpc/clnt.h.
    case RPC_SUCCESS: //=0
	WARN0(1, "\"RPC_SUCCESS\".\n");
	break;
    case RPC_CANTSEND: //=3
	WARN0(1, "\"RPC_CANTSEND\".\n");
	break;
    case RPC_CANTRECV: //=4
	WARN0(1, "\"RPC_CANTRECV\".\n");
	break;
    case RPC_TIMEDOUT: //=5
	WARN0(1, "\"RPC_TIMEDOUT\".\n");
	break;
    case RPC_UNKNOWNHOST: //=13
	WARN0(1, "\"RPC_UNKNOWNHOST\".\n");
	break;
    case RPC_UNKNOWNPROTO: //=17
	WARN0(1, "\"RPC_UNKNOWNPROTO\".\n");
	break;
    case RPC_UNKNOWNADDR: //=19
	WARN0(1, "\"RPC_UNKNOWNADDR\".\n");
	break;
    default:
	WARN0(1, "\"RPC_(UNKNOWN-KIND).\n");
	break;
    }
    WARN(1, "***  hostname=\"%s\", ip=%s, server[%d]\n", hostname, ip, id);
    WARN(1, "***  FAULT_TOLERANT_MODE= ");
    
    switch(ft_mode) {
    case FT_PLAIN:
	WARN0(1, "\"FT_PLAIN\"\n");
	WARN(1, "***  So, I give up to continue calculation, sorry.\n");
	WARN(1, "********************************************************\n");
	exit(1);
    case FT_REDUN:
	WARN0(1, "\"FT_REDUN\"\n");
	WARN(1, "***  So, I give up to continue calculation, sorry.\n");
	WARN(1, "********************************************************\n");
	exit(1);
    case FT_MIGRA:
	WARN0(1, "\"FT_MIGRA\"\n");
	WARN(1, "***  I am going to migrate to another device.\n");

	do {
	    sp = SvrSpare.findSpare();
	    if ( sp == NULL ) {
		WARN(0, "*** Not found any spare servers.\n");
		exit(EXIT_FAILURE);
	    }
	    WARN(1, "*** Found spare server.\n");
	    WARN(1, "***    + ip = %s:%d\n", sp->ip, sp->cid);
	    migrateServer(sp);
	    retval = setupConnection();
	    if (retval != 0) { //failed to connection.
		sp->ft_mode = FT_BROKEN; // write mark of broken.
		WARN(1, "***    + but looks like broken.\n");
	    }
	} while ( retval != 0 );
	
	migrateReallocAllRegions();
	migrateDeliverAllRegions();
	migrateRebuildModulelist();
	break;
    case FT_BOTH:
	WARN0(1, "\"FT_BOTH\"\n");
	break;
    default:
	WARN0(1, "FT_(UNKNOWN)\n");
    }
    
    WARN(1, "********************************************************\n");
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
    //invalidateModuleCache();
    for (int i=0; i<St.Nvdev; i++) {
	St.Vdev[i].invalidateAllModuleCache();
    }

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
cudaError_t RCServer::cudaMalloc(void **d_ptr, size_t size) {
    dscudaMallocResult *rp;
    cudaError_t cuerr = cudaSuccess;

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

    return cuerr;
}

cudaError_t VirDev_t::cudaMalloc(void **d_ptr, size_t size) {
    cudaError_t cuerr_phy;
    cudaError_t cuerr_vir;
    void       *adrs[RC_NREDUNDANCYMAX];
    void       *uva_ptr = NULL;


    WARN(3, "   Vdev[%d].cudaMalloc(%p, %zu) nredundancy=%d {\n",
	 id, d_ptr, size, nredundancy);

    for (int i=0; i<nredundancy; i++) {
	/*
	 * Virtual device looks like one device but has
	 * nredundancy devices.
	 */
	cuerr_phy = server[i].cudaMalloc(&adrs[i], size);
	WARN(3, "      Physical[%d]: d_ptr=%p\n", i, adrs[i]);
	if (cuerr_phy != cudaSuccess) {
	    WARN(0, "      server[%d].cudaMalloc() Faild\n", i);
	    cuerr_vir = cuerr_phy;
	    break;
	}
	
	if (i==0) {
	    uva_ptr = dscudaUvaOfAdr(adrs[0], id);
	}
	server[i].memlist.add( uva_ptr, adrs[i], size );
	WARN(3, "         + memlist.add(v_ptr=%p, d_ptr=%p, size=%zu)\n", uva_ptr, adrs[i], size);
    }

    this->memlist.add(uva_ptr, NULL, size);

    *d_ptr = uva_ptr; // Return UVA address of physical[0].

    /*
     * Record called history of CUDA APIs.
     */
    struct CudaMallocArgs_t args;
    if (ft_mode==FT_REDUN || ft_mode==FT_MIGRA || ft_mode==FT_BOTH) {
	args.devPtr = uva_ptr;
	args.size   = size;
	this->reclist.add(dscudaMallocId, &args);
	for (int i=0; i<nredundancy; i++) {
	    server[i].reclist.add(dscudaMallocId, &args);	    
	}
    }
    
    WARN(3, "   }\n");
    return cuerr_vir;
}

cudaError_t cudaMalloc(void **d_ptr, size_t size) {
    cudaError_t cuerr;
    int         vid  = vdevidIndex();
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];
    void       *adrs;

    WARN(3, "cudaMalloc(%p, %zu) on Vdev[%d] {\n", d_ptr, size, Vdevid[vid]);
    /*
     * 
     */
    cuerr = vdev->cudaMalloc(d_ptr, size);

    WARN(3, "}\n", d_ptr, size);
    WARN(3, "\n", d_ptr, size);
    return cuerr;
}
/*
 * cudaFree() series.
 */
cudaError_t RCServer::cudaFree(void *v_ptr) {
    cudaError_t  err = cudaSuccess;
    dscudaResult *rp;
    void *d_ptr = memlist.queryDevicePtr(v_ptr);
    
    WARN(3, "      + Physical[%d].cudaFree(%p) { }\n", id, d_ptr);

    rp = dscudafreeid_1((RCadr)d_ptr, Clnt);
    if (rp == NULL) {
	WARN( 0, "NULL pointer returned, %s(). exit.\n", __func__ );
	clnt_perror(Clnt, ip);
	exit(EXIT_FAILURE);
    }
    if ( rp->err != cudaSuccess ) {
	err = (cudaError_t)rp->err;
    }
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);

    return err;
}

cudaError_t VirDev_t::cudaFree(void *v_ptr) {
    cudaError_t  err = cudaSuccess;

    WARN(3, "   + Virtual[%d].cudaFree(%p) {\n", id, v_ptr);
    for (int i=0; i<nredundancy; i++) {
	err = server[i].cudaFree(v_ptr);
	server[i].memlist.remove(v_ptr);
    }
    this->memlist.remove(v_ptr);

    /*
     * Record called history of CUDA APIs.
     */
    struct CudaFreeArgs_t args;
    if (ft_mode==FT_REDUN || ft_mode==FT_MIGRA || ft_mode==FT_BOTH) {
	args.devPtr = v_ptr;
	this->reclist.add(dscudaFreeId, &args);
	
	for (int i=0; i<nredundancy; i++) {
	    server[i].reclist.add(dscudaFreeId, &args);	    
	}
    }

    WARN(3, "   + }\n");
    return err;
}

cudaError_t cudaFree(void *d_ptr) {
    int          vid = vdevidIndex();
    cudaError_t  err = cudaSuccess;

    WARN(3, "cudaFree(%p) {\n", d_ptr);
    Vdev_t     *vdev = St.Vdev + Vdevid[vid];

    err = vdev->cudaFree(d_ptr);

    /*
     * Automatic Recoverly
     */
    if ( St.ft_mode==FT_REDUN || St.ft_mode==FT_MIGRA ||
	 St.ft_mode==FT_BOTH ) {
	//TODO: rewrite BKUPMEM.removeRegion(mem);
    }
    WARN(3, "}\n");
    WARN(3, "\n");
    return err;
}

/*
 * cudaMemcpy( HostToDevice )
 */
cudaError_t RCServer::cudaMemcpyH2D(void *v_ptr, const void *h_ptr, size_t count) {

    dscudaResult *rp;
    RCbuf srcbuf;
    void *d_ptr;
    cudaError_t cuda_error;


    srcbuf.RCbuf_len = count;
    srcbuf.RCbuf_val = (char *)h_ptr;

    //<-- Translate virtual v_ptr to real d_ptr.
    d_ptr = memlist.queryDevicePtr(v_ptr);
    //--> Translate virtual v_ptr to real d_ptr.
    WARN(5, "      + Physical[%d]:%s(v_ptr=%p ==> d_ptr=%p, size=%zu)\n",
	 id, __func__, v_ptr, d_ptr, count);

    //<-- RPC communication.
    rp = dscudamemcpyh2did_1((RCadr)d_ptr, srcbuf, count, Clnt);
    
    //<-- RPC fault check.
    struct rpc_err rpc_error;
    
    clnt_geterr(Clnt, &rpc_error);
    if (rpc_error.re_status == RPC_SUCCESS) { /*Got response from remote client*/
	if (rp == NULL) {
	    WARN( 0, "NULL pointer returned, %s:%s():L%d.\nexit.\n\n\n", __FILE__, __func__, __LINE__ );
	    clnt_perror(Clnt, ip);
	    exit(EXIT_FAILURE);
	} else {
	    cuda_error = (cudaError_t)rp->err;
	}
    } else { /*In any trouble in RPC communication*/
	rpcErrorHook(&rpc_error);	
    }
    //--> RPC fault check.
    xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
    //--> RPC communication.

    return cuda_error;
}

cudaError_t VirDev_t::cudaMemcpyH2D(void *v_ptr, const void *h_ptr, size_t count) {
    WARN( 4, "   Vdev[%d].%s() {\n", id, __func__);
    cudaError_t    err = cudaSuccess;
	
    for (int i=0; i<nredundancy; i++) {

	server[i].cudaMemcpyH2D(v_ptr, h_ptr, count);
	if ( ft_mode==FT_REDUN || ft_mode==FT_MIGRA || ft_mode==FT_BOTH ) {
	    
	}
    }

    /*
     * Record called history of CUDA APIs.
     */
    struct CudaMemcpyArgs_t args;
    if (ft_mode==FT_REDUN || ft_mode==FT_MIGRA || ft_mode==FT_BOTH) {
	args.dst   = v_ptr;
	args.src   = (void *)h_ptr;
	args.count = count;
	args.kind  = cudaMemcpyHostToDevice;
	this->reclist.add(dscudaMemcpyH2DId, &args);
	
	for (int i=0; i<nredundancy; i++) {
	    server[i].reclist.add(dscudaMemcpyH2DId, &args);	    
	}
    }

    WARN( 4, "   } libdscuda:%s().\n", __func__);
    return err;
}

/*
 * cudaMemcpy( DeviceToHost )
 */
cudaError_t RCServer::cudaMemcpyD2H(void *h_ptr, const void *v_ptr, size_t count) {
    cudaMemcpyArgs args;

    switch ( St.ft_mode ) {
    case FT_PLAIN: //thru
    case FT_REDUN: //thru
    case FT_MIGRA: //thru
    case FT_BOTH:
	args.dst   = (void *)h_ptr;
	args.src   = (void *)v_ptr;// d_ptr;
	args.count = count;
	args.kind  = cudaMemcpyDeviceToHost;
	reclist.add(dscudaMemcpyD2HId, (void *)&args);
	break;
    default:
	WARN( 0, "Unexpected failure.\n");
	exit( EXIT_FAILURE );
    }
    
    //<-- RPC communication.
    dscudaMemcpyD2HResult *rp;
    //<-- Translate virtual d_ptr to real d_ptr.
    void *h_lptr = memlist.queryHostPtr(v_ptr);
    void *d_ptr  = memlist.queryDevicePtr(v_ptr);
    WARN(4, "      + Physical[%d]:cudaMemcpy( dst=%p, src=%p, count=%zu )\n",
	 id, h_lptr, d_ptr, count);

    if (d_ptr == NULL) {
	WARN(0, "%s():d_ptr = NULL.\n", __func__);
	exit(1);
    }
    if (h_lptr == NULL) {
	WARN(0, "%s():h_lptr = NULL.\n", __func__);
	exit(1);
    }
    //--> Translate virtual d_ptr to real d_ptr.

    rp = dscudamemcpyd2hid_1((RCadr)d_ptr, count, Clnt);

    //<--- RPC fault check.
    struct rpc_err rpc_error;
    cudaError_t    cuda_error;
    clnt_geterr(Clnt, &rpc_error);
    if (rpc_error.re_status == RPC_SUCCESS) {
	if (rp == NULL) {
	    WARN( 0, "NULL pointer returned, %s:%s():L%d.\nexit.\n\n\n", __FILE__, __func__, __LINE__ );
	    clnt_perror(Clnt, ip);
	    exit(EXIT_FAILURE);
	} else {
	    cuda_error = (cudaError_t)rp->err;
	}
    } else {
	rpcErrorHook(&rpc_error);	
    }
    //--> RPC fault check.

    memcpy(h_lptr, rp->buf.RCbuf_val, rp->buf.RCbuf_len);
    xdr_free( (xdrproc_t)xdr_dscudaMemcpyD2HResult, (char *)rp );
    //--> RPC communication.

    return cuda_error;
}

cudaError_t VirDev_t::cudaMemcpyD2H(void *dst, const void *src, size_t count) {
    WARN( 4, "   Virtual[%d]:%s() {\n", id, __func__);

    int matched_count   = 0;
    int unmatched_count = 0;
    int recall_result;

    RCServer_t *failed_1st;
    //    int fail_flag[RC_NVDEVMAX]={0};
    cudaError_t err = cudaSuccess;
    /*
     * Record called history of CUDA APIs.
     */
    struct CudaMemcpyArgs_t args;
    if (ft_mode==FT_REDUN || ft_mode==FT_MIGRA || ft_mode==FT_BOTH) {
	args.dst   = (void *)dst;
	args.src   = (void *)src;
	args.count = count;
	args.kind  = cudaMemcpyDeviceToHost;
	this->reclist.add(dscudaMemcpyD2HId, &args);
	
	for (int i=0; i<nredundancy; i++) {
	    server[i].reclist.add(dscudaMemcpyD2HId, &args);
	}
    }
    /* Get the data from remote GPU(s), then verify */
    for (int i=0; i<nredundancy; i++) {
	server[i].cudaMemcpyD2H(dst, src, count);
    }
    
#if 0
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
#endif
    int memcmp_ret;
    int all_matched = 1;
    
    switch (conf) {
    case VDEV_MONO:
	memcpy( dst, server[0].memlist.queryHostPtr(src), count );
	if (( ft_mode==FT_REDUN || ft_mode==FT_MIGRA || ft_mode==FT_BOTH ) && (St.isHistoCalling()==0 )) {
	    //BKUPMEM.updateRegion( src, dst, count );
	}
	break;
    case VDEV_POLY:
	for (int i=0; i<nredundancy-1; i++) {
	    for (int k=i+1; k<nredundancy; k++) {
		memcmp_ret = memcmp(server[i].memlist.queryHostPtr(src), server[k].memlist.queryHostPtr(src), count);
		if (memcmp_ret == 0) {
		    server[k].stat_correct++;
		} else {
		    server[k].stat_error++;
		    all_matched = 0;
		}
		WARN(2, "   UNMATCHED redundant device %d/%d with device 0. %s()\n", i, nredundancy - 1, __func__);
	    }
	}
	
	if ( all_matched==1 ) {
	    WARN(5, "   #\\(^_^)/ All %d Redundant device(s) matched. statics OK/NG = %d/%d.\n",
		 nredundancy-1, matched_count, unmatched_count);
	    memcpy( dst, server[0].memlist.queryHostPtr(src), count );
	    /*
	     * Update backuped memory region.
	     */
	    if (( St.ft_mode==FT_REDUN || St.ft_mode==FT_MIGRA || St.ft_mode==FT_BOTH ) && (St.isHistoCalling()==0 )) {
		//memlist.updateRegion(src, dst, count); /* mirroring copy. !!!src and dst is swapped!!! */
	    }
	} else {
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
cudaMemcpyP2P(void *dst, int ddev, const void *src, int sdev, size_t count) {
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
cudaError_t cudaMemcpy(void *dst, const void *src,
		       size_t count, enum cudaMemcpyKind kind) {
    RCuva      *suva, *duva;
    int         dev0;
    cudaError_t err  = cudaSuccess;

    void *lsrc = dscudaAdrOfUva((void *)src);
    void *ldst = dscudaAdrOfUva(dst);

    int         vdevid = Vdevid[ vdevidIndex() ];
    Vdev_t     *vdev   = St.Vdev + vdevid;
    
    switch ( kind ) {
    case cudaMemcpyDeviceToHost:
	WARN(3, "cudaMemcpy(%p, %p, %zu, D->H) called vdevid=%d...\n",
	     ldst, lsrc, count, vdevid);
	// Avoid conflict between CheckPointing thread.
	pthread_mutex_lock( &cudaMemcpyD2H_mutex );
	err = vdev->cudaMemcpyD2H(dst, src, count);
	pthread_mutex_unlock( &cudaMemcpyD2H_mutex ); 
	break;
    case cudaMemcpyHostToDevice:
	WARN(3, "cudaMemcpy(%p, %p, %zu, H->D) called\n", ldst, lsrc, count);
	// Avoid conflict with CheckPointing thread.	
	pthread_mutex_lock( &cudaMemcpyH2D_mutex );
	err = vdev->cudaMemcpyH2D(dst, src, count);
	pthread_mutex_unlock( &cudaMemcpyH2D_mutex );
	break;
    case cudaMemcpyDeviceToDevice:
	WARN(3, "cudaMemcpy(%p, %p, %zu, DeviceToDevice) called\n", ldst, lsrc, count);
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
    WARN(3, "} %s().\n", __func__);
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
cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
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

/*
 * LoadModule Management. 
 */
int RCServer::findModuleOpen(void) {
    int i;
    for (i=0; i<RC_NKMODULEMAX; i++) {
	if ( modulelist[i].valid != 1 ) {
	    return i;
	}
    }
    WARN(0, "%s():Module management array is full. and exit.\n", __func__);
    exit(EXIT_FAILURE);
}

int RCServer::queryModuleID(int module_index) {
    for (int i=0; i<RC_NKMODULEMAX; i++) {
	if ( modulelist[i].index == module_index ) {
	    return modulelist[i].id;
	}
    }
    WARN(0, "%s():Not found Module Index in array. and exit.\n", __func__);
    exit(EXIT_FAILURE);
}

int VirDev_t::findModuleOpen(void) { //TODO: almost same as above func.
    int i;
    for (i=0; i<RC_NKMODULEMAX; i++) {
	if ( modulelist[i].valid != 1 ) {
	    return i;
	}
    }
    WARN(0, "%s():Module management array is full. and exit.\n", __func__);
    exit(EXIT_FAILURE);
}

int RCServer::loadModule(unsigned int ipaddr, pid_t pid, char *modulename,
			 char *modulebuf, int module_index) {
    WARN(5, "      + RCServer::%s(modulename=%s, module_index=%d) { \n",
	 __func__, modulename, module_index);
    
    /* send to virtual GPU */
    dscudaLoadModuleResult *rp;
    rp = dscudaloadmoduleid_1(ipaddr, getpid(), modulename, modulebuf, Clnt);

    //<--- RPC Error Hook
    struct rpc_err rpc_error;
    clnt_geterr(Clnt, &rpc_error );
    if (rpc_error.re_status == RPC_SUCCESS) {
	if (rp == NULL) {
	    WARN( 0, "NULL pointer returned, %s(). exit.\n", __func__ );
	    clnt_perror(Clnt, ip);
	    exit(EXIT_FAILURE);
	}
    } else {
	rpcErrorHook(&rpc_error);
    }
    //---> RPC Error Hook.

    int module_id = rp->id;
    xdr_free((xdrproc_t)xdr_dscudaLoadModuleResult, (char *)rp);

    // register a new module into the list,
    // and then, return a module id assigned by the server.
    int n = this->findModuleOpen(); 
    modulelist[n].valid     = 1;
    modulelist[n].index     = module_index;
    modulelist[n].id        = module_id;
    modulelist[n].sent_time = time(NULL);
    modulelist[n].ptx_data  = PtxStore.query(modulename);
    WARN(5, "RCServer[%d]: New client module item was registered. id:%d\n", id, module_id);
    
    if (St.isAutoVerb() ) {
	/*Nop*/
    }

    WARN(5, "      + } // RCServer::%s()\n", __func__);
    return module_id;
}

int VirDev_t::loadModule(char *name, char *strdata) {
    WARN(5, "   + VirDev_t::loadModule( name=%p(%s), strdata=%p ) {\n", name, name, strdata);

    if (name != NULL) {
#if RC_CACHE_MODULE
	// look for modulename in the module list.
	for (int i=0; i<RC_NKMODULEMAX; i++) {
	    if ( modulelist[i].isInvalid() ) {
		continue;
	    }
	    if ( strcmp(name, modulelist[i].ptx_data->name) == 0 ) { //Found
		if ( modulelist[i].isAlive() ) {
		    WARN(5, "done. found a cached one. id:%d  age:%d  name:%s\n",
			 modulelist[i].index, time(NULL) - modulelist[i].sent_time, modulelist[i].ptx_data->name);
		    return modulelist[i].index; // module found. i.e, it's already loaded.
		} else {
		    WARN(5, "found a cached one with id:%d, but it is too old (age:%d). resend it.\n",
			 modulelist[i].index, time(NULL) - modulelist[i].sent_time);
		    modulelist[i].invalidate(); // invalidate the cache.
		}
	    }
	}
#endif // RC_CACHE_MODULE
    } else {
	WARN(5, "VirDev_t::loadModule(%p) modulename:-\n", name);
    }

    //<---
    char *strdata_found = NULL;
    char *name_found=NULL;
    if (name==NULL && strdata==NULL) {
        for (int i=0; i<RC_NKMODULEMAX; i++) {
	    WARN(10, "i=%d\n", i);
	    if (modulelist[i].isInvalid()) continue;
	    if (!strcmp(name, modulelist[i].ptx_data->name)) {     /* matched */
		strdata_found = modulelist[i].ptx_data->ptx_image;
		name_found = modulelist[i].ptx_data->name;
		break;
	    }
	}
    } else {
	strdata_found = strdata;
	name_found = name;
    }
    //--->

    // module not found in the module list.
    // really need to send it to the server.

    // <-- If target .ptx is not registered to PtxStore, then register first.
    PtxRecord_t *ptxrecord_ptr = PtxStore.query(name_found);
    if (ptxrecord_ptr == NULL) {
	PtxStore.add( name_found, strdata_found );
    }
    // --> If target .ptx is not registered to PtxStore, then register first.
    
    int j = this->findModuleOpen();
    this->modulelist[j].index     = j;
    this->modulelist[j].id        = j; //dummy; not used.
    this->modulelist[j].valid     = 1;
    this->modulelist[j].sent_time = time(NULL);
    this->modulelist[j].ptx_data  = PtxStore.query(name_found);
    PtxStore.print(4);
    WARN(5, "      + New client-module item was registered. index=%d\n", j);

    int mid;
    for (int i=0; i<nredundancy; i++) {
	mid = server[i].loadModule(St.getIpAddress(), getpid(), name_found, strdata_found, j);
        WARN(3, "(info) server[%d].loadModule() returns mid=%d.\n", i, mid);
    }

    WARN(5, "   + } // VirDev_t::loadModule().\n");
    return modulelist[j].index;
}//VirDev_t::loadModule(

/*
 * launch a kernel function of id 'kid', defined in a module of id 'moduleid'.
 * 'kid' must be unique inside a single module.
 */
void RCServer::launchKernel(int module_index, int kid, char *kname,
			    RCdim3 gdim, RCdim3 bdim, RCsize smemsize,
			    RCstream stream, RCargs args) {
    WARN(5, "      + RCServer[%d]::%s() {\n", id, __func__);
    RCargs lo_args;
    lo_args.RCargs_len = args.RCargs_len;
    lo_args.RCargs_val = (RCarg *)malloc(args.RCargs_len * sizeof(RCarg));
    if (lo_args.RCargs_val == NULL) {
	WARN(0, "%s():malloc() failed.\n", __func__);
	exit(1);
    }
    for (int k=0; k<lo_args.RCargs_len; k++) {
	lo_args.RCargs_val[k] = args.RCargs_val[k];
    }
    
    RCstreamArray *st = RCstreamArrayQuery((cudaStream_t)stream);
    if (!st) {
        WARN(0, "invalid stream : %p\n", stream);
        exit(1);
    }

    /*
     * Replace v_ptr to d_ptr int args.
     */
    RCarg *argp;
    void  *v_ptr;
    void  *d_ptr;
    for (int i=0; i<lo_args.RCargs_len; i++) {
        argp = &(lo_args.RCargs_val[i]);
	if (argp->val.type == dscudaArgTypeP) {
            v_ptr = (void*)(argp->val.RCargVal_u.address);
	    d_ptr = memlist.queryDevicePtr(v_ptr);
	    WARN(6, "      +    Virtual Address Translate: arg[%d]:v_ptr=%p -> d_ptr=%p\n", i, v_ptr, d_ptr);
	    argp->val.RCargVal_u.address = (RCadr)d_ptr;
	}
    }
    /*
     * Replace module_index with real module id.
     */
    int moduleid = queryModuleID(module_index);

    void *rp = dscudalaunchkernelid_1(moduleid, kid, kname, gdim, bdim,
				      smemsize, (RCstream)st->s[id], lo_args, Clnt);

    //<--- Timed Out
    struct rpc_err rpc_error;
    
    clnt_geterr(Clnt, &rpc_error);
    if (rpc_error.re_status == RPC_SUCCESS) {
	if (rp == NULL) {
	    WARN( 0, "NULL pointer returned, %s(). exit.\n", __func__ );
	    clnt_perror(Clnt, ip);
	    exit( EXIT_FAILURE );
	}
    } else {
	rpcErrorHook(&rpc_error);
    }
    //--->
    free(lo_args.RCargs_val);
    WARN(5, "      + } RCServer[%d]::%s()\n", id, __func__);
}

void VirDev_t::launchKernel(int module_index, int kid, char *kname,
			    RCdim3 gdim, RCdim3 bdim, RCsize smemsize,
			    RCstream stream, RCargs args) {
    WARN(5, "   + VirDev_t::%s() {\n", __func__);
    /*     
     * Automatic Recovery, Register to the called history.
     */
    if (St.isAutoVerb() ) {
        cudaRpcLaunchKernelArgs args2;
        args2.moduleid = module_index;
        args2.kid      = kid;
        args2.kname    = kname;
        args2.gdim     = gdim;
        args2.bdim     = bdim;
        args2.smemsize = smemsize;
        args2.stream   = stream;
        args2.args     = args;
        //HISTREC.add( dscudaLaunchKernelId, (void *)&args2 );
    }

    for (int i=0; i<nredundancy; i++) {
        server[i].launchKernel(module_index, kid, kname, gdim,
			       bdim, smemsize, stream, args);
    }
    WARN(5, "   + } // VirDev_t::%s()\n", __func__);
}

void
rpcDscudaLaunchKernelWrapper(int module_index, int kid, char *kname,  /* moduleid is got by "dscudaLoadModule()" */
                             RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
                             RCargs args) {
    WARN(5, "%s() {\n", __func__);
    pthread_mutex_lock( &cudaKernelRun_mutex ); // Avoid conflict with CheciPointing.p

    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    vdev->launchKernel(module_index, kid, kname, gdim, bdim, smemsize, stream, args);
    
    pthread_mutex_unlock( &cudaKernelRun_mutex ); // Avoid conflict with CheciPointing.
    WARN(5, "} %s().\n", __func__)
    WARN(5, "\n")
}

cudaError_t
cudaMallocArray(struct cudaArray **array, const struct cudaChannelFormatDesc *desc,
                size_t width, size_t height, unsigned int flags) {
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

cudaError_t cudaFreeArray(struct cudaArray *array) {
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
                  size_t count, enum cudaMemcpyKind kind) {
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

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
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

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width, size_t height) {
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
cudaMemset2D(void *devPtr, size_t pitch, int value, size_t width, size_t height) {
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

cudaError_t cudaMallocHost(void **ptr, size_t size) {
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

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
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

cudaError_t cudaFreeHost(void *ptr) {
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

cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
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
cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {
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
                        size_t count, size_t offset, int vdevid, int raidid) {
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
                        size_t count, size_t offset, int vdevid, int raidid) {
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
                          size_t count, size_t offset, int vdevid, int raidid) {
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
                          size_t count, size_t offset, int vdevid, int raidid) {
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
                             size_t count, size_t offset, RCstream stream, int vdevid, int raidid) {
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
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid) {
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
                               size_t count, size_t offset, RCstream stream, int vdevid, int raidid) {
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

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
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

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
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

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
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

cudaError_t cudaStreamQuery(cudaStream_t stream) {
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

cudaError_t cudaEventCreate(cudaEvent_t *event) {
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
cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
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
cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
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

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
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

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
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
cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
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
cufftExecC2R(cufftHandle plan, cufftComplex *idata, cufftComplex *odata) {
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
cufftExecZ2Z(cufftHandle plan, cufftComplex *idata, cufftComplex *odata, int direction) {
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
cufftExecZ2D(cufftHandle plan, cufftComplex *idata, cufftComplex *odata) {
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
cufftSetCompatibilityMode(cufftHandle plan, cufftCompatibility mode) {
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
cublasCreate_v2(cublasHandle_t *handle) {
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
cublasDestroy_v2(cublasHandle_t handle) {
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
cublasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy) {
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
cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy) {
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
               const float *B, int ldb, const float *beta, float *C, int ldc) {
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

