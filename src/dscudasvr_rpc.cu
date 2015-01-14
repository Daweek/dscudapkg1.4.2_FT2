//                             -*- Mode: C++ -*-
// Filename         : dacudasvr_rpc.cu
// Description      : DS-CUDA server node library for Remote Procedure Call.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-21 16:19:26
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <errno.h>
#include <rpc/pmap_clnt.h>
#include <cutil.h>
// remove definition of some macros which will be redefined in \"cutil_inline.h\".
#ifdef MIN
#undef MIN
#endif
#ifdef MAX
#undef MAX
#endif
#include <cutil_inline.h>
#include "dscuda.h"
#include "libdscuda.h"
#include "dscudadefs.h"
#include "dscudautil.h"
#include "dscudarpc.h"
#include "dscudasvr.h"
#include "dscudasvr_rpc.h"

/*
 * A thread to watch over the socket inherited from the daemon,
 * in order to detect disconnection by the client.
 * exit() immediately, if detected.
 */
static void *
rpcWatchDisconnection(void *arg) {
    int clientsock = *(int *)arg;
    int nrecvd;
    char buf[16];

#if 0 // oikawa, temporary
    sleep(5); // wait long enough so that connection is certainly establised.
#else
    sleep(59); // wait long enough so that connection is certainly establised.
#endif
    
    SWARN(3, "start socket polling:%d.\n", clientsock);
    for (;;) {
        // nrecvd = recv(clientsock, buf, 1, MSG_PEEK | MSG_DONTWAIT);
        nrecvd = recv(clientsock, buf, 1, MSG_PEEK);
#if 1 // debug
	SWARN(3, "recv(clientsock, buf, 1, MSG_PEEK);\n");
#endif
        if (nrecvd == 0) {
            SWARN(2, "disconnected.\n");
            exit(0);
        } else if (nrecvd == -1) {
            if (errno == ENOTCONN) {
                SWARN(0, "disconnected by peer.\n");
                exit(1);
            } else {
                perror("dscudasvr_rpc:rpcWatchDisconnection:");
                exit(1);
            }
        }
        SWARN(2, "got %d-byte side-band message from the client.\n", nrecvd);
    }

    return NULL;
}

int
rpcUnpackKernelParam(CUfunction *kfuncp, RCargs *argsp) {
    CUresult cuerr;
    CUfunction kfunc = *kfuncp;
    int ival;
    float fval;
    void *pval;
    RCarg noarg;
    RCarg *argp = &noarg;
    FaultConf *fault_conf; // moikawa add

    noarg.offset = 0;
    noarg.size = 0;

    SWARN(10, "argsp->RCargs_len = %d\n", argsp->RCargs_len);
    for (int i = 0; i < argsp->RCargs_len; i++) {
        argp = &(argsp->RCargs_val[i]);

        switch (argp->val.type) {
          case dscudaArgTypeP:
            pval = (void*)&(argp->val.RCargVal_u.address);
            cuerr = cuParamSetv(kfunc, argp->offset, pval, argp->size);
	    if (cuerr == CUDA_SUCCESS) {
                SWARN(0, "(P)cuParamSetv(%p, %d, %p, %d) success.\n",
                     kfunc, argp->offset, pval, argp->size);
	    } else if (cuerr != CUDA_SUCCESS) {
                SWARN(0, "(P)cuParamSetv(%p, %d, %p, %d) failed. %s\n",
                     kfunc, argp->offset, pval, argp->size,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeI:
            ival = argp->val.RCargVal_u.valuei;
            cuerr = cuParamSeti(kfunc, argp->offset, ival);
	    if (cuerr == CUDA_SUCCESS) {
                SWARN(0, "(I)cuParamSeti(%p, %d, %d) success.\n",
                     kfunc, argp->offset, ival);
	    } else if (cuerr != CUDA_SUCCESS) {
                SWARN(0, "(I)cuParamSeti(%p, %d, %d) failed. %s\n",
                     kfunc, argp->offset, ival,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeF:
            fval = argp->val.RCargVal_u.valuef;
            cuerr = cuParamSetf(kfunc, argp->offset, fval);
	    if (cuerr == CUDA_SUCCESS) {
                SWARN(5, "(F)cuParamSetf(%p, %d, %f) success.\n",
                     kfunc, argp->offset, fval);
	    } else if (cuerr != CUDA_SUCCESS) {
                SWARN(0, "(F)cuParamSetf(%p, %d, %f) failed. %s\n",
                     kfunc, argp->offset, fval,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          case dscudaArgTypeV: /*Structure, etc.*/
            pval = argp->val.RCargVal_u.valuev;
	    /*if environ var found, then update its value with localhost's one.*/
	    fault_conf = (FaultConf *)pval;
	    if (strncmp(fault_conf->tag, IDTAG_0, 32)==0) {
                SWARN(10, "DSCUDA_FAULT_INJECTION found, ");
		if (fault_conf->overwrite_en) {
		    SWARN(10, "then overwrite %d over %d.\n",
			 DscudaSvr.getFaultInjection(), fault_conf->fault_en);
		    fault_conf->fault_en = DscudaSvr.getFaultInjection();
		} else {
		    SWARN(10, "but leave as is %d.\n", fault_conf->fault_en);
		}
	    }
	    
            cuerr = cuParamSetv(kfunc, argp->offset, pval, argp->size);
	    if (cuerr == CUDA_SUCCESS) {
                SWARN(0, "(V)cuParamSetv(%p, %d, %p, %d) success.\n",
                     kfunc, argp->offset, pval, argp->size);
	    } else if (cuerr != CUDA_SUCCESS) {
                SWARN(0, "(V)cuParamSetv(%p, %d, %p, %d) failed. %s\n",
                     kfunc, argp->offset, pval, argp->size,
                     cudaGetErrorString((cudaError_t)cuerr));
                fatal_error(1);
            }
            break;

          default:
            SWARN(0, "rpcUnpackKernelParam: invalid RCargType\n", argp->val.type);
            fatal_error(1);
        }
    }
    return argp->offset + argp->size;
}

void
setupRpc(void) {
    register SVCXPRT *transp;
    unsigned long int prog = DSCUDA_PROG;
    pthread_t tid;
    static int sock; // must be static since refered by the poller.

    // TCP w/o portmapper, i.e., a fixed port.
    pmap_unset (prog, DSCUDA_VER);
    sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    SWARN(3, "rpc sock:%d\n", sock);

    if (sock == -1) {
        perror("dscudasvr_rpc:setupRpc:socket()");
        exit(1);
    }

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(TcpPort);
    //    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // listen only on 127.0.0.1
    addr.sin_addr.s_addr = htonl(INADDR_ANY);

    /* <-- For avoiding TIME_WAIT status on TCP port. */
    bool yes=1;
    setsockopt( sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));
    /* --> For avoiding TIME_WAIT status on TCP port. */

    if (bind(sock, (struct sockaddr *) &addr, sizeof addr) == -1) {
        perror("dscudasvr_rpc:bind");
        exit(1);
    }
    transp = svctcp_create(sock, RC_BUFSIZE, RC_BUFSIZE);
    if (transp == NULL) {
        fprintf (stderr, "%s", "cannot create tcp service.");
        exit(1);
    }
    if (!svc_register(transp, prog, DSCUDA_VER, dscuda_prog_1, 0)) {
        fprintf (stderr, "unable to register (prog:0x%lx DSCUDA_VER:%d, TCP).\n",
                 prog, DSCUDA_VER);
        exit(1);
    }
    if (D2Csock >= 0) {
        pthread_create(&tid, NULL, rpcWatchDisconnection, &D2Csock);
    }
}

/*
 * CUDA API stubs
 */

/*
 * Thread Management
 */

dscudaResult *
dscudathreadexitid_1_svc(struct svc_req *sr) {
    cudaError_t err;
    static dscudaResult res;

    SWARN(3, "cudaThreadExit(\n");
    if (!dscuContext) createDscuContext();

    err = cudaThreadExit();
    check_cuda_error(err);
    res.err = err;
    SWARN(3, ") done.\n");
    return &res;
}

dscudaResult *
dscudathreadsynchronizeid_1_svc(struct svc_req *sr) {
    cudaError_t err;
    static dscudaResult res;

    SWARN(3, "cudaThreadSynchronize(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadSynchronize();
    check_cuda_error(err);
    res.err = err;
    SWARN(3, ") done.\n");
    return &res;
}

dscudaResult *
dscudathreadsetlimitid_1_svc(int limit, RCsize value, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    SWARN(3, "cudaThreadSetLimit(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadSetLimit((enum cudaLimit)limit, value);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%d, %d) done.\n", limit, value);
    return &res;
}

dscudaThreadGetLimitResult *
dscudathreadgetlimitid_1_svc(int limit, struct svc_req *sr) {
    cudaError_t err;
    static dscudaThreadGetLimitResult res;
    size_t value;

    SWARN(3, "cudaThreadGetLimit(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadGetLimit(&value, (enum cudaLimit)limit);
    check_cuda_error(err);
    res.err = err;
    res.value = value;
    SWARN(3, "%p, %d) done.  value:%zu\n", &value, limit, value);
    return &res;
}

dscudaResult *
dscudathreadsetcacheconfigid_1_svc(int cacheConfig, struct svc_req *sr) {
    cudaError_t err;
    static dscudaResult res;

    SWARN(3, "cudaThreadSetCacheConfig(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadSetCacheConfig((enum cudaFuncCache)cacheConfig);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%d) done.\n", cacheConfig);
    return &res;
}

dscudaThreadGetCacheConfigResult *
dscudathreadgetcacheconfigid_1_svc(struct svc_req *sr) {
    cudaError_t err;
    static dscudaThreadGetCacheConfigResult res;
    int cacheConfig;

    SWARN(3, "cudaThreadGetCacheConfig(");
    if (!dscuContext) createDscuContext();

    err = cudaThreadGetCacheConfig((enum cudaFuncCache *)&cacheConfig);
    check_cuda_error(err);
    res.err = err;
    res.cacheConfig = cacheConfig;
    SWARN(3, "%p) done.  cacheConfig:%d\n", &cacheConfig, cacheConfig);
    return &res;
}


/*
 * Error Handling
 */

dscudaResult *
dscudagetlasterrorid_1_svc(struct svc_req *sr) {
    cudaError_t err;
    static dscudaResult res;

    SWARN(5, "cudaGetLastError(");
    if (!dscuContext) createDscuContext();

    err = cudaGetLastError();
    check_cuda_error(err);
    res.err = err;
    SWARN(5, ") done.\n");
    return &res;
}

dscudaResult *
dscudapeekatlasterrorid_1_svc(struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;

    SWARN(5, "cudaPeekAtLastError(");
    if (!dscuContext) createDscuContext();

    err = cudaPeekAtLastError();
    check_cuda_error(err);
    res.err = err;
    SWARN(5, ") done.\n");
    return &res;
}

dscudaGetErrorStringResult *
dscudageterrorstringid_1_svc(int err, struct svc_req *sr) {
    static dscudaGetErrorStringResult res;

    SWARN(3, "cudaGetErrorString(");
    if (!dscuContext) createDscuContext();

    res.errmsg = (char *)cudaGetErrorString((cudaError_t)err);
    SWARN(3, "%d) done.\n", err);
    return &res;
}
/*
 * Device Management
 */
dscudaGetDeviceResult *
dscudagetdeviceid_1_svc(struct svc_req *sr) {
    cudaError_t err;
    int device;
    static dscudaGetDeviceResult res;

    SWARN(3, "cudaGetDevice(");
    if (!dscuContext) createDscuContext();

    err = cudaGetDevice(&device);
    check_cuda_error(err);
    res.device = Devid2Vdevid[device];
    res.err = err;
    SWARN(3, "0x%08lx) done. device:%d  virtual device:%d\n",
         (unsigned long)&device, device, res.device);
    return &res;
}
dscudaGetDeviceCountResult *
dscudagetdevicecountid_1_svc(struct svc_req *sr) {
    int count;
    static dscudaGetDeviceCountResult res;

    SWARN(3, "cudaGetDeviceCount(");

#if 0
// this returns # of devices in the system, even if the number of
// valid devices set by cudaSetValidDevices() is smaller.
    cudaError_t err;
    err = cudaGetDeviceCount(&count);
    check_cuda_error(err);
    res.count = count;
    res.err = err;
#else
    res.count = count = Ndevice;
    res.err = cudaSuccess;
#endif
    SWARN(3, "0x%08lx) done. count:%d\n", (unsigned long)&count, count);
    return &res;
}

dscudaGetDevicePropertiesResult *
dscudagetdevicepropertiesid_1_svc(int device, struct svc_req *sr) {
    cudaError_t err;
    static int firstcall = 1;
    static dscudaGetDevicePropertiesResult res;

    SWARN(3, "cudaGetDeviceProperties(");

    if (firstcall) {
        firstcall = 0;
        res.prop.RCbuf_val = (char*)malloc(sizeof(cudaDeviceProp));
        res.prop.RCbuf_len = sizeof(cudaDeviceProp);
    }
    if (1 < Ndevice) {
        SWARN(0, "dscudagetdevicepropertiesid_1_svc() cannot handle multiple devices for now. Ndevice:%d\n",
             Ndevice);
        exit(1);
    }
    err = cudaGetDeviceProperties((cudaDeviceProp *)res.prop.RCbuf_val, Devid[0]);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08lx, %d) done.\n", (unsigned long)res.prop.RCbuf_val, Devid[0]);
    return &res;
}

dscudaDriverGetVersionResult *
dscudadrivergetversionid_1_svc(struct svc_req *sr) {
    cudaError_t err;
    int ver;
    static dscudaDriverGetVersionResult res;

    SWARN(3, "cudaDriverGetVersion(");

    if (!dscuContext) createDscuContext();

    err = cudaDriverGetVersion(&ver);
    check_cuda_error(err);
    res.ver = ver;
    res.err = err;
    SWARN(3, "0x%08lx) done.\n", (unsigned long)&ver);
    return &res;
}

dscudaRuntimeGetVersionResult *
dscudaruntimegetversionid_1_svc(struct svc_req *sr) {
    cudaError_t err;
    int ver;
    static dscudaRuntimeGetVersionResult res;

    SWARN(3, "cudaRuntimeGetVersion(");

    if (!dscuContext) createDscuContext();

    err = cudaRuntimeGetVersion(&ver);
    check_cuda_error(err);
    res.ver = ver;
    res.err = err;
    SWARN(3, "0x%08lx) done.\n", (unsigned long)&ver);
    return &res;
}

dscudaResult *
dscudasetdeviceid_1_svc(int device, struct svc_req *sr) {
    cudaError_t err;
    static dscudaResult res;

    SWARN(3, "cudaSetDevice(");

    if (dscuContext) destroyDscuContext();

    dscuDevice = Devid[device];
    err = createDscuContext();
    res.err = err;
    SWARN(3, "%d) done.  dscuDevice: %d\n",
         device, dscuDevice);
    return &res;
}

dscudaResult *
dscudasetdeviceflagsid_1_svc(unsigned int flags, struct svc_req *sr) {
    cudaError_t err;
    static dscudaResult res;

    SWARN(3, "cudaSetDeviceFlags(");

    /* cudaSetDeviceFlags() API should be called only when
     * the device is not active, i.e., dscuContext does not exist.
     * Before invoking the API, destroy the context if valid. */
    if (dscuContext) destroyDscuContext();

    err = cudaSetDeviceFlags(flags);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08x)\n", flags);

    return &res;
}

dscudaChooseDeviceResult *
dscudachoosedeviceid_1_svc(RCbuf prop, struct svc_req *sr) {
    cudaError_t err;
    int device;
    static dscudaChooseDeviceResult res;

    SWARN(3, "cudaGetDevice(");
    if (!dscuContext) createDscuContext();

    err = cudaChooseDevice(&device, (const struct cudaDeviceProp *)&prop.RCbuf_val);
    check_cuda_error(err);
    res.device = Devid2Vdevid[device];
    res.err = err;
    SWARN(3, "0x%08lx) done. device:%d  virtual device:%d\n",
         (unsigned long)&device, device, res.device);
    return &res;
}


dscudaResult *
dscudadevicesynchronize_1_svc(struct svc_req *sr) {
    cudaError_t err;
    static dscudaResult res;

    SWARN(3, "cudaDeviceSynchronize(");
    if (!dscuContext) createDscuContext();

    err = cudaDeviceSynchronize();
    check_cuda_error(err);
    res.err = err;
    SWARN(3, ") done.\n");

    return &res;
}

dscudaResult *
dscudadevicereset_1_svc(struct svc_req *sr) {
    cudaError_t err;
    bool all = true;
    static dscudaResult res;

    SWARN(3, "cudaDeviceReset(");
    if (!dscuContext) createDscuContext();

    err = cudaDeviceReset();
    check_cuda_error(err);
    res.err = err;
    releaseModules(all);
    SWARN(3, ") done.\n");

    return &res;
}

/*
 * Stream Management
 */

dscudaStreamCreateResult *
dscudastreamcreateid_1_svc(struct svc_req *sr) {
    static dscudaStreamCreateResult res;
    cudaError_t err;
    cudaStream_t stream;

    SWARN(3, "cudaStreamCreate(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamCreate(&stream);
    res.stream = (RCadr)stream;
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p) done. stream:%p\n", &stream, stream);

    return &res;
}
dscudaResult *
dscudastreamdestroyid_1_svc(RCstream stream, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaStreamDestroy(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamDestroy((cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx) done.\n", (long long)stream);

    return &res;
}

dscudaResult *
dscudastreamwaiteventid_1_svc(RCstream stream, RCevent event, unsigned int flags, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaStreamWaitEvent(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, flags);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx 0x%08llx, 0x%08x) done.\n", (long long)stream, (long long)event, flags);

    return &res;
}

dscudaResult *
dscudastreamsynchronizeid_1_svc(RCstream stream, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaStreamSynchronize(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamSynchronize((cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx) done.\n", (long long)stream);

    return &res;
}

dscudaResult *
dscudastreamqueryid_1_svc(RCstream stream, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaStreamQuery(");
    if (!dscuContext) createDscuContext();
    err = cudaStreamQuery((cudaStream_t)stream);
    // should not check error due to the nature of this API.
    // check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx) done.\n", (long long)stream);

    return &res;
}

/*
 * Event Management
 */

dscudaEventCreateResult *
dscudaeventcreateid_1_svc(struct svc_req *sr) {
    static dscudaEventCreateResult res;
    cudaError_t err;
    cudaEvent_t event;

    SWARN(3, "cudaEventCreate(");
    if (!dscuContext) createDscuContext();
    err = cudaEventCreate(&event);
    res.event = (RCadr)event;
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p) done. event:%p\n", &event, event);

    return &res;
}

dscudaEventCreateResult *
dscudaeventcreatewithflagsid_1_svc(unsigned int flags, struct svc_req *sr) {
    static dscudaEventCreateResult res;
    cudaError_t err;
    cudaEvent_t event;

    SWARN(3, "cudaEventCreateWithFlags(");
    if (!dscuContext) createDscuContext();
    err = cudaEventCreateWithFlags(&event, flags);
    res.event = (RCadr)event;
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, 0x%08x) done. event:0x%08llx\n", &event, flags, (long long)event);

    return &res;
}

dscudaResult *
dscudaeventdestroyid_1_svc(RCevent event, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaEventDestroy(");
    if (!dscuContext) createDscuContext();
    err = cudaEventDestroy((cudaEvent_t)event);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx) done.\n", (long long)event);

    return &res;
}

dscudaEventElapsedTimeResult *
dscudaeventelapsedtimeid_1_svc(RCevent start, RCevent end, struct svc_req *sr) {
    static dscudaEventElapsedTimeResult res;
    cudaError_t err;
    float millisecond;

    SWARN(3, "cudaEventElapsedTime(");
    if (!dscuContext) createDscuContext();
    err = cudaEventElapsedTime(&millisecond, (cudaEvent_t)start, (cudaEvent_t)end);
    check_cuda_error(err);
    res.ms = millisecond;
    res.err = err;
    SWARN(3, "%5.3f 0x%08llx 0x%08llx) done.\n", millisecond, (long long)start, (long long)end);

    return &res;
}

dscudaResult *
dscudaeventrecordid_1_svc(RCevent event, RCstream stream, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaEventRecord(");
    if (!dscuContext) createDscuContext();
    err = cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx 0x%08llx) done.\n", (long long)event, (long long)stream);

    return &res;
}
dscudaResult *
dscudaeventsynchronizeid_1_svc(RCevent event, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaEventSynchronize(");
    if (!dscuContext) createDscuContext();
    err = cudaEventSynchronize((cudaEvent_t)event);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx) done.\n", (long long)event);

    return &res;
}
dscudaResult *
dscudaeventqueryid_1_svc(RCevent event, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaEventQuery(");
    if (!dscuContext) createDscuContext();
    err = cudaEventQuery((cudaEvent_t)event);
    // should not check error due to the nature of this API.
    // check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx) done.\n", (long long)event);

    return &res;
}


dscudaFuncGetAttributesResult *
dscudafuncgetattributesid_1_svc(int moduleid, char *kname, struct svc_req *sr)
{
    static dscudaFuncGetAttributesResult res;
    CUresult err;
    CUfunction kfunc;

    if (!dscuContext) createDscuContext();

    err = getFunctionByName(&kfunc, kname, moduleid);
    check_cuda_error((cudaError_t)err);

    SWARN(3, "cuFuncGetAttribute(");
    err = cuFuncGetAttribute(&res.attr.binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kfunc);
    check_cuda_error((cudaError_t)err);
    SWARN(3, "%p, %d, %p) done.\n", &res.attr.binaryVersion, CU_FUNC_ATTRIBUTE_BINARY_VERSION, kfunc);

    err = cuFuncGetAttribute((int *)&res.attr.constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kfunc);
    check_cuda_error((cudaError_t)err);
    SWARN(3, "%p, %d, %p) done.\n", &res.attr.constSizeBytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, kfunc);

    err = cuFuncGetAttribute((int *)&res.attr.localSizeBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kfunc);
    check_cuda_error((cudaError_t)err);
    SWARN(3, "%p, %d, %p) done.\n", &res.attr.localSizeBytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, kfunc);

    err = cuFuncGetAttribute(&res.attr.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kfunc);
    check_cuda_error((cudaError_t)err);
    SWARN(3, "%p, %d, %p) done.\n", &res.attr.maxThreadsPerBlock, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kfunc);

    err = cuFuncGetAttribute(&res.attr.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, kfunc);
    check_cuda_error((cudaError_t)err);
    SWARN(3, "%p, %d, %p) done.\n", &res.attr.numRegs, CU_FUNC_ATTRIBUTE_NUM_REGS, kfunc);

    err = cuFuncGetAttribute(&res.attr.ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, kfunc);
    check_cuda_error((cudaError_t)err);
    SWARN(3, "%p, %d, %p) done.\n", &res.attr.ptxVersion, CU_FUNC_ATTRIBUTE_PTX_VERSION, kfunc);

    err = cuFuncGetAttribute((int *)&res.attr.sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kfunc);
    check_cuda_error((cudaError_t)err);
    SWARN(3, "%p, %d, %p) done.\n", &res.attr.sharedSizeBytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kfunc);

    res.err = err;

    return &res;
}
/*
 * Memory Management
 */
dscudaMallocResult *
dscudamallocid_1_svc(RCsize size, struct svc_req *sr) {
    static dscudaMallocResult res;
    cudaError_t err;
    int *devadr;

    SWARN(3, "cudaMalloc(");
#if 0 //force time out error
    sleep(60);
#endif
    if (!dscuContext) createDscuContext();
    err = cudaMalloc((void**)&devadr, size);
    SWARN0(3, "%p, %d) done. devadr:%p. return %d\n", &devadr, size, devadr, (int)err);
    res.devAdr = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
#if 1 //fill with zero for CheckPointing function.
    err = cudaMemset( devadr, 0, (size_t)size );
    SWARN(3, "cudaMemset( %p, %d ) done. return %d\n", devadr, size, (int)err);
    check_cuda_error(err);
#endif

    return &res;
}
dscudaResult *
dscudafreeid_1_svc(RCadr mem, struct svc_req *) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaFree(");
    if (!dscuContext) createDscuContext();
    err = cudaFree((void*)mem);
    check_cuda_error(err);
    res.err = err;
    SWARN0(3, "0x%08llx) done.\n", (long long)mem);

    return &res;
}
dscudaMemcpyH2HResult *
dscudamemcpyh2hid_1_svc(RCadr dst, RCbuf srcbuf, RCsize count, struct svc_req *sr) {
    static dscudaMemcpyH2HResult res;
    SWARN(0, "dscudaMemcpy() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}
dscudaResult *
dscudamemcpyh2did_1_svc(RCadr dst, RCbuf srcbuf, RCsize count, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaMemcpy(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpy((void*)dst, srcbuf.RCbuf_val, count, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN0(3, "0x%08llx, 0x%08lx, %d, %s) done.\n",
	 (long long)dst, (unsigned long)srcbuf.RCbuf_val, count, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}
dscudaMemcpyD2HResult *
dscudamemcpyd2hid_1_svc( RCadr src, RCsize count, int flag /*fault*/,
			 struct svc_req *sr ) {
    static RCsize maxcount = 0;
    static dscudaMemcpyD2HResult res;
    cudaError_t err;

    SWARN( 3, "cudaMemcpy(");
    if ( !dscuContext ) createDscuContext();
    if ( maxcount == 0 ) {
        res.buf.RCbuf_val = NULL;
    }
    if ( maxcount < count ) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpy(res.buf.RCbuf_val, (const void*)src, count, cudaMemcpyDeviceToHost);
    SWARN0( 3, "0x%08lx, 0x%08llx, %d, %s) done.\n",
         (unsigned long)res.buf.RCbuf_val, (long long)src, count, dscudaMemcpyKindName( cudaMemcpyDeviceToHost ));
    check_cuda_error(err);
    res.err = err;

//#if defined(FAULT_AT_D2H)
    //<--
    // !!! [debugging purpose only] destroy some part of the returning data
    // !!! in order to emulate a malfunctional GPU.
    const  int period_err_d2h = 10; //sec
    double period_err = (double)period_err_d2h;
    static double t_prev_err;
    double t_buf;
    double dt_ok;
    if (flag != 0) {
	static bool firstcall = true;
#if 1
	if (firstcall) {
	    firstcall = false;
	    dscuda::stopwatch( &t_prev_err ); // get time at firstcall.
	    SWARN(2, "################ Good data (1st call).\n" );
	}
	else {
	    t_buf = t_prev_err;
	    dt_ok = dscuda::stopwatch( &t_buf );
	    if ( dt_ok > period_err ) {
		SWARN(2, "################\n" );
		SWARN(2, "################\n" );
		SWARN(2, "################ Bad data generatad.\n" );
		SWARN(2, "################ (every %d sec)\n", period_err_d2h);
		SWARN(2, "################\n" );
		res.buf.RCbuf_val[0] = 123; // Overwrite with no mean bits.
		t_prev_err = t_buf;
	    }
	    else {
		SWARN(2, "################ Good data. (%5.1f/%d)\n", dt_ok, period_err_d2h );
	    }
	}
#else
        static bool err_prev_call = false; // avoid bad data generation in adjacent calls.
	const  double err_rate = 1.0 / 10.0; // 1.0 / 1000.0;
	
        if (firstcall) {
            firstcall = false;
            srand48( time(NULL));
        } 
        else if (!err_prev_call && (drand48() < err_rate)) {
            SWARN(2, "################\n" );
            SWARN(2, "################\n" );
            SWARN(2, "################ Bad data generatad.\n" );
            SWARN(2, "################\n" );
            SWARN(2, "################\n" );
            res.buf.RCbuf_val[0] = 123; // Overwrite with no mean bits.
            err_prev_call = true;
        }
	else {
            err_prev_call = false;
        }
#endif
    }
    //--> !!! [debugging purpose only] 
//#endif
    return &res;
}

dscudaResult *
dscudamemcpyd2did_1_svc(RCadr dst, RCadr src, RCsize count, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    SWARN(3, "cudaMemcpy(");
    err = cudaMemcpy((void *)dst, (void *)src, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx, 0x%08llx, %d, %s) done.\n",
	 (long long)dst, (long long)src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaMallocArrayResult *
dscudamallocarrayid_1_svc(RCchanneldesc desc, RCsize width, RCsize height, unsigned int flags, struct svc_req *sr)
{
    static dscudaMallocArrayResult res;
    cudaError_t err;
    cudaArray *devadr;
    cudaChannelFormatDesc descbuf = cudaCreateChannelDesc(desc.x, desc.y, desc.z, desc.w, (enum cudaChannelFormatKind)desc.f);

    SWARN(3, "cudaMallocArray(");
    if (!dscuContext) createDscuContext();
    err = cudaMallocArray((cudaArray**)&devadr, &descbuf, width, height, flags);
    res.array = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %p, %d, %d, 0x%08x) done. devadr:%p\n",
         &devadr, &descbuf, width, height, flags, devadr)

    return &res;
}

dscudaResult *
dscudafreearrayid_1_svc(RCadr array, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaFreeArray(");
    if (!dscuContext) createDscuContext();
    err = cudaFreeArray((cudaArray*)array);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx) done.\n", (long long)array);

    return &res;
}

dscudaMemcpyToArrayH2HResult *
dscudamemcpytoarrayh2hid_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize count, struct svc_req *sr)
{
    static dscudaMemcpyToArrayH2HResult res;
    SWARN(0, "dscudaMemcpyToArray() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpytoarrayh2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize count, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaMemcpyToArray(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpyToArray((cudaArray *)dst, wOffset, hOffset, src.RCbuf_val, count, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx, %d, %d, 0x%08lx, %d, %s) done.\n",
         (long long)dst, wOffset, hOffset, (unsigned long)src.RCbuf_val, count, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpyToArrayD2HResult *
dscudamemcpytoarrayd2hid_1_svc(RCsize wOffset, RCsize hOffset, RCadr src, RCsize count, struct svc_req *sr) {
    static dscudaMemcpyToArrayD2HResult res;
    SWARN(0, "dscudaMemcpyToArray() does not support cudaMemcpyDeviceToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpytoarrayd2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCadr src, RCsize count, struct svc_req *sr) {
    cudaError_t err;
    static dscudaResult res;
    SWARN(3, "cudaMemcpyToArray(");
    err = cudaMemcpyToArray((cudaArray *)dst, wOffset, hOffset, (void *)src, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %d, %d, %p, %d, %s) done.\n",
         (void *)dst, wOffset, hOffset, (void *)src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaResult *
dscudamemcpytosymbolh2did_1_svc(int moduleid, char *symbol, RCbuf src, RCsize count, RCsize offset, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SWARN(3, "cudaMemcpyToSymbol(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpy((char *)gsptr + offset, src.RCbuf_val, count, cudaMemcpyHostToDevice);
                             
    SWARN(3, "%p, 0x%08lx, %d, %d, %s) done. module name:%s  symbol:%s\n",
         gsptr, (unsigned long)src.RCbuf_val, count, offset,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice),
         SvrModulelist[moduleid].name, symbol);
    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpytosymbold2did_1_svc(int moduleid, char *symbol, RCadr src, RCsize count, RCsize offset, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SWARN(3, "cudaMemcpyToSymbol(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpy((char *)gsptr + offset, (void*)src, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx, %p, %d, %d, %s) done.\n",
         gsptr, (unsigned long)src, count, offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}

dscudaMemcpyFromSymbolD2HResult *
dscudamemcpyfromsymbold2hid_1_svc(int moduleid, char *symbol, RCsize count, RCsize offset, struct svc_req *sr) {
    static RCsize maxcount = 0;
    static dscudaMemcpyFromSymbolD2HResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SWARN(3, "cudaMemcpyFromSymbol(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpy(res.buf.RCbuf_val, (char *)gsptr + offset, count, cudaMemcpyDeviceToHost);
                             
    SWARN(3, "0x%08lx, %p, %d, %d, %s) done. module name:%s  symbol:%s\n",
         (unsigned long)res.buf.RCbuf_val, gsptr, count, offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost),
         SvrModulelist[moduleid].name, symbol);
    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpyfromsymbold2did_1_svc(int moduleid, RCadr dst, char *symbol, RCsize count, RCsize offset, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SWARN(3, "cudaMemcpyFromSymbol(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpy((void*)dst, (char *)gsptr + offset, count, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %p, %d, %d, %s) done.\n",
         (void *)dst, gsptr, count, offset,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}
dscudaResult *
dscudamemcpytosymbolasynch2did_1_svc(int moduleid, char *symbol, RCbuf src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SWARN(3, "cudaMemcpyToSymbolAsync(");
    if (!dscuContext) createDscuContext();
    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpyAsync((char *)gsptr + offset, src.RCbuf_val, count, cudaMemcpyHostToDevice, (cudaStream_t)stream);
                             
    SWARN(3, "%p, 0x%08lx, %d, %d, %s, %p) done. module name:%s  symbol:%s\n",
         gsptr, (unsigned long)src.RCbuf_val, count, offset,
         dscudaMemcpyKindName(cudaMemcpyHostToDevice), stream,
         SvrModulelist[moduleid].name, symbol);

    check_cuda_error(err);
    res.err = err;
    return &res;
}
dscudaResult *
dscudamemcpytosymbolasyncd2did_1_svc(int moduleid, char *symbol, RCadr src, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SWARN(3, "cudaMemcpyToSymbolAsync(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpyAsync((char *)gsptr + offset, (void*)src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %p, %d, %d, %s, 0x%08llx) done.\n",
         gsptr, (void *)src, count, offset, stream,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}

dscudaMemcpyFromSymbolAsyncD2HResult *
dscudamemcpyfromsymbolasyncd2hid_1_svc(int moduleid, char *symbol, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    static dscudaMemcpyFromSymbolAsyncD2HResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SWARN(3, "cudaMemcpyFromSymbolAsync(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);
    err = cudaMemcpyAsync(res.buf.RCbuf_val, (char *)gsptr + offset, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
                             
    SWARN(3, "0x%08lx, %p, %d, %d, %s, 0x%08llx) done. module name:%s  symbol:%s\n",
         (unsigned long)res.buf.RCbuf_val, gsptr, count, offset, stream,
         dscudaMemcpyKindName(cudaMemcpyDeviceToHost),
         SvrModulelist[moduleid].name, symbol);
    check_cuda_error(err);
    res.err = err;
    return &res;
}

dscudaResult *
dscudamemcpyfromsymbolasyncd2did_1_svc(int moduleid, RCadr dst, char *symbol, RCsize count, RCsize offset, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;
    CUdeviceptr gsptr;
    size_t gssize;

    SWARN(3, "cudaMemcpyFromSymbolAsync(");
    if (!dscuContext) createDscuContext();

    getGlobalSymbol(moduleid, symbol, &gsptr, &gssize);

    err = cudaMemcpyAsync((void*)dst, (char *)gsptr + offset, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %p, %d, %d, %s, 0x%08llx) done.\n",
         (void *)dst, (void *)gsptr, count, offset, stream,
         dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));

    return &res;
}
dscudaResult *
dscudamemsetid_1_svc(RCadr dst, int value, RCsize count, struct svc_req *sq) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaMemset(");
    if (!dscuContext) createDscuContext();
    err = cudaMemset((void *)dst, value, count);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %d, %d) done.\n", (void *)dst, value, count);
    return &res;
}

dscudaHostAllocResult *
dscudahostallocid_1_svc(RCsize size, unsigned int flags, struct svc_req *sr) {
    static dscudaHostAllocResult res;
    cudaError_t err;
    int *devadr;

    SWARN(3, "cudaHostAlloc(");
    if (!dscuContext) createDscuContext();
    err = cudaHostAlloc((void**)&devadr, size, flags);
    res.pHost = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %d, 0x%08x) done.\n", res.pHost, size, flags);

    return &res;
}
dscudaMallocHostResult *
dscudamallochostid_1_svc(RCsize size, struct svc_req *sr) {
    static dscudaMallocHostResult res;
    cudaError_t err;
    int *devadr;

    SWARN(3, "cudaMallocHost(");
    if (!dscuContext) createDscuContext();
    err = cudaMallocHost((void**)&devadr, size);
    res.ptr = (RCadr)devadr;
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %d) done. devadr:%p\n", &devadr, size, devadr);

    return &res;
}
dscudaResult *
dscudafreehostid_1_svc(RCadr ptr, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaFreeHost(");
    if (!dscuContext) createDscuContext();
    err = cudaFreeHost((void*)ptr);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p) done.\n", ptr);

    return &res;
}

dscudaHostGetDevicePointerResult *
dscudahostgetdevicepointerid_1_svc(RCadr pHost, unsigned int flags , struct svc_req *sr)
{
    cudaError_t err;
    static dscudaHostGetDevicePointerResult res;
    RCadr pDevice;

    SWARN(3, "cudaHostGetDevicePointer(");
    if (!dscuContext) createDscuContext();

    err = cudaHostGetDevicePointer((void **)&pDevice, (void *)pHost, flags);
    check_cuda_error(err);
    res.pDevice = pDevice;
    res.err = err;
    SWARN(3, ") done.\n");
    return &res;
}
dscudaHostGetFlagsResult *
dscudahostgetflagsid_1_svc(RCadr pHost, struct svc_req *sr) {
    cudaError_t err;
    static dscudaHostGetFlagsResult res;
    unsigned int flags;

    SWARN(3, "cudaHostGetFlags(");
    if (!dscuContext) createDscuContext();

    err = cudaHostGetFlags(&flags, (void *)pHost);
    check_cuda_error(err);
    res.err = err;
    res.flags = flags;
    SWARN(3, ") done.\n");
    return &res;
}

dscudaMemcpyAsyncH2HResult *
dscudamemcpyasynch2hid_1_svc(RCadr dst, RCbuf src, RCsize count, RCstream stream, struct svc_req *sr)
{
    static dscudaMemcpyAsyncH2HResult res;
    SWARN(0, "dscudaMemcpyAsync() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpyasynch2did_1_svc(RCadr dst, RCbuf src, RCsize count, RCstream stream, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaMemcpyAsync(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpyAsync((void*)dst, src.RCbuf_val, count, cudaMemcpyHostToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08lx, 0x%08lx, %d, %s, 0x%08lx) done.\n",
         dst, (unsigned long)src.RCbuf_val, count, dscudaMemcpyKindName(cudaMemcpyHostToDevice), stream);
    return &res;
}

dscudaMemcpyAsyncD2HResult *
dscudamemcpyasyncd2hid_1_svc(RCadr src, RCsize count, RCstream stream, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    cudaError_t err;
    static dscudaMemcpyAsyncD2HResult res;

    SWARN(3, "cudaMemcpyAsync(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpyAsync(res.buf.RCbuf_val, (const void*)src, count, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx, %p, %d, %s, %p) done.\n",
         (unsigned long)res.buf.RCbuf_val, (void *)src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToHost), stream);
    return &res;
}

dscudaResult *
dscudamemcpyasyncd2did_1_svc(RCadr dst, RCadr src, RCsize count, RCstream stream, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    SWARN(3, "cudaMemcpyAsync(");
    err = cudaMemcpyAsync((void *)dst, (void *)src, count, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %p, %d, %s, %p) done.\n",
         (void *)dst, (void *)src, count, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice), stream);
    return &res;
}


dscudaMallocPitchResult *
dscudamallocpitchid_1_svc(RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaMallocPitchResult res;
    cudaError_t err;
    int *devadr;
    size_t pitch;

    SWARN(3, "cudaMallocPitch(");
    if (!dscuContext) createDscuContext();
    err = cudaMallocPitch((void**)&devadr, &pitch, width, height);
    res.devPtr = (RCadr)devadr;
    res.pitch = pitch;
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %d, %d) done. devadr:%p\n", &devadr, width, height, devadr);

    return &res;
}

dscudaMemcpy2DToArrayH2HResult *
dscudamemcpy2dtoarrayh2hid_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaMemcpy2DToArrayH2HResult res;
    SWARN(0, "dscudaMemcpy2DToArray() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpy2dtoarrayh2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCbuf srcbuf, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaMemcpy2DToArray(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpy2DToArray((cudaArray*)dst, wOffset, hOffset, srcbuf.RCbuf_val, spitch, width, height, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %d, %d, 0x%08llx, %d, %d, %d, %s) done.\n",
         (void *)dst, wOffset, hOffset, (unsigned long)srcbuf.RCbuf_val, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpy2DToArrayD2HResult *
dscudamemcpy2dtoarrayd2hid_1_svc(RCsize wOffset, RCsize hOffset, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    cudaError_t err;
    static dscudaMemcpy2DToArrayD2HResult res;
    int count = spitch * height;

    SWARN(3, "cudaMemcpy2DToArray(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpy2DToArray((cudaArray *)res.buf.RCbuf_val, wOffset, hOffset, (void *)src, spitch, width, height, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx, %d, %d, %p, %d, %d, %d, %s) done. 2D buf size : %d\n",
         (unsigned long)res.buf.RCbuf_val, wOffset, hOffset, (void *)src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToHost), count);
    return &res;
}

dscudaResult *
dscudamemcpy2dtoarrayd2did_1_svc(RCadr dst, RCsize wOffset, RCsize hOffset, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    cudaError_t err;
    static dscudaResult res;
    SWARN(3, "cudaMemcpy2DToArray(");
    err = cudaMemcpy2DToArray((cudaArray *)dst, wOffset, hOffset, (void *)src, spitch, width, height, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %d, %d, %p, %d, %d, %d, %s) done.\n",
         (void *)dst, wOffset, hOffset,
	 (void *)src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaMemcpy2DH2HResult *
dscudamemcpy2dh2hid_1_svc(RCadr dst, RCsize dpitch, RCbuf src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaMemcpy2DH2HResult res;
    SWARN(0, "dscudaMemcpy2D() does not support cudaMemcpyHostToHost transfer yet.\n");
    return &res;
}

dscudaResult *
dscudamemcpy2dh2did_1_svc(RCadr dst, RCsize dpitch, RCbuf srcbuf, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaMemcpy2D(");
    if (!dscuContext) createDscuContext();
    err = cudaMemcpy2D((void*)dst, dpitch, srcbuf.RCbuf_val, spitch, width, height, cudaMemcpyHostToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08lx, %d, 0x%08lx, %d, %d, %d, %s) done.\n",
         dst, dpitch, (unsigned long)srcbuf.RCbuf_val, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyHostToDevice));
    return &res;
}

dscudaMemcpy2DD2HResult *
dscudamemcpy2dd2hid_1_svc(RCsize dpitch, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr)
{
    static RCsize maxcount = 0;
    cudaError_t err;
    static dscudaMemcpy2DD2HResult res;
    int count = spitch * height;

    SWARN(3, "cudaMemcpy2D(");
    if (!dscuContext) createDscuContext();
    if (maxcount == 0) {
        res.buf.RCbuf_val = NULL;
    }
    if (maxcount < count) {
        res.buf.RCbuf_val = (char*)realloc(res.buf.RCbuf_val, count);
        maxcount = count;
    }
    res.buf.RCbuf_len = count;
    err = cudaMemcpy2D(res.buf.RCbuf_val, dpitch, (void *)src, spitch, width, height, cudaMemcpyDeviceToHost);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "0x%08llx, %d, %p, %d, %d, %d, %s) done. 2D buf size : %d\n",
         (unsigned long)res.buf.RCbuf_val, dpitch,
	 (void *)src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToHost), count);
    return &res;
}

dscudaResult *
dscudamemcpy2dd2did_1_svc(RCadr dst, RCsize dpitch, RCadr src, RCsize spitch, RCsize width, RCsize height, struct svc_req *sr) {
    cudaError_t err;
    static dscudaResult res;
    SWARN(3, "cudaMemcpy2D(");
    err = cudaMemcpy2D((void *)dst, dpitch, (void *)src, spitch, width, height, cudaMemcpyDeviceToDevice);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %d, %p, %d, %d, %d, %s) done.\n",
         (void *)dst, dpitch,
	 (void *)src, spitch, width, height, dscudaMemcpyKindName(cudaMemcpyDeviceToDevice));
    return &res;
}

dscudaResult *
dscudamemset2did_1_svc(RCadr dst, RCsize pitch, int value, RCsize width, RCsize height, struct svc_req *sq) {
    static dscudaResult res;
    cudaError_t err;

    SWARN(3, "cudaMemset2D(");
    if (!dscuContext) createDscuContext();
    err = cudaMemset2D((void *)dst, pitch, value, width, height);
    check_cuda_error(err);
    res.err = err;
    SWARN(3, "%p, %d, %d, %d, %d) done.\n",
	 (void *)dst, pitch, value, width, height);
    return &res;
}


/*
 * Texture Reference Management
 */

dscudaCreateChannelDescResult *
dscudacreatechanneldescid_1_svc(int x, int y, int z, int w, RCchannelformat f, struct svc_req *sr) {
    static dscudaCreateChannelDescResult res;
    cudaChannelFormatDesc desc;

    SWARN(3, "cudaCreateChannelDesc(");
    if (!dscuContext) createDscuContext();
    desc = cudaCreateChannelDesc(x, y, z, w, (enum cudaChannelFormatKind)f);
    res.x = desc.x;
    res.y = desc.y;
    res.z = desc.z;
    res.w = desc.w;
    res.f = desc.f;
    SWARN(3, "%d, %d, %d, %d, %d) done.\n", x, y, z, w, f)
    return &res;
}
dscudaGetChannelDescResult *
dscudagetchanneldescid_1_svc(RCadr array, struct svc_req *sr) {
    static dscudaGetChannelDescResult res;
    cudaError_t err;
    cudaChannelFormatDesc desc;

    SWARN(3, "cudaGetChannelDesc(");
    if (!dscuContext) createDscuContext();
    err = cudaGetChannelDesc(&desc, (const struct cudaArray*)array);
    res.err = err;
    res.x = desc.x;
    res.y = desc.y;
    res.z = desc.z;
    res.w = desc.w;
    res.f = desc.f;
    SWARN(3, "0x%08llx, 0x&08llx) done.\n", &desc, array)
    return &res;
}
dscudaBindTextureResult *
dscudabindtextureid_1_svc(int moduleid, char *texname, RCadr devPtr, RCsize size, RCtexture texbuf, struct svc_req *sr) {
    static dscudaBindTextureResult res;
    cudaError_t err;
    CUtexref texref;
    ServerModule *mp = SvrModulelist + moduleid;

    if (!dscuContext) createDscuContext();

    err = (cudaError_t)cuModuleGetTexRef(&texref, mp->handle, texname);
    SWARN(3, "cuModuleGetTexRef(%p, %p, %s) : module: %s\n",
         &texref, mp->handle, texname, mp->name);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }

    err = setTextureParams(texref, texbuf, texname);
    if (err != cudaSuccess) {
        res.err = err;
        return &res;
    }

    SWARN(4, "cuTexRefSetAddress(%p, %p, %p, %d)\n", &res.offset, texref, devPtr, size);
    err = (cudaError_t)cuTexRefSetAddress((size_t *)&res.offset, texref, (CUdeviceptr)devPtr, size);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }
    res.err = err;

    return &res;
}

dscudaBindTexture2DResult *
dscudabindtexture2did_1_svc(int moduleid, char *texname, RCadr devPtr, RCsize width, RCsize height, RCsize pitch, RCtexture texbuf, struct svc_req *sr) {
    static dscudaBindTexture2DResult res;
    cudaError_t err;
    CUtexref texref;
    ServerModule *mp = SvrModulelist + moduleid;
    CUDA_ARRAY_DESCRIPTOR desc;

    if (!dscuContext) createDscuContext();

    err = (cudaError_t)cuModuleGetTexRef(&texref, mp->handle, texname);
    SWARN(3, "cuModuleGetTexRef(%p, %p, %s) : module: %s\n",
         &texref, mp->handle, texname, mp->name);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }

    err = setTextureParams(texref, texbuf, texname, &desc);
    if (err != cudaSuccess) {
        res.err = err;
        return &res;
    }
    desc.Height = height;
    desc.Width  = width;

    SWARN(4, "cuTexRefSetAddress2D(%p, 0x%08llx, %p, %d)\n", texref, desc, devPtr, pitch);
    err = (cudaError_t)cuTexRefSetAddress2D(texref, &desc, (CUdeviceptr)devPtr, pitch);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }
    res.err = err;

    unsigned int align = CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT;
    unsigned long int roundup_adr = ((devPtr - 1) / align + 1) * align;
    res.offset = roundup_adr - devPtr;
    SWARN(4, "align:0x%x  roundup_adr:%p  devPtr:%p  offset:0x%08llx\n",
         align, roundup_adr, devPtr, res.offset);
    return &res;
}
dscudaResult *
dscudabindtexturetoarrayid_1_svc(int moduleid, char *texname, RCadr array, RCtexture texbuf, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err;
    CUtexref texref;
    ServerModule *mp = SvrModulelist + moduleid;

    if (!dscuContext) createDscuContext();

    err = (cudaError_t)cuModuleGetTexRef(&texref, mp->handle, texname);
    SWARN(3, "cuModuleGetTexRef(%p, %p, %s) : module: %s  moduleid:%d\n",
         &texref, mp->handle, texname, mp->name, moduleid);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }

    err = setTextureParams(texref, texbuf, texname);
    if (err != cudaSuccess) {
        res.err = err;
        return &res;
    }

    SWARN(4, "cuTexRefSetArray(%p, %p, %d)\n", texref, array, CU_TRSA_OVERRIDE_FORMAT);
    err = (cudaError_t)cuTexRefSetArray(texref, (CUarray)array, CU_TRSA_OVERRIDE_FORMAT);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        res.err = err;
        return &res;
    }
    res.err = err;
    return &res;
}

dscudaResult *
dscudaunbindtextureid_1_svc(RCtexture texrefbuf, struct svc_req *sr) {
    static dscudaResult res;
    cudaError_t err = cudaSuccess;

    SWARN(4, "Current implementation of cudaUnbindTexture() does nothing "
         "but returning cudaSuccess.\n");

    res.err = err;
    return &res;
}

dscudaLoadModuleResult *
dscudaloadmoduleid_1_svc(RCipaddr ipaddr, RCpid pid, char *mname, char *image,
			 struct svc_req *sr) {
    static dscudaLoadModuleResult res;
    res.id = dscudaLoadModule(ipaddr, pid, mname, image);
    return &res;
}

/*
 * launch a kernel function of id 'kid' (or name 'kname', if it's not loaded yet),
 * defined in a module of id 'moduleid'.
 */
void *
dscudalaunchkernelid_1_svc(int moduleid, int kid, char *kname,
			   RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream,
			   RCargs args, int flag, struct svc_req *sr) {
    static int dummyres     = 0;
    
    SWARN(5, "<---Entering %s()\n", __func__ );
    dscudaLaunchKernel(moduleid, kid, kname, gdim, bdim, smemsize, stream, args);
    SWARN(5, "--->Exiting  %s\n", __func__);
    return &dummyres; // seems necessary to return something even if it's not used by the client.
}

/*
 * CUFFT library
 */
dscufftPlanResult *
dscufftplan3did_1_svc(int nx, int ny, int nz, unsigned int type, struct svc_req *sr) {
    static dscufftPlanResult res;
    cufftResult err = CUFFT_SUCCESS;
    cufftHandle plan;

    SWARN(3, "cufftplan1d(");
    err = cufftPlan3d(&plan, nx, ny, nz, (cufftType)type);
    SWARN(3, "%d, %d, %d, %d, %d) done.\n", plan, nx, ny, nz, type);

    check_cuda_error((cudaError_t)err);
    res.err = err;
    res.plan = (unsigned int)plan;
    return &res;
}

dscufftResult *
dscufftdestroyid_1_svc(unsigned int plan, struct svc_req *sr) {
    static dscufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    SWARN(3, "cufftDestroy(")
    err = cufftDestroy((cufftHandle)plan);
    SWARN(3, "%d) done.\n", plan);

    res.err = err;
    return &res;
}

dscufftResult *
dscufftexecc2cid_1_svc(unsigned int plan, RCadr idata, RCadr odata, int direction, struct svc_req *sr) {
    static dscufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    SWARN(3, "cufftExecC2C(");
    err = cufftExecC2C((cufftHandle)plan, (cufftComplex *)idata, (cufftComplex *)odata, direction);
    SWARN(3, "%d, %p, %p, %d) done.\n", plan, (void *)idata, (void *)odata, direction);

    res.err = err;
    return &res;
}

#if 0 // CUFFT

/*
 * Interface to CUFFT & CUBLAS written by Yoshikawa for old Remote CUDA.
 * some are already ported to DS-CUDA (see 'dscufftXXXid_1_svc' function defs above),
 * but some are not. Maybe someday, when I have time...
 */

/*
 * cufft library functions
 */
rcufftPlanResult *
rcufftplan1did_1_svc(int nx, unsigned int type, int batch, struct svc_req *sr) {
    static rcufftPlanResult res;
    cufftResult err = CUFFT_SUCCESS;
        cufftHandle plan;

    SWARN(3, "cufftplan1d(");
    cufftPlan1d(&plan, nx, (cufftType)type, batch);
    SWARN(3, "%d, %d, %d, %d) done.\n", plan, nx, type, batch);

    check_cuda_error((cudaError_t)err);
    res.err = err;
        res.plan = (unsigned int)plan;
    return &res;
}
rcufftPlanResult *
rcufftplan2did_1_svc(int nx, int ny, unsigned int type, struct svc_req *sr) {
    static rcufftPlanResult res;
    cufftResult err = CUFFT_SUCCESS;
        cufftHandle plan;

    SWARN(3, "cufftplan1d(");
    cufftPlan2d(&plan, nx, ny, (cufftType)type);
    SWARN(3, "%d, %d, %d, %d) done.\n", plan, nx, ny, type);

    check_cuda_error((cudaError_t)err);
    res.err = err;
        res.plan = (unsigned int)plan;
    return &res;
}
/*
rcufftplanresult *
rcufftplanmanyid_1_svc(int nx, int ny, int nz, unsigned int type, int batch, struct svc_req *sr) {
    static rcufftPlan1dResult res;
    cufftResult err = CUFFT_SUCCESS;
        cufftHandle plan;

    SWARN(3, "cufftplan1d(");
    cufftPlanMany(&plan, nx, ny, nz, (cufftType)type, batch);
    SWARN(3, "%d, %d, %d, %d, %d, %d) done.", plan, nx, ny, nz, type, batch);

    check_cuda_error((cudaError_t)err);
    res.err = err;
        res.plan = (unsigned int)plan;
    return &res;
}
*/
rcufftResult *
rcufftexecr2cid_1_svc(unsigned int plan, RCadr idata, RCadr odata, struct svc_req *sr) {
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    SWARN(3, "cufftExecR2C(");
    cufftExecR2C((cufftHandle)plan, (cufftReal *)idata, (cufftComplex *)odata);
        SWARN(3, "%d, %p, %p) done.\n", plan, idata, odata);

    res.err = err;
    return &res;
}
rcufftResult *
rcufftexecc2rid_1_svc(unsigned int plan, RCadr idata, RCadr odata, struct svc_req *sr) {
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    SWARN(3, "cufftExecC2R(");
    cufftExecC2R((cufftHandle)plan, (cufftComplex *)idata, (cufftReal *)odata);
        SWARN(3, "%d, %p, %p) done.\n", plan, idata, odata);

    res.err = err;
    return &res;
}
rcufftResult *
rcufftexecz2zid_1_svc(unsigned int plan, RCadr idata, RCadr odata, int direction, struct svc_req *sr) {
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    SWARN(3, "cufftExecZ2Z(");
    cufftExecZ2Z((cufftHandle)plan, (cufftDoubleComplex *)idata, (cufftDoubleComplex *)odata, direction);
        SWARN(3, "%d, %p, %p, %d) done.\n", plan, idata, odata, direction);

    res.err = err;
    return &res;
}

rcufftResult *
rcufftexecd2zid_1_svc(unsigned int plan, RCadr idata, RCadr odata, struct svc_req *sr) {
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    SWARN(3, "cufftExecD2Z(");
    cufftExecD2Z((cufftHandle)plan, (cufftDoubleReal *)idata, (cufftDoubleComplex *)odata);
        SWARN(3, "%d, %p, %p) done.\n", plan, idata, odata);

    res.err = err;
    return &res;
}

rcufftResult *
rcufftexecz2did_1_svc(unsigned int plan, RCadr idata, RCadr odata, struct svc_req *sr) {
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    SWARN(3, "cufftExecZ2D(");
    cufftExecZ2D((cufftHandle)plan, (cufftDoubleComplex *)idata, (cufftDoubleReal *)odata);
        SWARN(3, "%d, %p, %p) done.\n", plan, idata, odata);

    res.err = err;
    return &res;
}

/*
rcufftResult *
rcufftSetStreamId()
*/

rcufftResult *
rcufftsetcompatibilitymodeid_1_svc(unsigned int plan, unsigned int mode, struct svc_req *sr) {
    static rcufftResult res;
    cufftResult err = CUFFT_SUCCESS;

    SWARN(3, "cufftSetCompatibilityMode(");
    cufftSetCompatibilityMode((cufftHandle)plan, (cufftCompatibility)mode);
    SWARN(3, "%d, %d) done.\n", plan, mode);

    res.err = err;
    return &res;
}


/*
 * cublas library functions
 */

rcublasCreateResult *
rcublascreate_v2id_1_svc(struct svc_req *sr) {
    static rcublasCreateResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;
    cublasHandle_t handle;

    SWARN(3, "cublasCreate(");
    stat = cublasCreate(&handle);
    SWARN(3, "%p) done.\n", &handle);

    res.err = err;
    res.stat = stat;
    res.handle = (RCadr)handle;

    return &res;
}

rcublasResult *
rcublasdestroy_v2id_1_svc(RCadr handle, struct svc_req *sr) {
    static rcublasResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

    SWARN(3, "cublasDestroy(");
    stat = cublasDestroy((cublasHandle_t)handle);
    SWARN(3, "%d) done.\n", handle);

    res.err = err;
    res.stat = stat;

    return &res;
}

rcublasResult *
rcublassetvectorid_1_svc(int n, int elemSize, RCbuf x, int incx, RCadr y, int incy, struct svc_req *sr) {
    static rcublasResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

    SWARN(3, "cublasSetVector(");
    stat = cublasSetVector(n, elemSize, (const void *)x.RCbuf_val, incx, (void *)y, incy);
    SWARN(3, "%d, %d, %p, %d, %p, %d) done.\n", n, elemSize, x.RCbuf_val, incx, y, incy);

    res.err = err;
    res.stat = stat;

    return &res;
}

rcublasGetVectorResult *
rcublasgetvectorid_1_svc(int n, int elemSize, RCadr x, int incx, int incy, struct svc_req *sr) {
    static rcublasGetVectorResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

    res.y.RCbuf_val = (char *)malloc(n * elemSize);
    res.y.RCbuf_len = n * elemSize;

    SWARN(3, "cublasGetVector(");
    stat = cublasGetVector(n, elemSize, (const void *)x, incx, (void *)res.y.RCbuf_val, incy);
    SWARN(3, "%d, %d, %p, %d, %p, %d) done.\n", n, elemSize, x, incx, res.y.RCbuf_val, incy);

    res.err = err;
    res.stat = stat;

    return &res;
}

rcublasResult *
rcublassgemm_v2id_1_svc(RCadr handle, unsigned int transa, unsigned int transb, int m, int n, int k, float alpha, RCadr A, int lda, RCadr B, int ldb, float beta, RCadr C, int ldc, struct svc_req *sr) {
    static rcublasResult res;
    cudaError_t err = cudaSuccess;
    cublasStatus_t stat = CUBLAS_STATUS_SUCCESS;

    SWARN(3, "cublasSgemm(");
    stat = cublasSgemm((cublasHandle_t)handle, (cublasOperation_t)transa, (cublasOperation_t)transb, m, n, k,
                           (const float *)&alpha, (const float *)A, lda, (const float *)B, ldb, (const float *)&beta, (float *)C, ldc);
    SWARN(3, "%p, %d, %d, %d, %d, %d, %f, %p, %d, %p, %d, %f, %p, %d) done.\n", handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    res.err = err;
    res.stat = stat;

    return &res;
}

#endif // CUFFT
