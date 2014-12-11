//                             -*- Mode: C++ -*-
// Filename         : dacudasvr.cu
// Description      : DS-CUDA server node.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-08 13:17:22
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <unistd.h>
#include <rpc/rpc.h>
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
#include <cufft.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <poll.h>
#include <errno.h>

#include "dscudarpc.h"
#include "dscudasvr.h"
#include "dscudasvr_rpc.h"
#include "dscuda.h"
#include "sockutil.h"

//<-- oikawa
#define FAULT_INJECTION_LEN (32)
//--> oikawa

//********************************************************************
//***  Class Name: "ServerModule_t"
//***  Description:
//***      - CUDA Kernel function module management for Server.
//********************************************************************
ServerModule SvrModulelist[RC_NKMODULEMAX] = {0};

ServerState_t DscudaSvr;
    
int D2Csock = -1; // socket for sideband communication to the client. inherited from the daemon.
int TcpPort = RC_SERVER_IP_PORT;
static int Connected = 0;
static int UseIbv = 0; // use IB Verbs if set to 1. use RPC by default.
int Ndevice = 1;                 // # of devices in the system.
int Devid[RC_NDEVICEMAX] = {0,}; // real device ids of the ones in the system.
int dscuDevice;                   // virtual device id of the one used in the current context.
CUcontext dscuContext = NULL;
int Devid2Vdevid[RC_NDEVICEMAX]; // device id conversion table from real to virtual.

static void notifyIamReady(void);
static void showUsage(char *command);
static void showConf(void);
static void parseArgv(int argc, char **argv);
static cudaError_t initDscuda(void);

static void initEnv(void);

static void notifyIamReady(void) {
    char msg[] = "ready";
    if (D2Csock >= 0) {
        SWARN(3, "send \"ready\" to the client.\n");
        sendMsgBySocket(D2Csock, msg);
    }
}

static int receiveProtocolPreference(void)
{
    char msg[256], rc[64];

    if (D2Csock >= 0) {
        SWARN(3, "wait for remotecall preference (\"rpc\" or \"ibv\") from the client.\n");
        recvMsgBySocket(D2Csock, msg, sizeof(msg));
        sscanf(msg, "remotecall:%s", rc);
        SWARN(2, "method of remote procedure call: %s\n", rc);
        if (!strncmp("ibv", rc, strlen("ibv"))) {
            return 1;
        } else {
            return 0;
        }
    } else {
        return UseIbv; // do not modify the preference.
    }
}

int main(int argc, char **argv)
{
    parseArgv(argc, argv);
    initEnv();
    initDscuda();
    showConf();

    UseIbv = receiveProtocolPreference();

    if (UseIbv) {
#if !defined(RPC_ONLY)
        setupIbv();
        notifyIamReady();
        ibvMainLoop(NULL);
#endif
    } else {
        setupRpc();
        notifyIamReady();
        svc_run(); // RPC main loop.
    }
    fprintf (stderr, "main loop returned.\n"); // never reached.
    exit (1);
}

static void showUsage(char *command) {
    fprintf(stderr,
            "usage: %s [-s server_id] [-d 'deviceid'] [-p port] [-S socket]\n"
            "       (-p & -S are used by the daemon only.)\n",
            command);
}

static void showConf(void) {
    int i;
    char str[1024], str0[1024];

    SWARN(2, "TCP port : %d (base + %d)\n", TcpPort, TcpPort - RC_SERVER_IP_PORT);
    SWARN(2, "ndevice : %d\n", Ndevice);
    sprintf(str, "real device%s      :", Ndevice > 1 ? "s" : " ");
    for (i = 0; i < Ndevice; i++) {
        sprintf(str0, " %d", Devid[i]);
        strcat(str, str0);
    }
    SWARN(2, "%s\n", str);
    sprintf(str, "virtual device%s   :", Ndevice > 1 ? "s" : " ");
    for (i = 0; i < Ndevice; i++) {
        sprintf(str0, " %d", Devid2Vdevid[Devid[i]]);
        strcat(str, str0);
    }
    SWARN(2, "%s\n", str);
}

extern char *optarg;
extern int optind;
static void
parseArgv(int argc, char **argv) {
    int c, ic;
    char *param = "d:hp:s:S:";
    char *num;
    char buf[256];
    int device_used[RC_NDEVICEMAX] = {0,};
    int tcpport_set = 0;
    int serverid = 0;

    while ((c = getopt(argc, argv, param)) != EOF) {
        switch (c) {
          case 'p':
            TcpPort = atoi(optarg);
            tcpport_set = 1;
            break;

          case 's':
            serverid = atoi(optarg);
            break;

          case 'S':
            D2Csock = atoi(optarg);
            break;

          case 'd':
            Ndevice = 0;
            strncpy(buf, optarg, sizeof(buf));
            num = strtok(buf, " ");
            while (num) {
                ic = atoi(num);
                if (ic < 0 || RC_NDEVICEMAX <= ic ) {
                    fprintf(stderr, "device id out of range: %d\n", ic);
                    exit(2);
                }
                if (!device_used[ic]) { // care for malformed optarg value such as "0 1 2 2 3".
                    device_used[ic] = 1;
                    Devid[Ndevice] = ic;
                    Ndevice++;
                }
                num = strtok(NULL, " ");
            }
            break;

          case 'h':
          default:
            showUsage(argv[0]);
            exit(1);
        }
    }
    if (!tcpport_set) {
        TcpPort = RC_SERVER_IP_PORT + serverid;
        SWARN(3, "TCP port number not given by '-p' option. Use default (%d).\n", TcpPort);
    }
}

// should be called only once in a run.
static cudaError_t initDscuda(void) {
    unsigned int flags = 0; // should always be 0.
    CUresult err;

    SWARN(4, "#(info.)<--- %s()...\n", __func__);

    for (int i = 0; i < Ndevice; i++) {
        Devid2Vdevid[Devid[i]] = i;
    }

    for (unsigned int u=0; u<RC_NKMODULEMAX; u++) { /* Set sequential number. */
	SvrModulelist[u].id = u;
    }

    err = cuInit(flags);
    if (err != CUDA_SUCCESS) {
        SWARN(0, "cuInit(%d) failed.\n", flags);
        exit(1);
    }
    err = (CUresult)cudaSetValidDevices(Devid, Ndevice);
    if (err != CUDA_SUCCESS) {
        SWARN(0, "cudaSetValidDevices(%p, %d) failed.\n", Devid, Ndevice);
        exit(1);
    }
    dscuDevice = Devid[0];
    SWARN(3, "cudaSetValidDevices(%p, %d). dscuDevice:%d\n",
         Devid, Ndevice, dscuDevice);
    SWARN(4, "#(info.)---> %s() done.\n", __func__);
    return (cudaError_t)err;
}

cudaError_t createDscuContext(void)
{
    //    unsigned int flags = 0; // should always be 0.
    CUdevice dev = 0;
    CUresult err;

    err = cuDeviceGet(&dev, dscuDevice);
    if (err != CUDA_SUCCESS) {
        SWARN(0, "cuDeviceGet() failed.\n");
        return (cudaError_t)err;
    }

#if 0
    err = cuCtxCreate(&dscuContext, flags, dev);
    if (err != CUDA_SUCCESS) {
        SWARN(0, "cuCtxCreate() failed.\n");
        return (cudaError_t)err;
    }
#else // not used. set a dummy value not to be called repeatedly.
    dscuContext = (CUcontext)-1;
#endif

    return (cudaError_t)err;
}

cudaError_t destroyDscuContext(void)
{
#if 0

    CUresult cuerr;
    bool all = true;

    SWARN(3, "destroyDscuContext(");
    releaseModules(all);

    cuerr = cuCtxDestroy(dscuContext);
    SWARN(4, "cuCtxDestroy(0x%08llx", dscuContext);
    if (cuerr != CUDA_SUCCESS) {
        SWARN(0, "cuCtxDestroy() failed.\n");
        fatal_error(1);
        return (cudaError_t)cuerr;
    }
    dscuContext = NULL;
    SWARN(4, ") done.\n");
    SWARN(3, ") done.\n");

#else

    dscuContext = NULL;

#endif
    return cudaSuccess;
}

static void initEnv(void) {
    static int firstcall = 1;
    int tmp, tmp2[FAULT_INJECTION_LEN];
    char *env;

    if (!firstcall) return;

    firstcall = 0;

    // DSCUDA_WARNLEVEL
    env = getenv("DSCUDA_WARNLEVEL");
    if (env) {
        tmp = atoi(strtok(env, " "));
        if (0 <= tmp) {
            dscudaSetWarnLevel(tmp);
        }
        SWARN(1, "WarnLevel: %d\n", dscudaWarnLevel());
    }

    // DSCUDA_REMOTECALL
    env = getenv("DSCUDA_REMOTECALL");
#if defined(RPC_ONLY)
    UseIbv = 0;
    SWARN(2, "method of remote procedure call: RPC\n");
#else
    if (D2Csock >= 0) { // launched by daemon.
        SWARN(3, "A server launched by the daemon "
             "does not use the evironment variable 'DSCUDA_REMOTECALL'.\n");
    }
    else { // launched by hand.
        if (!env) {
            fprintf(stderr, "Set an environment variable 'DSCUDA_REMOTECALL' to 'ibv' or 'rpc'.\n");
            exit(1);
        }
        if (!strcmp(env, "ibv")) {
            UseIbv = 1;
            SWARN(2, "method of remote procedure call: InfiniBand Verbs\n");
        }
        else if (!strcmp(env, "rpc")) {
            UseIbv = 0;
            SWARN(2, "method of remote procedure call: RPC\n");
        }
        else {
            UseIbv = 0;
            SWARN(2, "method of remote procedure call '%s' is not available. use RPC.\n", env);
        }
    }
#endif
    // <-- add by Oikawa
    /* Fault Injection Configuration */
    env = getenv("DSCUDA_FAULT_INJECTION"); // integer type.
    if (env) {
	for (int i=0; i<FAULT_INJECTION_LEN; i++) {
	    tmp2[i]=0;
	}
	tmp2[0] = atoi(strtok(env, ""));
	if (tmp2[0] >= 0) {
	    DscudaSvr.setFaultInjection(tmp2[0]);
	}
    }
    SWARN(1, "Fault Injection Config: 0x%x\n", DscudaSvr.getFaultInjection());

    /* Timed out */
    env = getenv("DSCUDA_FORCE_TIMEOUT"); // integer type.
    if (env) {
	for (int i = 0; i < FAULT_INJECTION_LEN; i++) {
	    tmp2[i]=0;
	}
	tmp2[0] = atoi(strtok(env, ""));
	if (tmp2[0] >= 0) {
	    DscudaSvr.force_timeout = tmp2[0];
	}
    }
    SWARN(1, "Force Timeout Config: 0x%x\n", DscudaSvr.force_timeout);

    // --> add by Oikawa
}

/*
 * Unload Modules never been used for a long time.
 */
void releaseModules(bool releaseall = false)
{
    ServerModule *mp;
    int i;
    for (i=0, mp=SvrModulelist; i < RC_NKMODULEMAX; i++, mp++) {
        if (mp->isInvalid()) continue;
        if (releaseall || time(NULL) - mp->loaded_time > RC_SERVER_CACHE_LIFETIME) {
            cuModuleUnload((CUmodule)mp->handle);
	    mp->invalidate();
            mp->handle = NULL;
            for (i = 0; i < RC_NKFUNCMAX; i++) {
                mp->kfunc[i] = NULL;
            }
            SWARN(3, "%s() unloaded a module. name:%s pid:%d ip:%s age:%d\n",
		 __func__, mp->name, mp->pid, dscudaGetIpaddrString(mp->ipaddr),
                 time(NULL) - mp->loaded_time);
        }
    }
}

static void
printSvrModuleList(ServerModule *module_list)
{
    for (int i=0; i<RC_NKMODULEMAX; i++) {
	if (module_list[i].isValid()) {
	    SWARN(10, "#--- SvrModulelist[%d]\n", i);
	    SWARN(10, "#---    + ID=%u, ip=%u\n", module_list[i].id, module_list[i].ipaddr);
	    SWARN(10, "#---    + name=%s\n",   module_list[i].name);
	}
    }
}

CUresult
getFunctionByName(CUfunction *kfuncp, const char *kname, int moduleid) {
    SWARN(10, "   + %s(kname=%s) {\n", __func__, kname);
    CUresult cuerr;
    ServerModule *mp = SvrModulelist + moduleid;

    cuerr = cuModuleGetFunction(kfuncp, mp->handle, kname);
    if (cuerr == CUDA_SUCCESS) {
        SWARN(3, "(^_^) cuModuleGetFunction() : function '%s' found.\n", kname);
	SWARN(3, "(^_^) moduleid=%d, valid=%d, id=%d, name=%s\n",
	     moduleid, mp->valid, mp->id, mp->name);
	printSvrModuleList(SvrModulelist);
    } else {
        SWARN(0, "(;_;) cuModuleGetFunction() : function:'%s'. %s\n",
             kname, cudaGetErrorString((cudaError_t)cuerr));
	SWARN(0, "(;_;) moduleid=%d, valid=%d, id=%d, name=%s\n",
	     moduleid, mp->valid, mp->id, mp->name);
	switch (cuerr) {
	  case CUDA_ERROR_DEINITIALIZED:
	    SWARN(0, "CUDA_ERROR_DEINITIALIZED.\n");   break;
	  case CUDA_ERROR_NOT_INITIALIZED:
	    SWARN(0, "CUDA_ERROR_NOT_INITIALIZED.\n"); break;
	  case CUDA_ERROR_INVALID_CONTEXT:
	    SWARN(0, "CUDA_ERROR_INVALID_CONTEXT.\n"); break;
	  case CUDA_ERROR_INVALID_VALUE:
	    SWARN(0, "CUDA_ERROR_INVALID_VALUE.\n");   break;
	  case CUDA_ERROR_NOT_FOUND:
	    SWARN(0, "CUDA_ERROR_NOT_FOUND.\n");       break;
	  case CUDA_ERROR_INVALID_HANDLE:
	    SWARN(0, "CUDA_ERROR_INVALID_HANDLE.\n");  break;
	  default:
	    SWARN(0, "(unknown error code: %d)\n", cuerr);
	}
        fatal_error(1);
    }
    SWARN(10, "   + }\n");
    return cuerr;
}

void getGlobalSymbol(int moduleid, char *symbolname, CUdeviceptr *dptr, size_t *size)
{
    CUresult cuerr;
    ServerModule *mp;

    if (moduleid < 0 || RC_NKMODULEMAX <= moduleid) {
        SWARN(0, "getGlobalSymbol() : invalid module id:%d.\n", moduleid);
        fatal_error(1);
    }
    mp = SvrModulelist + moduleid;
    cuerr = cuModuleGetGlobal(dptr, size, mp->handle, symbolname);
    if (cuerr == CUDA_SUCCESS) {
	SWARN(3, "cuModuleGetGlobal(%p, %p, %p, %s) done."
	 " modulename:%s  symbolname:%s  *dptr:0x%08lx\n",
	 dptr, size, mp->handle, symbolname,
	 mp->name, symbolname, *dptr);
    }
    else {
        SWARN(0, "cuModuleGetGlobal(%p, %p, %p, 0x%08llx) failed."
             " modulename:%s  symbolname:%s  %s\n",
             dptr, size, mp->handle, symbolname,
             mp->name, symbolname, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
}

int dscudaLoadModule(RCipaddr ipaddr, RCpid pid, char *mname, char *image)
{
    CUresult cuerr;
    ServerModule   *mp;
    int      i;
    SWARN(10, "<---Entering %s()\n", __func__);

#if RC_CACHE_MODULE
    // look for mname in the module list, which may found if the client
    // resent multiple requests for the same mname:pid:ipaddr.
    int found = 0;
    mp = SvrModulelist;
    for (i=0; i < RC_NKMODULEMAX; i++) {
        if (mp->isInvalid()) continue;
        if ((unsigned int)ipaddr == mp->ipaddr && pid == mp->pid &&
            !strcmp(mname, mp->name)) {
            found = 1;
            break;
        }
	SWARN(4, "ip:%x  %x    pid:%d  %d    name:%s  %s\n",
	     (unsigned int)ipaddr, mp->ipaddr, pid, mp->pid, mname, mp->name);
	mp++;
    }

    if (found) { // module found. i.e, it's already loaded.
        SWARN(3, "\n\n------------------------------------------------------------------\n"
             "dscudaloadmoduleid_1_svc() got multiple requests for\n"
             "  the same module name : %s,\n"
             "  the same process id  : %d, and\n"
             "  the same IP address  : %s,\n"
             "which means a client resent the same module twice or more.\n"
             "If you see this message too often, you may want to increase\n"
             "$dscuda/include/dscudadefs.h:RC_CLIENT_CACHE_LIFETIME\n"
             "for better performance.\n"
             "------------------------------------------------------------------\n\n",
             mname, pid, dscudaGetIpaddrString(ipaddr));
        SWARN(3, "cuModuleLoadData() : a module found in the cache. id:%d  name:%s  age:%d\n",
             mp->id, mname, time(NULL) - mp->loaded_time);
    }
    else  // module not found in the cache. load it from image.
#endif // RC_CACHE_MODULE

    {
        for (i=0, mp=SvrModulelist; i < RC_NKMODULEMAX; i++, mp++) { /* look for .valid==0 */
            if (mp->isInvalid()) break;
            if (i >= RC_NKMODULEMAX) { SWARN(0, "(+_+) module cache is full.\n"); fatal_error(1); }
        }
	/* Register new SvrModulelist[i] */
        /* mp->id = i; */
        cuerr = cuModuleLoadData(&mp->handle, image); /* load .ptx string */
        if (cuerr != CUDA_SUCCESS) {
            SWARN(0, "cuModuleLoadData() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        mp->validate(); /* mp->valid  = 1;*/
        mp->ipaddr = ipaddr;
        mp->pid    = pid;
        strncpy(mp->name, mname, sizeof(SvrModulelist[0].name));
        for (i=0; i<RC_NKFUNCMAX; i++) { mp->kfunc[i] = NULL; }
        SWARN(3, "cuModuleLoadData() : a module loaded. id:%d  name:%s\n", mp->id, mname);
    }
    mp->loaded_time = time(NULL); // (re)set the lifetime of the cache.
    releaseModules();

    SWARN(10, "--->Exiting  %s()\n", __func__);
    return mp->id;
}

void *dscudaLaunchKernel(int moduleid, int kid, const char *kname /*kernel func name*/,
                   RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, RCargs args)
{
    static int called_count = 0;
    static int dummyres     = 123;
    SWARN(10, "%s(int moduleid=%d, int kid=%d, char *kname=%s), %d called.\n",
	 __func__, moduleid, kid, kname, called_count);

    if ( DscudaSvr.force_timeout > 0 ) {
	if ( called_count >= 3 ) {
	    SWARN(2, "sleeping 60 sec...\n");
	    sleep(60);
	}
    }
    
    int paramsize;
    CUresult cuerr;

#if !RC_SUPPORT_CONCURRENT_EXEC
    stream = 0;
#endif

    if (!dscuContext) createDscuContext();

    // load a kernel function into Module[moduleid].kfunc[kid]
    // form Module[moduleid].handle.
    if (moduleid < 0 || RC_NKMODULEMAX <= moduleid) {
        SWARN(0, "dscudalaunchkernelid_1_svc() : invalid module id:%d.\n", moduleid);
        fatal_error(1);
    }

#if 1 // look inside a module for a function by name.
    CUfunction kfunc;
    getFunctionByName(&kfunc, kname, moduleid);
#else // look for a function by its ID.
    // this is faster, but not used since it would cause a problem
    // when called from a kernel function that uses C++ template.
    // in that case kid might not be unique for each instance of the template.
    ServerModule *mp = SvrModulelist + moduleid;
    CUfunction kfunc = mp->kfunc[kid];
    if (!kfunc) {
        getFunctionByName(&kfunc, kname, moduleid);
        mp->kfunc[kid] = kfunc;
    }
#endif

    // a kernel function found.
    // now make it run.
    if (UseIbv) {
#if !defined(RPC_ONLY)
	SWARN(10, "ibvUnpackKernelParam()\n");
        paramsize = ibvUnpackKernelParam(&kfunc, args.RCargs_len, (IbvArg *)args.RCargs_val);
#endif
    } else {
	SWARN(10, "rpcUnpackKernelParam()\n");
        paramsize = rpcUnpackKernelParam(&kfunc, &args);
    }
    cuerr = cuParamSetSize(kfunc, paramsize);
    if (cuerr != CUDA_SUCCESS) {
        SWARN(0, "cuParamSetSize() failed. size:%d %s\n",
             paramsize, cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
    SWARN(5, "cuParamSetSize() done.\n");

    cuerr = cuFuncSetBlockShape(kfunc, bdim.x, bdim.y, bdim.z);
    if (cuerr != CUDA_SUCCESS) {
        SWARN(0, "cuFuncSetBlockShape() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
        fatal_error(1);
    }
    SWARN(5, "cuFuncSetBlockShape() done.\n");

    if (smemsize != 0) {
        cuerr = cuFuncSetSharedSize(kfunc, smemsize);
        if (cuerr != CUDA_SUCCESS) {
            SWARN(0, "cuFuncSetSharedSize() failed. %s\n", cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        SWARN(5, "cuFuncSetSharedSize() done.\n");
    }

    if ((cudaStream_t)stream == NULL) {
        cuerr = cuLaunchGrid(kfunc, gdim.x, gdim.y);  /* Launch Kernel Function */
        if (cuerr != CUDA_SUCCESS) {
            SWARN(0, "cuLaunchGrid() failed. kname:%s %s\n",
                 kname, cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        SWARN(3, "cuLaunchGrid() done. kname:%s\n", kname);
    } else {
        cuerr = cuLaunchGridAsync(kfunc, gdim.x, gdim.y, (cudaStream_t)stream);
        if (cuerr != CUDA_SUCCESS) {
            SWARN(0, "cuLaunchGridAsync() failed. kname:%s  %s\n",
                 kname, cudaGetErrorString((cudaError_t)cuerr));
            fatal_error(1);
        }
        SWARN(3, "cuLaunchGridAsync() done.  kname:%s  stream:0x%08llx\n", kname, stream);
    }
    SWARN(10, "+--- Done. %s() %d called.\n", __func__, called_count );
    called_count++;
    return &dummyres; // seems necessary to return something even if it's not used by the client.
}

cudaError_t
setTextureParams(CUtexref texref, RCtexture texbuf, char *texname, CUDA_ARRAY_DESCRIPTOR *descp)
{
    cudaError_t err;
    int ncomponent, i;
    unsigned int texref_flags = 0;
    unsigned int fmt_high, fmt_low, fmt_index;

    CUarray_format fmt[] = {
        CU_AD_FORMAT_UNSIGNED_INT8,    // 00_00
        CU_AD_FORMAT_UNSIGNED_INT16,   // 00_01
        CU_AD_FORMAT_UNSIGNED_INT32,   // 00_10
        (CUarray_format)0,
        CU_AD_FORMAT_SIGNED_INT8,      // 01_00
        CU_AD_FORMAT_SIGNED_INT16,     // 01_01
        CU_AD_FORMAT_SIGNED_INT32,     // 01_10
        (CUarray_format)0,
        (CUarray_format)0,
        CU_AD_FORMAT_HALF,             // 10_01
        (CUarray_format)0,
        (CUarray_format)0,
        (CUarray_format)0,
        (CUarray_format)0,
        CU_AD_FORMAT_FLOAT,            // 11_10
        (CUarray_format)0,
    };

    // set addressmode (wrap/clamp/mirror/border)
    //
    for (i = 0; i < 3; i++) {
        err = (cudaError_t)cuTexRefSetAddressMode(texref, i, (CUaddress_mode_enum)texbuf.addressMode[i]);
        if (err != cudaSuccess) {
            check_cuda_error(err);
            return err;
        }
    }

    // set filtermode (point/linear)
    //
    err = (cudaError_t)cuTexRefSetFilterMode(texref, (CUfilter_mode_enum)texbuf.filterMode);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        return err;
    }

    // set flags (integer/normalized)
    //
    if (texbuf.normalized) {
        texref_flags |= CU_TRSF_NORMALIZED_COORDINATES;
    }
    else {
        texref_flags |= CU_TRSF_READ_AS_INTEGER;
    }
    err = (cudaError_t)cuTexRefSetFlags(texref, texref_flags);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        return err;
    }

    // set format (unsigned/signed/float, 32/16/8-bit)
    //
    switch (texbuf.x) {
      case 8:
        fmt_low = 0;
        break;
      case 16:
        fmt_low = 1;
        break;
      case 32:
        fmt_low = 2;
        break;
      default:
        SWARN(0, "cuModuleGetTexRef() invalid channel format. texture name:%s descriptor.x:%d\n",
             texname, texbuf.x);
        err = cudaErrorInvalidValue;
        return err;
    }
    switch (texbuf.f) {
      case cudaChannelFormatKindUnsigned:
        fmt_high = 0;
        break;

      case cudaChannelFormatKindSigned:
        fmt_high = 1;
        break;

      case cudaChannelFormatKindFloat:
        fmt_high = 3;
        break;

      case cudaChannelFormatKindNone:
        SWARN(0, "cuModuleGetTexRef() invalid channel format. texture name:%s descriptor.f:%s\n",
             texname, "cudaChannelFormatKindNone");
        err = cudaErrorInvalidValue;
        return err;

      default:
        SWARN(0, "cuModuleGetTexRef() invalid channel format. texture name:%s descriptor.f:%s\n",
             texname, texbuf.f);
        err = cudaErrorInvalidValue;
        return err;
    }
    fmt_index = fmt_high << 2 | fmt_low;
    ncomponent = 1;
    if (texbuf.y) ncomponent = 2;
    if (texbuf.z) ncomponent = 3;
    if (texbuf.w) ncomponent = 4;
    if (descp) {
        descp->Format = fmt[fmt_index];
        descp->NumChannels = ncomponent;
    }
    SWARN(4, "cuTexRefSetFormat(%p, %d, %d)\n", texref, fmt[fmt_index], ncomponent);
    err = (cudaError_t)cuTexRefSetFormat(texref, fmt[fmt_index], ncomponent);
    if (err != cudaSuccess) {
        check_cuda_error(err);
        return err;
    }

    return cudaSuccess;
}
