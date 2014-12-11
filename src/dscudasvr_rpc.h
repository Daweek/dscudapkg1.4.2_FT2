#ifndef DSCUDASVR_RPC_H
#define DSCUDASVR_RPC_H

#include "dscudadefs.h"

       int  rpcUnpackKernelParam(CUfunction *kfuncp, RCargs *argsp);
       void setupRpc(void);
extern void dscuda_prog_1(struct svc_req *rqstp, register SVCXPRT *transp); //dscudarpc_svc.c

extern int           Ndevice;
extern int           Devid[RC_NDEVICEMAX];          // in "dscudasvr.cu"
extern int           Devid2Vdevid[RC_NDEVICEMAX];   // 
extern int           D2Csock;
extern int           TcpPort;
extern int           dscuDevice;                    //
extern ServerState_t DscudaSvr;
extern CUcontext     dscuContext;
extern ServerModule  SvrModulelist[RC_NKMODULEMAX];  // in "dscudasvr.cu"

#endif // DSCUDASVR_RPC_H
