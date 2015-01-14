//                             -*- Mode: C++ -*-
// Filename         : dacudasvr_rpc.h
// Description      : DS-CUDA server node.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-08 13:17:22
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
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
extern ServerState   DscudaSvr;
extern CUcontext     dscuContext;
extern ServerModule  SvrModulelist[RC_NKMODULEMAX];  // in "dscudasvr.cu"

#endif // DSCUDASVR_RPC_H
