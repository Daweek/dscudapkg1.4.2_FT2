//                             -*- Mode: C++ -*-
// Filename         : dacudasvr.h
// Description      : DS-CUDA server node.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-08 13:17:22
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef DSCUDASVR_H
#define DSCUDASVR_H

#include "dscudadefs.h"
#include "dscudautil.h"

struct ServerModule {
    unsigned int id;          /* Static and Serial Identical Number */
    int          valid;       /* 1:valid, 0:invalid */
    unsigned int ipaddr;
    unsigned int pid;
    time_t       loaded_time;
    char         name[256];
    CUmodule     handle;      /* CUDA module, loaded by "cuModuleLoadData()"  */
    CUfunction   kfunc[RC_NKFUNCMAX]; // this is not used for now.
    void         validate(void)   { valid = 1; }
    void         invalidate(void) { valid = 0; }
    int          isValid(void)    { return valid; }
    int          isInvalid(void)  { if (isValid()) { return 0; } else { return 1; } }
};
struct ServerState {
    int  fault_injection; // fault injection pattarn. "0" means no faults.
    int  force_timeout;
    
    void setFaultInjection(int val=1) { fault_injection=val; }
    void unsetFaultInjection(void)    { fault_injection=0; }
    int  getFaultInjection(void)      { return fault_injection; }
    ServerState() {
	fault_injection = 0;
	force_timeout = 0;
    }
};
int dscudaLoadModule(RCipaddr ipaddr, RCpid pid, char *mname, char *image);
void *dscudaLaunchKernel(int moduleid, int kid, const char *kname, RCdim3 gdim, RCdim3 bdim, RCsize smemsize, RCstream stream, RCargs args);
void releaseModules(bool releaseall);
void getGlobalSymbol(int moduleid, char *symbolname, CUdeviceptr *dptr, size_t *size);
CUresult getFunctionByName(CUfunction *kfuncp, const char *kname, int moduleid);
cudaError_t createDscuContext(void);
cudaError_t destroyDscuContext(void);
cudaError_t setTextureParams(CUtexref texref, RCtexture texbuf, char *texname, CUDA_ARRAY_DESCRIPTOR *descp = NULL);
#endif
