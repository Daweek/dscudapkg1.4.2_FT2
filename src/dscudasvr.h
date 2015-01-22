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
    //<-- Fault Injection
    int    fault_injection; // fault injection pattarn. "0" means no faults.
    int    fault_period; // 0: "never", >0: "fixed", <0: "Distributed"
    double first_fault_time;
    double last_fault_time; // second from "start_time"
    double next_fault_time; // second from "start_time"
    int    fault_count;
    //--> Fault Injection
    int    force_timeout;
    double start_time; //sec
    double stop_time;  //sec
    double exe_time;
    
    void setFaultInjection(int val=1) { fault_injection=val; }
    void unsetFaultInjection(void)    { fault_injection=0; }
    int  getFaultInjection(void)      { return fault_injection; }
    double getPassedTime(void) {
	double curr_time, passed_time;
	dscuda::stopwatch(&curr_time);
	passed_time = curr_time - start_time;
	return passed_time;
    }
    void faultCountUp(void) {
	fault_count++;
    }
    ServerState() {
	dscuda::stopwatch( &start_time );
	fault_injection = 0;
	fault_count = 0;
	fault_period = 60;//sec
	force_timeout = 0;
	last_fault_time = 0.0; //sec
    }
    ~ServerState() {
	char strtime[64];
	dscuda::stopwatch( &stop_time );
	exe_time = stop_time - start_time;
	//SWARN0(1,"start_time= %8.3f sec\n", start_time);
	//SWARN0(1,"stop_time=  %8.3f sec\n", stop_time);
	SWARN0(1,"######## duration-time= %8.3f sec\n", exe_time);
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
