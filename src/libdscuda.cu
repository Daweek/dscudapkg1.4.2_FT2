//                             -*- Mode: C++ -*-
// Filename         : libdscuda.cu
// Description      : DS-CUDA client node library for Remote Procedure Call.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-15 11:45:48
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <ctype.h>
#include <pwd.h>
#include <netdb.h>
#include <pthread.h>
#include "dscuda.h"
#include "dscudautil.h"
#include "libdscuda.h"

static void  extractENV(bool &bool_var, const char *envname);
static void  extractENV(int  &int_var,  const char *envname, int undef=0);
static void  extractENV(char *str_var,  const char *envname, int len);
static void  getenvDSCUDA_WARNLEVEL(void);
static void  updateSpareServerList(void);
static char* readServerConf(char *fname);

static int   VdevidIndexMax = 0; //# of pthreads which utilize virtual devices.
const  char *DEFAULT_SVRIP = "localhost";

static pthread_mutex_t VdevidMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t       VdevidIndex2ptid[RC_NPTHREADMAX]; // convert an Vdevid index into pthread id.
// CheckPointing mutual exclusion
int cp_thread_exit=0;
       pthread_mutex_t cudaMemcpyD2H_mutex = PTHREAD_MUTEX_INITIALIZER;
       pthread_mutex_t cudaMemcpyH2D_mutex = PTHREAD_MUTEX_INITIALIZER;
       pthread_mutex_t cudaKernelRun_mutex = PTHREAD_MUTEX_INITIALIZER;
       pthread_mutex_t cudaElse_mutex      = PTHREAD_MUTEX_INITIALIZER;

       pthread_mutex_t Tc_reset_mutex      = PTHREAD_MUTEX_INITIALIZER;
       int             Tc_reset_req;

       RCmappedMem    *RCmappedMemListTop     = NULL;
       RCmappedMem    *RCmappedMemListTail    = NULL;

//#if RC_SUPPORT_STREAM
static RCstreamArray  *RCstreamArrayListTop   = NULL;
static RCstreamArray  *RCstreamArrayListTail  = NULL;
//#endif

static RCeventArray   *RCeventArrayListTop    = NULL;
static RCeventArray   *RCeventArrayListTail   = NULL;

static RCcuarrayArray *RCcuarrayArrayListTop  = NULL;
static RCcuarrayArray *RCcuarrayArrayListTail = NULL;

int    Vdevid[RC_NPTHREADMAX] = {0};   // the virtual device currently in use.

/*
 * Physical GPU device server
 */
ServerArray SvrCand;
ServerArray SvrSpare;   // Alternative GPU Device Servers.
ServerArray SvrIgnore;  // Forbidden GPU Device Servers.

void (*errorHandler)(void *arg) = NULL;
void *errorHandlerArg = NULL;

//struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX];

ClientState   St;
PtxStore      Ptx;
/*
 * Client initializer.
 * This function may be executed in parallel threads, so need mutex lock.
 */
ClientState::ClientState(void) {
    //<-- Open dscuda output file.
    char curr_time[80];
    dscuda::sprintfDate( curr_time );
    sprintf( this->dslog_filename, "c%s.dslog", curr_time );
    sprintf( this->dserr_filename, "c%s.dserr", curr_time );
    sprintf( this->dschp_filename, "c%s.dschp", curr_time );
    
    dscuda_stdout = fopen( dslog_filename, "w" );
    if (dscuda_stdout == NULL) {
	fprintf(stderr, "dscuda: failed to open file %s.", dslog_filename);
	exit(EXIT_FAILURE);
    } else {
	fprintf(stderr, "dscuda: log file ==> %s\n", dslog_filename);
    }
    //
    dscuda_stderr = fopen( dserr_filename, "w" );
    if (dscuda_stderr == NULL) {
	fprintf(stderr, "dscuda: failed to open file %s.", dserr_filename);
	exit(EXIT_FAILURE);
    } else {
	fprintf(stderr, "dscuda: err file ==> %s\n", dserr_filename);
    }
    //
    dscuda_chkpnt = fopen( dschp_filename, "w" );
    if (dscuda_chkpnt == NULL) {
	fprintf(stderr, "dscuda: failed to open file %s.", dschp_filename);
	exit(EXIT_FAILURE);
    } else {
	fprintf(stderr, "dscuda: err file ==> %s\n", dschp_filename);
    }
    //--> Open dscuda output file.
	
    INFO0("\
###******************************************************************************\n\
###***                                                                          *\n\
###***   Start process of DS-CUDA client library.                               *\n\
###***                                                                          *\n\
###******************************************************************************\n");
    INFO0("[ DS-CUDA Version      ] %s\n", RC_DSCUDA_VER);
    {
	/* Capture start time and print to logfile. */
	char s_time[80];
	struct tm *timebuf;
	start_time = time(NULL);
	timebuf = localtime(&start_time);
	strftime(s_time, 80, "%T (%F)", timebuf);
	INFO0("[ Start time           ] %s\n", s_time);
    }
    {
	/* Print IP address of DS-CUDA client host. */
	sockaddr_in addrin;
	get_myaddress(&addrin);
	setMyIPAddr(addrin.sin_addr.s_addr);
	INFO0("[ IP address of client ] %s\n",dscudaGetIpaddrString(St.getIpAddress()));
    }
    {
	char path[1024];
	getcwd(path, 1024);
	INFO0("[ Working Directory    ] %s\n", path);
	INFO0("[ Process ID (PID)     ] %d\n", getpid());
    }


    use_ibv     = 0;
    autoverb    = 0;
    daemon      = 0;
    this->unsetRollbackCalling();

    extractENV( dscuda_path, "DSCUDA_PATH", 512 );
    INFO0("[ Environment varialbe ] DSCUDA_PATH      = %s\n", dscuda_path);

    getenvDSCUDA_WARNLEVEL();       /* set from DSCUDA_WARNLEVEL */
    INFO0("[ Environment variable ] DSCUDA_WARNLEVEL = %d\n", dscuda::getWarnLevel());

    this->configFT();
    INFO0("[ Environment variable ] DSCUDA_USEDAEMON = %d\n", daemon);
    INFO0("[ Environment variable ] DSCUDA_AUTOVERB  = %d\n", autoverb);
    INFO0("[ Environment variable ] DSCUDA_CP_PERIOD = %d\n", cp_period);
    INFO0("[ Fault Tolerant Mode  ] ");
    switch (ft_mode) {
    case FT_NONE:
	INFO0("\"FT_NONE\"\n");
	break;
    case FT_ERRSTAT:
	INFO0("\"FT_ERRSTAT\"\n");
	break;
    case FT_BYCPY:
	INFO0("\"FT_BYCPY\"\n");
	break;
    case FT_BYTIMER:
	INFO0("\"FT_BYTIMER\"\n");
	break;
    case FT_OPTION:
	INFO0("\"FT_OPTION\"\n");
	break;
    default:
	WARN0(0, "(UNKNOWN).\n");
	exit(EXIT_FAILURE);
    }
    INFO0("[Environment var] DSCUDA_FT0  = %d (d2h_simple)\n",   ft.d2h_simple   );
    INFO0("[Environment var] DSCUDA_FT1  = %d (d2h_reduncpy)\n", ft.d2h_reduncpy );
    INFO0("[Environment var] DSCUDA_FT2  = %d (d2h_compare)\n",  ft.d2h_compare  );
    INFO0("[Environment var] DSCUDA_FT3  = %d (d2h_statics)\n",  ft.d2h_statics  );
    INFO0("[Environment var] DSCUDA_FT4  = %d (d2h_rollback)\n", ft.d2h_rollback );
    
    INFO0("[Environment var] DSCUDA_FT8  = %d (cp_periodic)\n",  ft.cp_periodic );
    INFO0("[Environment var] DSCUDA_FT9  = %d (cp_reduncpy)\n",  ft.cp_reduncpy );
    INFO0("[Environment var] DSCUDA_FT10 = %d (cp_compare)\n",   ft.cp_compare  );
    INFO0("[Environment var] DSCUDA_FT11 = %d (cp_statics)\n",   ft.cp_statics  );
    INFO0("[Environment var] DSCUDA_FT12 = %d (cp_rollback)\n",  ft.cp_rollback );
    
    INFO0("[Environment var] DSCUDA_FT16 = %d (rec_en)\n", ft.rec_en );
    INFO0("[Environment var] DSCUDA_FT24 = %d (migrate)\n", ft.gpu_migrate );

    this->initVirtualDevice();  /* Update the list of virtual devices */
    WARN0(0, "\n");

    // Search around cluster.
    WARN(2, "   <---Start Searching DS-CUDA daemon program                            *\n");
    dscuda::searchDaemon();
    WARN(2, "   --->Stop  Searching DS-CUDA daemon program                            *\n\n");

    ServerArray svr_array;
    svr_array.captureEnv("DSCUDA_SERVER_IGNORE", hl_BAD);
    svr_array.print();

    updateSpareServerList();
    svr_array.captureEnv("DSCUDA_SERVER_SPARE", hl_GOOD);
    svr_array.print();
    for (int i=0; i<svr_array.num; i++) {
	SvrSpare.append( &svr_array.svr[i] );
    }
    
    printVirtualDeviceList(); /* Print result to terminal. */

    WARN(2, "method of remote procedure call: ");
    switch ( dscudaRemoteCallType() ) {
    case RC_REMOTECALL_TYPE_RPC:
	WARN0(2, "RPC\n");
	break;
    case RC_REMOTECALL_TYPE_IBV:
	WARN0(2, "InfiniBand Verbs\n");
	break;
    default:
	WARN0(0, "(Unkown)\n"); exit(1);
    }

    /*
     * Establish connections of all physical devices.
     */
    for (int i=0; i<Nvdev; i++) {
	for (int j=0; j<Vdev[i].nredundancy; j++) {
	    Vdev[i].server[j].setupConnection();
	    Vdev[i].server[j].setID( j );
	    WARN(1, "setupConn. Vdev[%d].server[%d].Clnt=%p\n",
		 i, j, Vdev[i].server[j].Clnt);
        }
    }
    
    if (ft.d2h_statics) {
	if (ft.cp_statics) { WARN(1, "[ERRORSTATICS] count both @D2H and @CP.\n" ); }
	else               { WARN(1, "[ERRORSTATICS] count @D2H but @CP.\n" );      }
    } else {
	if (ft.cp_statics) { WARN(1, "[ERRORSTATICS] count @CP  but @D2H.\n" );     }
	else               { WARN(1, "[ERRORSTATICS] not counted.\n" );             }
    }

    if (ft.cp_periodic) {
	WARN(1, "Starts Automatic CheckPointing Threads.\n" );	
	pthread_create(&tid, NULL, periodicCheckpoint, (void *)&cp_period);
    }
    
    INFO0("\
###******************************************************************************\n\
###***   Start user application process.                                        *\n\
###******************************************************************************\n");
} //--> ClientState::ClientState(void)
//--
//--
//--
ClientState::~ClientState(void) {
    INFO0("\
###******************************************************************************\n\
###***   Completed user application process.                                    *\n\
###******************************************************************************\n");

    PhyDev  *svr;
    time_t     exe_time;
    char       my_tfmt[64];	      
    struct tm *my_local;

    //--- Terminate the checkpointing thread.
    cp_thread_exit = 1;
    if (ft.cp_periodic) {
	WARN(1, "Stops Automatic CheckPointing Threads.\n" );
	//pthread_cancel(tid);
	pthread_join(tid, NULL);
    }
	
    stop_time = time( NULL);
    exe_time = stop_time - start_time;

    //--- Report start time.
    my_local = localtime( &start_time);
    strftime( my_tfmt, 64, "%c", my_local);
    WARN0(1, "    Start_time: %s\n", my_tfmt);
    //--- Report stop time.
    my_local = localtime( &stop_time);
    strftime( my_tfmt, 64, "%c", my_local);
    WARN0(1, "    Stop_time:  %s\n", my_tfmt);
    //--- Report run time.
    my_local = localtime( &exe_time);
    strftime( my_tfmt, 64, "%s", my_local);
    WARN0(1, "    Run_time:   %s (sec)\n", my_tfmt);

    if (ft.d2h_statics) {
	if (ft.cp_statics) {
	    WARN_CP(1, "[ERRORSTATICS] count both @D2H and @CP.\n" );
	    for (int i=0; i<Nvdev; i++) {
		WARN_CP(1, "    Virtual[%d]'s total error = %d [times]\n", i, Vdev[i].ft_unmatch_total);
	    }
	}
	else {
	    WARN0(1, "[ERRORSTATICS] count @D2H not @CP.\n" );
	    for (int i=0; i<Nvdev; i++) {
		WARN0(1, "    [ERRORSTAT]  Virtual[%2d]\n", i);
		for (int j=0; j<Vdev[i].nredundancy; j++) {
		    svr = &Vdev[i].server[j];
		    WARN0(1, "    [ERRORSTAT]  + Physical[%2d]:%s:%s: ErrorCount= %d , MatchCount= %d\n",
			  j, svr->ip, svr->hostname, svr->stat_error, svr->stat_correct);
		}
	    }
	}
    }
    else {
	if (ft.cp_statics) { WARN(1, "[ERRORSTATICS] TODO.\n" );        }
	else               { WARN(1, "[ERRORSTATICS] not counted.\n" ); }
    }

    INFO0("\
###******************************************************************************\n\
###***                                                                          *\n\
###***   Completed process of DS-CUDA client library.                           *\n\
###***                                                                          *\n\
###******************************************************************************\n");
} //--> ClientState::~ClientState(void)

void
ClientState::configFT(void) {
    extractENV( this->daemon,    "DSCUDA_USEDAEMON",  0 );
    extractENV( this->cp_period, "DSCUDA_CP_PERIOD", 60 );
    extractENV( this->autoverb,  "DSCUDA_AUTOVERB",   0 );
    //<--- Define Fault Tolerant behavior from env.var.
    switch (autoverb) {
    case -1:
	ft_mode = FT_OPTION;
	break;
    case 0:
	ft_mode = FT_NONE;
	break;
    case 1:
	ft_mode = FT_ERRSTAT;
	break;
    case 2:
	ft_mode = FT_BYCPY;
	break;
    case 3:
	ft_mode = FT_BYTIMER;
	break;
    default:
	WARN(0, "Found invalid setting of DSCUDA_AUTOVERB=%d\n", autoverb);
	exit(EXIT_FAILURE);
    }
    //---> Define Fault Tolerant behavior from env.var.
    extractENV( ft.d2h_simple,    "DSCUDA_FT0" );
    extractENV( ft.d2h_reduncpy,  "DSCUDA_FT1" );
    extractENV( ft.d2h_compare,   "DSCUDA_FT2" );
    extractENV( ft.d2h_statics,   "DSCUDA_FT3" );
    extractENV( ft.d2h_rollback,  "DSCUDA_FT4" );
    //
    extractENV( ft.cp_periodic,   "DSCUDA_FT8" ); // 2nd: checkpointing
    extractENV( ft.cp_reduncpy,   "DSCUDA_FT9" );
    extractENV( ft.cp_compare,    "DSCUDA_FT10" );
    extractENV( ft.cp_statics,    "DSCUDA_FT11" );
    extractENV( ft.cp_rollback,   "DSCUDA_FT12" );
    //
    extractENV( ft.rec_en,        "DSCUDA_FT16" ); // 3rd: CUDA API recording
    //
    extractENV( ft.gpu_migrate,   "DSCUDA_FT24" ); // 4th: GPU Migration

    //<--- copy same value to virtual and physical device.
    for (int i=0; i<RC_NVDEVMAX; i++) {
	this->Vdev[i].ft_mode = this->ft_mode;
	this->Vdev[i].ft      = this->ft;
	for (int k=0; k<Vdev[i].nredundancy; k++) {
	    Vdev[i].server[k].ft_mode = this->ft_mode;
	    Vdev[i].server[k].ft      = this->ft;
	}
    }
    //---> copy same value to virtual and physical device.

    if (ft.rec_en) {
	for (int i=0; i<RC_NVDEVMAX; i++) {
	    Vdev[i].recordON();
	}
    }
}//--> void ClientState::configsFT(void)
/*
 *
 */
void
ClientState::initVirtualDevice(void) {
    char   *ip;
    char    ips[RC_NVDEVMAX][256];
    char    buf[1024*RC_NVDEVMAX];
    PhyDev *sp;
    char   *vdev_token;
    char   *pdev_token;
    {// DSCUDA_SERVER
	char *sconfname;
	char *env;    
	if (sconfname = getenv("DSCUDA_SERVER_CONF")) {
	    env = readServerConf(sconfname);
	    INFO0("[ Environment variable ] DSCUDA_SERVER_CONF = %s\n", env);
	} else {
	    env = getenv("DSCUDA_SERVER");
	    INFO0("[ Environment variable ] DSCUDA_SERVER    = %s\n", env);
	}
	// check DSCUDA_SERVER (1)
	if (env == NULL) {
	    Nvdev = 1;
	    Vdev[0].nredundancy = 1;
	    sp = Vdev[0].server;
	    sp->id = 0;
	    strncpy(sp->ip, DEFAULT_SVRIP, sizeof(sp->ip));
	    return;
	}
	
	// check DSCUDA_SERVER (2)
	if (sizeof(buf) < strlen(env)) {
	    WARN(0, "Too long length of DSCUDA_SERVER.\n");
	    exit(EXIT_FAILURE);
	}
	strncpy( buf, env, sizeof(buf) );
    }
    //<-- set "Nvdev", # of virtual device count.
    Nvdev = 0;
    vdev_token = strtok(buf, DELIM_VDEV); // a list of IPs which consist a single vdev.
    while (vdev_token != NULL) {
	strcpy(ips[Nvdev], vdev_token);
	Nvdev++;
	if (RC_NVDEVMAX < Nvdev) {
	    WARN(0, "number of devices exceeds the limit, RC_NVDEVMAX (=%d).\n",
		 RC_NVDEVMAX);
	    exit(EXIT_FAILURE);
	}
	vdev_token = strtok(NULL, DELIM_VDEV);
    }
    //--> set "Nvdev", # of virtual device count.
    
    for (int i=0; i<Nvdev; i++) {
	int nred=0;
	int uniq=0; // begin with 0.
	pdev_token = strtok(ips[i], DELIM_REDUN); // an IP (optionally with devid preceded by a comma) of
	// a single element of the vdev.
	while (pdev_token != NULL) {
	    strcpy(Vdev[i].server[nred].ip, pdev_token);
	    pdev_token = strtok(NULL, DELIM_REDUN);
	    nred++;
	}
	/*
	 * update Vdev.info.
	 */
	Vdev[i].setConfInfo(nred);
	
	for (int j=0; j<nred; j++) {
	    sp = &Vdev[i].server[j];
	    strncpy(buf, sp->ip, sizeof(buf));
	    ip = strtok(buf, ":");
	    sp->setIP(ip);
	    ip = strtok(NULL, ":");
	    sp->setCID(ip);
	    sp->setUNIQ(uniq);
	    uniq++;
	}
    } //for ( int i=0; ...
    /* convert hostname to ip address. */
    char *hostname;
    int  det_abc;
    char letter;
    char *ip_ref;
    struct hostent *hostent0;
    for (int i=0; i<Nvdev; i++) {
	Vdev[i].id = i;
	for (int j=0; j < Vdev[i].nredundancy; j++) {
	    ip = Vdev[i].server[j].ip;
	    hostname = Vdev[i].server[j].hostname;
	    det_abc=1;
	    for (int k=0; k < strlen(ip); k++) {
		letter = ip[k];
		if (isdigit((unsigned char)letter || letter=='.')) {
		    det_abc = 0;
		    printf("%c", letter);
		} else {
		    det_abc = 1;
		    break;
		}
		printf("\n");
	    }
	    if (det_abc == 1) {
		strcpy( hostname, ip );
		hostent0 = gethostbyname( hostname );
		if ( hostent0 == NULL ) {
		    WARN( 0, "May be set invalid hostname \"%s\" to DSCUDA_SERVER or something.\n", hostname );
		    WARN( 0, "Program terminated.\n\n\n\n" );
		    exit(EXIT_FAILURE);
		} else {
		    ip_ref = inet_ntoa( *(in_addr*)hostent0->h_addr_list[0] );
		    strcpy( ip, ip_ref );
		}
	    }
	}
    } // for (int i=0; ...
} //---> void ClientState::initVirtualDevice(void)
unsigned
ClientState::getIpAddress(void) {
    return this->ip_addr;
}
void
ClientState::useIbv(void) {
    this->use_ibv = true;
}
void
ClientState::useRpc(void) {
    this->use_ibv = false;
}
bool
ClientState::isIbv(void) {
    return this->use_ibv;     
}
bool
ClientState::isRpc(void) {
    return !this->use_ibv;     
}
void
ClientState::setRollbackCalling(void) {
    this->rollback_calling = true;
}
void
ClientState::unsetRollbackCalling(void) {
    this->rollback_calling = false;
}
bool
ClientState::isRollbackCalling(void) {
    return this->rollback_calling;
}
void
ClientState::setMyIPAddr(unsigned val) {
    this->ip_addr = val;
}

ServerArray::ServerArray(void) {
    num = 0;
}
//*********************************************************
//*** CLASS: PtxRecord
//*********************************************************
PtxRecord::PtxRecord(void) {
    if (RC_KMODULENAMELEN < 16) {
	WARN(0, "%s():RC_KMODULENAMELEN is too small.\n", __func__);
	exit(1);
    }
    if (RC_KMODULEIMAGELEN < 16) {
	WARN(0, "%s():RC_KMODULEIMAGELEN is too small.\n", __func__);
	exit(1);
    }
    strcpy(name, "unknown");
    strcpy(ptx_image, "empty");
    valid = 0;
}
void
PtxRecord::invalidate(void) {
    strncpy(name, "unknown", RC_KMODULENAMELEN);
    strncpy(ptx_image, "empty", RC_KMODULEIMAGELEN);
    valid = 0;
}

void PtxRecord::set(char *name0, char *ptx_image0) {
    strncpy(name,      name0,      RC_KMODULENAMELEN);
    strncpy(ptx_image, ptx_image0, RC_KMODULEIMAGELEN);
    valid = 1;
    
    if (name[RC_KMODULENAMELEN-1] != '\0') {
	WARN(0, "%s():RC_KMODULENAMELEN is too small.\n");
	exit(1);
    }
    if (ptx_image[RC_KMODULEIMAGELEN-1] != '\0') {
	WARN(0, "%s():RC_KMODULEIMAGELEN is too small.\n");
	exit(1);
    }
}
//*********************************************************
//*** CLASS: PtxStore
//*********************************************************
PtxStore::PtxStore(void) {
    used_count = 0;
}
PtxRecord*
PtxStore::add(char *name0, char *ptx_image0) {
    PtxRecord *ptx_ptr = &ptx_record[used_count];
    if (used_count > RC_NKMODULEMAX) {
	WARN(0, "PtxStore::%s(): PtxStore array FULL!\n");
	exit(1);
    }
    ptx_ptr->set(name0, ptx_image0);
    used_count++;
    return ptx_ptr;
}

PtxRecord*
PtxStore::query(char *name0) {
    PtxRecord *ptx_ptr;
    for (int i=0; i<RC_NKMODULEMAX; i++) {
	ptx_ptr = &ptx_record[i];
	if ( strcmp(name0, ptx_ptr->name)==0 ) {/*found*/
	    WARN(9, "      +PtxStore::query(): Found ptx.\n")
	    return ptx_ptr;
	}
    }
    WARN(5, "      + PtxStore::query(): Not found ptx.\n");
    return NULL;
}

void
PtxStore::print(int n) {
    for (int i=0; i<n; i++) {
	WARN(1, "ptx_record[%d]: valid=%d, name=%s.\n",
	     i, ptx_record[i].valid, ptx_record[i].name); 
    }
}
ClientModule::ClientModule(void) {
    valid    = -1;
    id       = -1;
    ptx_data = NULL;
}
int
ClientModule::isValid(void) {
    if (valid<-1 || valid>1) {
	fprintf(stderr, "Unexpected error. %s:%d\n", __FILE__, __LINE__);
	exit(1);
    } else if (valid==1) {
	return 1;
    } else {
	return 0;
    }
}

int
ClientModule::isInvalid(void) {
    if (valid<-1 || valid>1) {
	fprintf(stderr, "Unexpected error. %s:%d\n", __FILE__, __LINE__);
	exit(1);
    } else if (valid==1) {
	return 0;
    } else {
	return 1;
    }
}

int
ServerArray::append(const char *ip, int ndev, const char *hname) {
    if ( num >= (RC_NVDEVMAX - 1) ) {
	WARN(0, "(+_+) Too many DS-CUDA daemons, exceeds RC_NVDEVMAX(=%d)\n",
	     RC_NVDEVMAX);
	exit(EXIT_FAILURE);
    }
    svr[num].setIP( ip );
    strcpy(svr[num].hostname, hname);
    svr[num].setID( ndev );
    svr[num].setCID( ndev );
    svr[num].uniq = RC_UNIQ_CANDBASE + num;
    num++;
    return 0;
}
int
ServerArray::append(PhyDev *svrptr) {
    if ( num >= (RC_NVDEVMAX - 1) ) {
	WARN(0, "(+_+) Too many DS-CUDA daemons, exceeds RC_NVDEVMAX(=%d)\n",
	     RC_NVDEVMAX);
	exit(EXIT_FAILURE);
    }
    svr[num].setIP( svrptr->ip );
    strcpy(svr[num].hostname, svrptr->hostname);
    svr[num].setID( svrptr->id );
    svr[num].setCID( svrptr->cid );
    svr[num].setUNIQ( RC_UNIQ_CANDBASE + num );
    svr[num].setFTMODE( svrptr->ft_mode );
    num++;
    return 0;
}
PhyDev*
ServerArray::findSpareOne(void) {
    PhyDev *sp = NULL;
    for (int i=0; i<num; i++) {
	if (svr[i].ft_health==hl_GOOD || svr[i].ft_health==hl_RECYCLED) {
	    sp = &svr[i];
	}
    }
    return sp;
}
PhyDev*
ServerArray::findBrokenOne(void) {
    PhyDev *sp = NULL;
    for (int i=0; i<num; i++) {
	if (svr[i].ft_health==hl_BAD) {
	    sp = &svr[i];
	}
    }
    return sp;
}
void
ServerArray::captureEnv(char *env_str, FThealth cond) {
    char buf[1024*RC_NVDEVMAX];
    char *svr_token;
    char svr_token_ar[RC_NVDEVMAX][256];

    {/* update buf[] */
	char *env = getenv(env_str);
	if (env == NULL) {
	    INFO0("[ Environment variable ] %s = (Not found)\n", env_str);
	    return;
	}
	if (sizeof(buf) < strlen(env)) {
	    WARN(0, "Too long length of DSCUDA_SERVER.\n");
	    exit(EXIT_FAILURE);
	}
	strncpy(buf, env, sizeof(buf));
    }

    //<--- svr_token_sr[x]="hostname:n"
    int  svr_count = 0;    
    svr_token = strtok(buf, " ");
    while (svr_token != NULL) {
	strcpy(svr_token_ar[svr_count], svr_token);
	svr_count++;
	if (svr_count > RC_NVDEVMAX) {
	    WARN(0, "number of devices exceeds the limit, RC_NVDEVMAX (=%d).\n",
		 RC_NVDEVMAX);
	    exit(EXIT_FAILURE);
	}
	svr_token = strtok(NULL, " ");
    }

    for (int i=0; i<svr_count; i++) {
	svr_token = strtok( svr_token_ar[i], ":" );
	this->svr[i].setIP( svr_token );
	svr_token = strtok( NULL, ":" );
	svr[i].setCID( svr_token );
	svr[i].setHealth(cond);
    }
    this->num = svr_count;
}//-->captureEnv()
void
ServerArray::print(void) {
    WARN(5, "ServerArray.num = %d\n", num);
    for (int i=0; i<num; i++) {
	WARN(1, "      + svrarr[%d].id= %d\n", i, svr[i].id);
	WARN(1, "      + svrarr[%d].cid= %d\n", i, svr[i].cid);
	WARN(1, "      + svrarr[%d].ip= %s\n", i, svr[i].ip);
	WARN(1, "      + svrarr[%d].hostname= %s\n", i, svr[i].hostname);
    }
}

void
FToption::infoD2H(void) {
    WARN(1, "d2h_simple  =%d\n", (d2h_simple)?   1:0 );
    WARN(1, "d2h_reduncpy=%d\n", (d2h_reduncpy)? 1:0 );
    WARN(1, "d2h_compare =%d\n", (d2h_compare)?  1:0 );
    WARN(1, "d2h_statics =%d\n", (d2h_statics)?  1:0 );
    WARN(1, "d2h_rollback=%d\n", (d2h_rollback)? 1:0 );
}

int
requestDaemonForDevice(char *ip, int devid, bool useibv) {
    int dsock; // socket for side-band communication with the daemon & server.
    int sport; // port number of the server. given by the daemon.
    char msg[256];
    sockaddr_in sockaddr;

    sockaddr = setupSockaddr( ip, RC_DAEMON_IP_PORT );
    dsock = socket(AF_INET, SOCK_STREAM, 0);
    if (dsock < 0) {
        perror("socket");
        exit(1);
    }
    
    if ( connect(dsock, (struct sockaddr *)&sockaddr, sizeof(sockaddr)) == -1 ) {
        perror("(;_;) connect(...)");
	WARN(0, "+--- Program terminated at %s:L%d\n", __FILE__, __LINE__ );
	WARN(0, "+--- Maybe DS-CUDA daemon is not running...\n" );
        //exit(1);
	return -1;
    }
    sprintf(msg, "deviceid:%d", devid);
    sendMsgBySocket(dsock, msg);
    WARN(1, "<--- Send message: \"%s\".\n", msg);

    memset(msg, 0, strlen(msg));
    recvMsgBySocket(dsock, msg, sizeof(msg));
    WARN(1, "---> Recv message: \"%s\".\n", msg);    
    sscanf(msg, "sport:%d", &sport);

    if (sport < 0) {
        WARN(0, "max possible ports on %s already in use.\n", ip);
        exit(1);
    }

    WARN(3, "server port: %d  daemon socket: %d\n", sport, dsock);

    if (useibv) {
        sprintf(msg, "remotecall:ibv");
    } else {
        sprintf(msg, "remotecall:rpc");
    }
    WARN(3, "send \"%s\" to the server.\n", msg);
    sendMsgBySocket(dsock, msg);

    WARN(2, "waiting for the server to be set up...\n");
    memset(msg, 0, strlen(msg));
    recvMsgBySocket(dsock, msg, sizeof(msg)); // wait for "ready" from the server.
    if (strncmp("ready", msg, strlen("ready"))) {
        WARN(0, "unexpected message (\"%s\") from the server. abort.\n", msg);
        exit(1);
    }
    return sport;
}

/*
 * Obtain a small integer unique for each thread.
 * The integer is used as an index to 'Vdevid[]'.
 */
int
vdevidIndex(void) {
    int i;
    pthread_t ptid = pthread_self();

    for (i=0; i<VdevidIndexMax; i++) {
        if (VdevidIndex2ptid[i] == ptid) {
            return i;
        }
    }

    pthread_mutex_lock(&VdevidMutex);
    i = VdevidIndexMax;
    VdevidIndex2ptid[i] = ptid;
    VdevidIndexMax++;
    pthread_mutex_unlock(&VdevidMutex);

    if (RC_NPTHREADMAX <= VdevidIndexMax) {
        fprintf(stderr, "vdevidIndex():device requests from too many (more than %d) pthreads.\n", RC_NPTHREADMAX);
        exit(1);
    }

    return i;
}

void
RCmappedMemRegister(void *pHost, void* pDevice, size_t size) {
    RCmappedMem *mem = (RCmappedMem *)malloc(sizeof(RCmappedMem));
    if (!mem) {
        perror("RCmappedMemRegister");
    }
    mem->pHost = pHost;
    mem->pDevice = pDevice;
    mem->size = size;
    mem->prev = RCmappedMemListTail;
    mem->next = NULL;
    if ( !RCmappedMemListTop ) { // mem will be the 1st entry.
        RCmappedMemListTop = mem;
    } else {
        RCmappedMemListTail->next = mem;
    }
    RCmappedMemListTail = mem;
}

RCmappedMem*
RCmappedMemQuery(void *pHost) {
    RCmappedMem *mem = RCmappedMemListTop;
    while (mem) {
        if (mem->pHost == pHost) {
            return mem;
        }
        mem = mem->next;
    }
    return NULL; // pHost not found in the list.
}

void
RCmappedMemUnregister(void *pHost) {
    RCmappedMem *mem = RCmappedMemQuery(pHost);
    if (!mem) return;

    if (mem->prev) { // reconnect the linked list.
        mem->prev->next = mem->next;
    } else { // mem was the 1st entry.
        RCmappedMemListTop = mem->next;
        if (mem->next) {
            mem->next->prev = NULL;
        }
    }
    if ( !mem->next ) { // mem was the last entry.
        RCmappedMemListTail = mem->prev;
    }
    free( mem );
}

//#if RC_SUPPORT_STREAM
/*
 * Register a stream array. each component is associated to a stream
 * on each Server[]. User see only the 1st element, streams[0].
 * Others, i.e., streams[1..Nredunddancy-1], are used by this library
 * to handle redundant calculation mechanism.
 */
static void
RCstreamArrayRegister(cudaStream_t *streams) {
    RCstreamArray *st = (RCstreamArray *)malloc(sizeof(RCstreamArray));
    if (!st) {
        perror("RCstreamArrayRegister");
    }
    for (int i=0; i<RC_NREDUNDANCYMAX; i++) {
        st->s[i] = streams[i];
    }
    st->prev = RCstreamArrayListTail;
    st->next = NULL;
    if (!RCstreamArrayListTop) { // st will be the 1st entry.
        RCstreamArrayListTop = st;
    } else {
        RCstreamArrayListTail->next = st;
    }
    RCstreamArrayListTail = st;
}

#if 0
static void
showsta(void) {
    RCstreamArray *st = RCstreamArrayListTop;
    while (st) {
        fprintf(stderr, ">>> 0x%08llx    prev:%p  next:%p\n", st, st->prev, st->next);
        st = st->next;
    }
}
#endif

RCstreamArray*
RCstreamArrayQuery(cudaStream_t stream0) {
    static RCstreamArray default_stream = { 0,};

    if (stream0 == 0) {
        return &default_stream;
    }

    RCstreamArray *st = RCstreamArrayListTop;
    while (st) {
        if (st->s[0] == stream0) {
            return st;
        }
        st = st->next;
    }
    return NULL;
}

static void
RCstreamArrayUnregister(cudaStream_t stream0) {
    RCstreamArray *st = RCstreamArrayQuery(stream0);
    if (!st) return;

    if (st->prev) { // reconnect the linked list.
        st->prev->next = st->next;
    } else { // st was the 1st entry.
        RCstreamArrayListTop = st->next;
        if (st->next) {
            st->next->prev = NULL;
        }
    }
    if (!st->next) { // st was the last entry.
        RCstreamArrayListTail = st->prev;
    }
    free(st);
    //    showsta();
}
//#endif


/*
 * Register a cudaArray array. each component is associated to a cudaArray
 * on each Server[]. User see only the 1st element, cuarrays[0].
 * Others, i.e., cuarrays[1..Nredunddancy-1], are used by this library
 * to handle redundant calculation mechanism.
 */
void
RCcuarrayArrayRegister(cudaArray **cuarrays) {
    RCcuarrayArray *ca = (RCcuarrayArray *)malloc(sizeof(RCcuarrayArray));
    if (!ca) {
        perror("RCcuarrayArrayRegister");
    }
    for (int i = 0; i < RC_NREDUNDANCYMAX; i++) {
        ca->ap[i] = cuarrays[i];
    }
    ca->prev = RCcuarrayArrayListTail;
    ca->next = NULL;
    if (!RCcuarrayArrayListTop) { // ca will be the 1st entry.
        RCcuarrayArrayListTop = ca;
    } else {
        RCcuarrayArrayListTail->next = ca;
    }
    RCcuarrayArrayListTail = ca;
}

RCcuarrayArray*
RCcuarrayArrayQuery(cudaArray *cuarray0) {
    RCcuarrayArray *ca = RCcuarrayArrayListTop;
    while (ca) {
        if (ca->ap[0] == cuarray0) {
            return ca;
        }
        ca = ca->next;
    }
    return NULL;
}

void
RCcuarrayArrayUnregister(cudaArray *cuarray0) {
    RCcuarrayArray *ca = RCcuarrayArrayQuery(cuarray0);
    if (!ca) return;

    if (ca->prev) { // reconnect the linked list.
        ca->prev->next = ca->next;
    } else { // ca was the 1st entry.
        RCcuarrayArrayListTop = ca->next;
        if (ca->next) {
            ca->next->prev = NULL;
        }
    }
    if (!ca->next) { // ca was the last entry.
        RCcuarrayArrayListTail = ca->prev;
    }
    free(ca);
}


/*
 * Register an event array. each component is associated to an event
 * on each Server[]. User see only the 1st element, events[0].
 * Others, i.e., events[1..Nredunddancy-1], are used by this library
 * to handle redundant calculation mechanism.
 */
void
RCeventArrayRegister(cudaEvent_t *events) {
    RCeventArray *ev = (RCeventArray *)malloc(sizeof(RCeventArray));
    if (!ev) {
        perror("RCeventArrayRegister");
    }
    for (int i = 0; i < RC_NREDUNDANCYMAX; i++) {
        ev->e[i] = events[i];
    }
    ev->prev = RCeventArrayListTail;
    ev->next = NULL;
    if (!RCeventArrayListTop) { // ev will be the 1st entry.
        RCeventArrayListTop = ev;
    } else {
        RCeventArrayListTail->next = ev;
    }
    RCeventArrayListTail = ev;
}

RCeventArray*
RCeventArrayQuery(cudaEvent_t event0) {
    RCeventArray *ev = RCeventArrayListTop;
    while (ev) {
        if (ev->e[0] == event0) {
            return ev;
        }
        ev = ev->next;
    }
    return NULL;
}

void
RCeventArrayUnregister(cudaEvent_t event0) {
    RCeventArray *ev = RCeventArrayQuery(event0);
    if (!ev) return;

    if (ev->prev) { // reconnect the linked list.
        ev->prev->next = ev->next;
    } else { // ev was the 1st entry.
        RCeventArrayListTop = ev->next;
        if (ev->next) {
            ev->next->prev = NULL;
        }
    }
    if (!ev->next) { // ev was the last entry.
        RCeventArrayListTail = ev->prev;
    }
    free(ev);
}
/*
 * Compose UVA from GPU local address and its deviceID.
 */
void*
dscudaUvaOfAdr( void *adr, int devid ) {
    DscudaUva_t adri = (DscudaUva_t)adr;
#if __LP64__
    adri |= ((DscudaUva_t)devid << 48);
#endif
    return (void *)adri;
}
/*====================================================================
 * Get GPU deviceID from UVA.
 */
int
dscudaDevidOfUva( void *adr ) {
#if __LP64__
    DscudaUva_t adri = (DscudaUva_t)adr;
    int devid = adri >> 48;
    return devid;
#else
    return 0;
#endif
}
/*
 * Get GPU local address from UVA.
 */
void*
dscudaAdrOfUva( void *adr ) {
    DscudaUva_t adri = (DscudaUva_t)adr;
#if __LP64__
    adri &= 0x0000ffffffffffffLL;
#endif
    return (void *)adri;
}

static char*
readServerConf(char *fname) {
    FILE *fp = fopen(fname, "r");
    char linebuf[1024];
    int len;
    static char buf[1024*RC_NVDEVMAX];

    buf[0] = 0;
    if (!fp) {
        WARN(0, "cannot open file '%s'\n", fname);
        exit(1);
    }

    while (!feof(fp)) {
        char *s = fgets(linebuf, sizeof(linebuf), fp);
        if (!s) break;
        len = strlen(linebuf);
        if (linebuf[len-1] == '\n') {
            linebuf[len-1] = 0;
        }
        if (sizeof(buf) < strlen(buf) + len) {
            WARN(0, "readServerConf:file %s too long.\n", fname);
            exit(1);
        }
        strncat(buf, linebuf, sizeof(linebuf));
        strcat(buf, " ");
    }
    fclose(fp);
    return buf;
}

/*
 *
 */
void
printVirtualDeviceList( void ) {
    VirDev     *pVdev;
    PhyDev   *pSvr;
    int         i,j;
    
    INFO0("###***   <---Start Virtual Device Construction. (Total %d device%s)\n", St.Nvdev, (St.Nvdev>1)? "s":"" );
    for (i=0, pVdev=St.Vdev; i<St.Nvdev; i++, pVdev++) {
	if (i >= RC_NVDEVMAX) {
	    INFO0("(;_;) Too many virtual devices. %s().\nexit.", __func__);
	    exit(EXIT_FAILURE);
	}
	if (pVdev->nredundancy == 1) {
	    INFO0("    Virt[%d] (MONO)\n", i );
	} else if ( pVdev->nredundancy > 1 ) {
	    INFO0("    Virt[%d] (POLY:%d)\n", i, pVdev->nredundancy );
	} else {
	    INFO0("    Virt[%d] (????:%d)\n", i, pVdev->nredundancy );
	}
	
	for (j=0, pSvr=pVdev->server; j<pVdev->nredundancy; j++, pSvr++) {
	    if (j >= RC_NREDUNDANCYMAX) {
		WARN(0, "(;_;) Too many redundant devices %d. %s().\nexit.\n", __func__);
		exit( EXIT_FAILURE );
	    }
	    INFO0("    +  Phy[%d]: id=%d, cid=%d, IP=%s(%s), uniq=%d.\n", j,
		   pSvr->id, pSvr->cid, pSvr->ip, pSvr->hostname, pSvr->uniq);
	}
    }
    INFO0("###***   --->Stop Virtual Device Construction.\n\n");

    if (St.ft_mode==FT_BYCPY || St.ft_mode==FT_BYTIMER) {
	/*
	 * Device Candidates
	 */
	INFO0("###*** Physical Device Candidates. (Ncand=%d)\n", SvrCand.num );
	for( i=0, pSvr=SvrCand.svr; i < SvrCand.num; i++, pSvr++ ){
	    if (i >= RC_NVDEVMAX) {
		WARN(0, "(;_;) Too many candidate devices. %s().\nexit.", __func__);
		exit( EXIT_FAILURE );
	    }
	    INFO0("###***    - Cand[%2d]: id=%d, cid=%d, IP=%s, uniq=%d.\n", i,
		  pSvr->id, pSvr->cid, pSvr->ip, pSvr->uniq);
	}
	/*
	 * Alternate Devices
	 */
	INFO0("###*** Spare Server Info.(Nspare=%d)\n", SvrSpare.num);
	for( i=0, pSvr=SvrSpare.svr; i < SvrSpare.num; i++, pSvr++ ){
	    if (i >= RC_NVDEVMAX) {
		WARN(0, "(;_;) Too many spare devices. %s().\nexit.", __func__);
		exit( EXIT_FAILURE );
	    }
	    INFO0("###***    - Spare[%d]: id=%d, cid=%d, IP=%s, uniq=%d.\n", i,
		  pSvr->id, pSvr->cid, pSvr->ip, pSvr->uniq);
	}
    }
}

void
VirDev::setConfInfo(int redun) {
    nredundancy = redun; //Update Vdev.nredundancy.
    if (redun == 1) {
	conf = VDEV_MONO;
	sprintf(info, "MONO");
    } else if (redun > 1) {
	conf = VDEV_POLY;
	sprintf(info, "POLY%d", redun);
    } else {
	WARN(0, "Detect invalid nredundancy = %d.\n", redun);
	exit(EXIT_FAILURE);
    }
}

void
VirDev::printModuleList(void) {
    const int len = 256;
    char printbuf[len];
    int valid_cnt = 0;
    
    WARN(5, "====================================================\n");
    WARN(5, "===  VirDev::%s(void)\n", __func__ );
    WARN(5, "====================================================\n");
    WARN(5, "RC_NKMODULEMAX= %d\n", RC_NKMODULEMAX);
    
    for (int i=0; i<RC_NKMODULEMAX; i++) {
	if( modulelist[i].valid==1 || modulelist[i].valid==0 ) {
	    WARN( 5, "Virtual[%d]:modulelist[%d]:\n", id, i);
	    WARN( 5, "    + name= %s\n", modulelist[i].ptx_data->name);
	    //printf("    + send_time= \n", sent_time., sent_time.);
	    //strncpy(printbuf, modulelist[i].ptx_data->ptx_image, len - 1 );
	    //printbuf[255]='\0';
	    //printf("# %s():    + ptx_image=\n%s\n", __func__, printbuf);
	    valid_cnt++;
	}
    }
    WARN(5, "%d valid modules registered.\n",  valid_cnt);
    WARN(5, "====================================================\n");
    
}

void
printModuleList(void) {
    for (int i=0; i<St.Nvdev; i++) {
	St.Vdev[i].printModuleList();
    }
}
uint32_t
dscuda::calcChecksum(void *sta, size_t size_byte) {
    uint32_t *p = (uint32_t *)sta;
    uint32_t  s           = 0;
    uint32_t  s_remain    = 0; // zero padding, ignore sign bit.
    size_t    sum_count   = size_byte / sizeof(uint32_t);
    size_t    size_remain = size_byte - (sum_count * sizeof(uint32_t));
    for (int i=0; i<sum_count; i++) {
	s += *p;
	p++;
    }
    if (size_remain >= sizeof(s_remain)) {
	fprintf(stderr, "Unexpected ERROR: %s()\n", __func__);
	exit(1);
    }
    memcpy( &s_remain, p, size_remain );
    s += s_remain;
    return s;
}
int
dscuda::searchDaemon(void) {
    int sendsock;
    int recvsock;

    char sendbuf[SEARCH_BUFLEN_TX];
    char recvbuf[SEARCH_BUFLEN_RX];
    
    int recvlen;
    int num_daemon = 0;
    int num_device = 0;
    int num_ignore = 0;

    unsigned int adr, mask;
    socklen_t    sin_size;
    int          setsockopt_ret;

    sockaddr_in addr, svr;
    struct ifreq ifr[2];
    struct ifconf ifc;
    struct passwd *pwd;

    INFO0("[ Constant             ] RC_DAEMON_IP_PORT = %d\n", RC_DAEMON_IP_PORT);
    sendsock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    recvsock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if ( sendsock == -1 || recvsock == -1 ) {
	perror("searchDaemon: socket()");
	exit(1);
    }
    {
	int val = 1;
	setsockopt_ret = setsockopt(sendsock, SOL_SOCKET, SO_BROADCAST, &val, sizeof(val));
	if (setsockopt_ret != 0) {
	    perror("searchDaemon: setsockopt()");
	    exit(1);
	}
    }
    ifc.ifc_len = sizeof(ifr) * 2;
    ifc.ifc_ifcu.ifcu_buf = (char *)ifr;
    ioctl(sendsock, SIOCGIFCONF, &ifc);

    ifr[1].ifr_addr.sa_family = AF_INET;
    ioctl(sendsock, SIOCGIFADDR, &ifr[1]);
    adr = ((sockaddr_in *)(&ifr[1].ifr_addr))->sin_addr.s_addr;
    ioctl(sendsock, SIOCGIFNETMASK, &ifr[1]);
    mask = ((sockaddr_in *)(&ifr[1].ifr_netmask))->sin_addr.s_addr;

    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(RC_DAEMON_IP_PORT - 1);
    addr.sin_addr.s_addr = adr | ~mask;

    strncpy( sendbuf, SEARCH_PING, SEARCH_BUFLEN_TX - 1 );
    sendto( sendsock, sendbuf, SEARCH_BUFLEN_TX, 0, (sockaddr *)&addr, sizeof(addr));
    INFO0("Broadcast \"%s\" message\n", SEARCH_PING);
    sin_size = sizeof(sockaddr_in);

    svr.sin_family      = AF_INET;
    svr.sin_port        = htons(RC_DAEMON_IP_PORT - 2);
    svr.sin_addr.s_addr = htonl(INADDR_ANY);
    
    // Set timeout for recvsock.
    {
	timeval tout;
	tout.tv_sec  = RC_SEARCH_DAEMON_TIMEOUT ;
	tout.tv_usec = 0;
	setsockopt_ret = setsockopt(recvsock, SOL_SOCKET, SO_RCVTIMEO, (char *)&tout, sizeof(tout));
	if (setsockopt_ret != 0) {
	    perror("searchDaemon: setsockopt(recvsock)");
	    exit(1);
	}
    }

    {
	int bind_ret = bind( recvsock, (struct sockaddr *)&svr, sizeof(svr) );
	if( bind_ret != 0 ) {
	    fprintf(stderr, "Error: bind() returned %d. recvsock=%d, port=%d\n",
		    bind_ret, recvsock, svr.sin_port); //port:38655
	    perror("searchDaemon: bind()");
	    return -1;
	}
    }
    
    pwd = getpwuid( getuid() );

    /* Recieve ack message from dscudad running at other host. */
    char *magic_word;
    char *user_name;
    char *host_name;
    char *dev_count;
    char  ipaddr[32];
    int   num_eachdev;

    SvrCand.num = 0;

    memset( recvbuf, 0, SEARCH_BUFLEN_RX );
    while(( recvlen = recvfrom( recvsock, recvbuf, SEARCH_BUFLEN_RX - 1, 0, (struct sockaddr *)&svr, &sin_size)) > 0) {
	INFO0(" + Detect ACK \"%s\" ", recvbuf);
	/*
	 * Analyze message.
	 */
	magic_word = strtok( recvbuf, SEARCH_DELIM );
	user_name  = strtok( NULL,    SEARCH_DELIM );
	host_name  = strtok( NULL,    SEARCH_DELIM );
	dev_count  = strtok( NULL,    SEARCH_DELIM ); // Ndev=4
	sscanf( dev_count, "Ndev=%d", &num_eachdev );
	sprintf( ipaddr, "%s", inet_ntoa( svr.sin_addr )); //192.168.1.1
	if ( magic_word == NULL ) {
	    WARN(0, "\n\n###(ERROR) Unexpected token in %s().\n\n", __func__);
	    exit(1);
	} else {
	    INFO0("from \"%s\" ", ipaddr );
	    if ( strcmp( magic_word, SEARCH_ACK   )==0 &&
		 strcmp( user_name,  pwd->pw_name )==0 ) { /* Found */
		INFO0("valid.\n");
		/*
		 * Updata SvrCand;
		 */
		for (int d=0; d<num_eachdev; d++) {
		    SvrCand.append(ipaddr, d, host_name);
		}
		num_daemon += 1;
		num_device += num_eachdev;
	    } else {
		INFO0("ignored.\n");
		num_ignore++;
	    }
	}
	memset( recvbuf, 0, SEARCH_BUFLEN_RX );
    }

    {
	int close_ret = close( sendsock );
	if ( close_ret != 0 ) {
	    WARN(0, "close(sendsock) failed.\n");
	    exit(EXIT_FAILURE);
	}
	
	close_ret = close( recvsock );
	if ( close_ret != 0 ) {
	    WARN(0, "close(recvsock) failed.\n");
	    exit(EXIT_FAILURE);
	}
    }

    if (num_daemon > 0) {
	INFO0("Found %d valid DSCUDA daemon%s. (%d ignored).\n",
	      num_daemon, (num_daemon>1)? "s":"", num_ignore );
    }
    else if (num_daemon==0) {
	//
	// Even if no daemons found, the servers defined in DSCUDA_SERVER are available.
	//
	WARN( 0, "%s(): Not found DS-CUDA daemon in this network.\n", __func__ );
	WARN( 0, "%s(): And Trying to continue execution.\n", __func__ );
    }
    else { 	/* Terminate program and exit. */
	WARN( 0, "%s(): Detected unexpected trouble; num_daemon=%d?\n", __func__, num_daemon );
	exit(EXIT_FAILURE);
    }
    return num_daemon;
} //---> int dscuda::searchDaemon(void)
//
//
//
#if 0
void
ServerArray::removeArray(ServerArray *sub) {

    for (int i=0; i<num; i++) {
	for (int k=0; k<sub->num; k++) {
	}
    }
}
#endif
static void
updateSpareServerList(void) {
    int         spare_count = 0;;
    VirDev     *pVdev;
    PhyDev *pSvr;

    for (int i=0; i<SvrCand.num; i++) {    // Sweep all Vdev.server[] and compare.
	int found = 0;
	pVdev = St.Vdev;
	for (int j=0; j<St.Nvdev; j++) {
	    pSvr = pVdev->server;
	    for (int k=0; k < pVdev->nredundancy; k++) {
		if ( strcmp( SvrCand.svr[i].ip,  pSvr->ip  )==0 &&
		     SvrCand.svr[i].cid==pSvr->cid ) { /* check same IP */
		    found=1;
		}
		pSvr++;
	    }
	    pVdev++;
	}
	if (found==0) { /* not found */
	    SvrSpare.svr[spare_count].id   = SvrCand.svr[i].id;
	    SvrSpare.svr[spare_count].cid  = SvrCand.svr[i].cid;
	    SvrSpare.svr[spare_count].uniq = SvrCand.svr[i].uniq;
	    SvrSpare.svr[spare_count].ft_health = hl_GOOD;
	    strcpy(SvrSpare.svr[spare_count].ip, SvrCand.svr[i].ip);
	    spare_count++;
	}
    }
    SvrSpare.num = spare_count;
}

static void
getenvDSCUDA_WARNLEVEL(void) {
    char *env = getenv("DSCUDA_WARNLEVEL");
    int val;
    if ( env ) {
        val = atoi(strtok(env, " "));
        if ( val >= 0 ) {
	    dscuda::setWarnLevel( val );
	} else {
	    WARN(0, "(;_;) Invalid DSCUDA_WARNLEVEL(%d), set 0 or positive integer.\n", val);
	    exit(EXIT_FAILURE);
	}
    } else {
	dscuda::setWarnLevel(RC_WARNLEVEL_DEFAULT);
    }

}
static void
extractENV(bool &bool_var, const char *envname) {
    char *env = getenv(envname);
    if (env==NULL) {
	bool_var = false;
    } else {
	int val = atoi(env);
	if (val==0) {
	    bool_var = false;
	} else {
	    bool_var = true;
	}
    }
}
static void
extractENV(int &int_var, const char *envname, int undef) {
    char *env = getenv(envname);
    if (env==NULL) {
	int_var = undef;
    } else {
	int_var = atoi(env);
    }
}
static void
extractENV(char *str_var, const char *envname, int len) {
    char *env = getenv(envname);
    if (env==NULL) {
	strncpy(str_var, "(undef)", len);  
    } else {
	strncpy(str_var, env, len);  
    }
}
/****** CHECK-POINTING THREAD ****************************************
 * Take the data backups of each virtualized GPU to client's host
 * memory after verifying between redundant physical GPUs every
 * specified wall clock
 * time period. The period is defined in second.
 */

void*
periodicCheckpoint(void *arg) {
    int cp_period = *(int *)arg;
    int cp_trim = 3;
    int trim_grid_usec = 100000; //100msec
    int correct_count = 0;
    int faulted_count = 0;
    int cp_count = 1;
    int Tc_reset_req0 = 0;
    //<-- timer
    double Tc_exp,Tc_exp_l=(double)cp_period*0.7, Tc_exp_h=(double)cp_period*1.3;
    int    Tc_exp_sec;
    double Tc_exp_usec;
    double Ta, Ta_sum=0.0, Ta_sta, Ta_avr=-1.0, Ta_min=1.0e6, Ta_max=0.0; //all
    double Tm, Tm_sum=0.0, Tm_sta, Tm_avr=0.0,  Tm_min=1.0e6, Tm_max=0.0; //mutexlock
    double Ts, Ts_sum=0.0, Ts_sta, Ts_avr=-1.0, Ts_min=1.0e6, Ts_max=0.0; //store
    double Tc, Tc_sum=0.0, Tc_sta, Tc_avr=-1.0, Tc_min=1.0e6, Tc_max=0.0; //cp sleep
    double Tr, Tr_sum=0.0, Tr_sta, Tr_avr=-1.0, Tr_min=1.0e6, Tr_max=0.0; //restre mem
    double Tx, Tx_sum=0.0, Tx_sta, Tx_avr=-1.0, Tx_min=1.0e6, Tx_max=0.0; //redo api
    //
    double Td2h, Td2h_partial, Td2h_sum=0.0, Td2h_sta, Td2h_avr=-1.0, Td2h_min=1.0e6,
	Td2h_max=0.0; //Tr+Tx in D2h
    //--> timer 
    /*
      Memo: "Tc" is defined by "int cp_period" above.
            "Ts" is not able to defined.
      |<----  Ta  ----->| 
      |<--  Tc  -->| Ts |
      +------------+====+
      |         |Tm|
     */
    while (cp_thread_exit==0) {
	dscuda::stopwatch(&Ta_sta);
	dscuda::stopwatch(&Tc_sta);
	Td2h = 0.0;
	//<-- Wait for specified period (sec) passed.
	Tc_exp = ((double)cp_period * (double)cp_count) - Tc_sum - Tm_avr; //in sec
	if      (Tc_exp < Tc_exp_l) Tc_exp = Tc_exp_l; // saturate to lower bound time.
	else if (Tc_exp > Tc_exp_h) Tc_exp = Tc_exp_h; //             upper bound time.
	Tc_exp_sec  = (int)floor(Tc_exp);
	Tc_exp_usec = (Tc_exp - Tc_exp_sec)*1e6;
	for (int i=0; i<Tc_exp_sec; i++) {
	    for (int j=0; j<10; j++) { // 1.000s = 100ms * 10;
		//<-- mutex lock
		dscuda::stopwatch(&Td2h_sta);
		do {
		    pthread_mutex_lock( &Tc_reset_mutex );
		    Tc_reset_req0 = Tc_reset_req;
		    pthread_mutex_unlock( &Tc_reset_mutex );
		} while (Tc_reset_req0 == 1);
		Td2h_partial = dscuda::stopwatch(&Td2h_sta);
		Td2h += Td2h_partial;
		//--> mutex lock
		usleep( 100000 ); // =100ms
	    }
	}
	//<-- mutex lock
	dscuda::stopwatch(&Td2h_sta);
	do {
	    pthread_mutex_lock( &Tc_reset_mutex );
	    Tc_reset_req0 = Tc_reset_req;
	    pthread_mutex_unlock( &Tc_reset_mutex );
	} while (Tc_reset_req0 == 1);
	Td2h_partial = dscuda::stopwatch(&Td2h_sta);
	Td2h += Td2h_partial; // Fix the value of Td2h.
	//--> mutex lock
	usleep( (int)Tc_exp_usec );
	
	//--> Wait for specified period (sec) passed.

	dscuda::stopwatch(&Tm_sta);
	//<-- mutex locks for avoiding R/W collisions
	pthread_mutex_lock( &cudaMemcpyD2H_mutex );
	pthread_mutex_lock( &cudaMemcpyH2D_mutex );
	pthread_mutex_lock( &cudaKernelRun_mutex );
	pthread_mutex_lock( &cudaElse_mutex );
	//--> mutex locks for avoiding R/W collisions

	//<-- flush all cuda stream
	for (int i=0; i<St.Nvdev; i++) {
	    St.Vdev[i].cudaThreadSynchronize();
	}
	//--> flush all cuda stream
		    
	//*****
	//***** <-- "Ts" starts here.
	//*****
	Tm = dscuda::stopwatch(&Tm_sta, &Tm_min, &Tm_max);
	Tm_sum += Tm;
	Tm_avr =  Tm_sum / (double)cp_count;

	//Tc = dscuda::stopwatch(&Tc_sta, &Tc_min, &Tc_max);
	Tc = dscuda::stopwatch(&Tc_sta) - Td2h;
	if (Tc < Tc_min) Tc_min = Tc;
	if (Tc > Tc_max) Tc_max = Tc;
	Tc_sum += Tc;
	Tc_avr =  Tc_sum / (double)cp_count;

	//Td2h
	if (Td2h < Td2h_min) Td2h_min = Td2h;
	if (Td2h > Td2h_max) Td2h_max = Td2h;
	Td2h_sum += Td2h;
	Td2h_avr = Td2h_sum / (double)cp_count;
	
	dscuda::stopwatch(&Ts_sta);	
	//<-- Output beginning message.
	WARN_CP(0,"==================================================== #%d begin\n", cp_count);
	WARN_CP(0,"periodicCheckpoint( period = %d sec )\n", cp_period );
	//--> Output beginning message.

	//<-- copy from all cudaMalloc() regions of all devices.
	St.collectEntireRegions();
	//--> copy from all cudaMalloc() regions of all devices.

	bool correct = St.verifyEntireRegions();
#if 0 // force pseudo error
	if (correct_count % 5 == 4) {
	    correct = false;
	}
#endif
	if (correct) {
	    correct_count++;
	    //***
	    //*** All memory regions on all virtual devices are correct.
	    //*** Then, collect clean device memory regions to host memory.
	    //*** and clear CUDA API called history.
	    //***
	    for (int i=0; i<St.Nvdev; i++) {
		St.Vdev[i].updateMemlist();
	    }
	    WARN_CP(0, "(^_^)Update clean backup region, age=%d\n",
		    St.Vdev[0].memlist.getAge());
	    
	    for (int i=0; i<St.Nvdev; i++) {
		St.Vdev[i].clearReclist();
	    }
	}
	//*****
	//***** --> "Ts" completes here.
	//*****
	Ts = dscuda::stopwatch(&Ts_sta, &Ts_min, &Ts_max);
	Ts_sum += Ts;
	Ts_avr =  Ts_sum / (double)cp_count;
	
	if (!correct) {
	    faulted_count++;
	    //***
	    //*** Some memory regions on any virtual devices are currupted.
	    //*** Then, restore clean memory regions to all devices, and
	    //*** redo the historical cuda API calls.
	    //***
	    dscuda::stopwatch(&Tr_sta);
	    WARN_CP(0,"(+_+) Detect corrupted region.\n");
	    WARN_CP(0,"%8.3f sec from start. nth=%d\n",
		    Tr_sta - (double)St.start_time, faulted_count);
	    for (int i=0; i<St.Nvdev; i++) {
		St.Vdev[i].restoreMemlist();
	    }
	    Tr = dscuda::stopwatch(&Tr_sta, &Tr_min, &Tr_max);
	    Tr_sum += Tr;
	    Tr_avr =  Tr_sum / (double)faulted_count;

	    WARN_CP(0, "(._.)Completed restoring the device memory previous backup ");
	    WARN_CP0(0, "age=%d\n", St.Vdev[0].memlist.getAge());
		    
	    WARN_CP(0, "(+_+)Rollback the CUDA APIs by CP.\n");
	    dscuda::stopwatch(&Tx_sta);
	    for (int i=0; i<St.Nvdev; i++) {
		St.Vdev[i].reclist.print();
		St.Vdev[i].recordOFF();
		WARN_CP(1, "        VirDev[%d]\n", St.Vdev[i].id);
		St.Vdev[i].reclist.recall();
		St.Vdev[i].recordON();
	    }
	    //<-- flush all cuda stream
	    for (int i=0; i<St.Nvdev; i++) {
		St.Vdev[i].cudaThreadSynchronize();
	    }
	    WARN_CP(0, "Synchronize() Rollbacked CUDA APIs.\n");
	    //--> flush all cuda stream
	    Tx = dscuda::stopwatch(&Tx_sta, &Tx_min, &Tx_max);
	    Tx_sum += Tx;
	    Tx_avr =  Tx_sum / (double)faulted_count;
	}


	Ta = dscuda::stopwatch(&Ta_sta, &Ta_min, &Ta_max);
	Ta_sum  += Ta;
	Ta_avr  =  Ta_sum / (double)cp_count;
	
	//<-- Output ending message.	
	WARN_CP(0,"} elapsed time report #%d (sec)\n", cp_count);
	WARN_CP(0," 'Name' = 'now' { 'min' , 'avr' , 'max' } 'sum'\n");
	WARN_CP(0," Tm = %8.3f { %8.3f , %8.3f , %8.3f } %8.3f\n",
		Tm, Tm_min, Tm_avr, Tm_max, Tm_sum);
	WARN_CP(0," Tc = %8.3f { %8.3f , %8.3f , %8.3f } %8.3f\n",
		Tc, Tc_min, Tc_avr, Tc_max, Tc_sum);
	WARN_CP(0," Ts = %8.3f { %8.3f , %8.3f , %8.3f } %8.3f\n",
		Ts, Ts_min, Ts_avr, Ts_max, Ts_sum);
	WARN_CP(0," Ta = %8.3f { %8.3f , %8.3f , %8.3f } %8.3f\n",
		Ta, Ta_min, Ta_avr, Ta_max, Ta_sum);
	WARN_CP(0," Td2h = %8.3f { %8.3f , %8.3f , %8.3f } %8.3f\n",
		Td2h, Td2h_min, Td2h_avr, Td2h_max, Td2h_sum);
		
	if (faulted_count>0) {
	    WARN_CP(0," *Tr= %8.3f { %8.3f , %8.3f , %8.3f } %8.3f (%d)\n",
		    Tr, Tr_min, Tr_avr, Tr_max, Tr_sum, faulted_count);
	    WARN_CP(0," *Tx= %8.3f { %8.3f , %8.3f , %8.3f } %8.3f (%d)\n",
		    Tx, Tx_min, Tx_avr, Tx_max, Tx_sum, faulted_count);
	}
	else {
	    WARN_CP(0," *Tr= - { - , - , - } %8.3f (%d)\n", Tr_sum, faulted_count);
	    WARN_CP(0," *Tx= - { - , - , - } %8.3f (%d)\n", Tx_sum, faulted_count);
	}
	//<-- flush all cuda stream
	for (int i=0; i<St.Nvdev; i++) {
	    WARN_CP(0," Vdev[%d].ft_unmatch_count= %d\n", i, St.Vdev[i].ft_unmatch_total);
	}
	//--> flush all cuda stream
	pthread_testcancel();/* thread cancelation point */
	WARN_CP(0,"==================================================== #%d end\n", cp_count);
	//--> Output ending message.
	cp_count++;
	//<-- mutex unlocks for following R/W.
	pthread_mutex_unlock( &cudaMemcpyD2H_mutex );
	pthread_mutex_unlock( &cudaMemcpyH2D_mutex );
	pthread_mutex_unlock( &cudaKernelRun_mutex );
	pthread_mutex_unlock( &cudaElse_mutex );
	//--> mutex unlocks for following R/W.
    }//while(true)
    
    WARN_CP0(0,"periodicCheckpoint() thread completed.\n");
    WARN_CP0(0,"====================================================\n");
    WARN_CP0(0,"= Summary\n");
    WARN_CP0(0,"= Total Checkpointed count = %d times.\n", cp_count-1);
    WARN_CP0(0,"=       Correct      count = %d times.\n", correct_count);
    WARN_CP0(0,"=       Fault        count = %d times.\n", faulted_count);
    WARN_CP0(0,"=    : { 'min' , 'avr' , 'max' } 'sum' [sec]\n");
    WARN_CP0(0,"=  Tm: { %8.3f , %8.3f , %8.3f } %8.3f\n", Tm_min, Tm_avr, Tm_max, Tm_sum);
    WARN_CP0(0,"=  Tc: { %8.3f , %8.3f , %8.3f } %8.3f\n", Tc_min, Tc_avr, Tc_max, Tc_sum);
    WARN_CP0(0,"=  Ts: { %8.3f , %8.3f , %8.3f } %8.3f\n", Ts_min, Ts_avr, Ts_max, Ts_sum);
    WARN_CP0(0,"=  Ta: { %8.3f , %8.3f , %8.3f } %8.3f\n", Ta_min, Ta_avr, Ta_max, Ta_sum);
    if (faulted_count > 0) {
	WARN_CP0(0,"=  *Tr:{ %8.3f , %8.3f , %8.3f } %8.3f\n", Tr_min, Tr_avr, Tr_max, Tr_sum);
	WARN_CP0(0,"=  *Tx:{ %8.3f , %8.3f , %8.3f } %8.3f\n", Tx_min, Tx_avr, Tx_max, Tx_sum);
    }
    else {
	WARN_CP0(0,"=  *Tr= - { - , - , - } %8.3f (%d)\n", Tr_sum, faulted_count);
	WARN_CP0(0,"=  *Tx= - { - , - , - } %8.3f (%d)\n", Tx_sum, faulted_count);
    }
    WARN_CP0(0, "====================================================\n");
    sleep(1);
    return NULL;
} // periodicCheckpoint()

void
VirDev::invalidateAllModuleCache(void) {
    for (int i=0; i<RC_NKMODULEMAX; i++) {
        if( modulelist[i].isValid() ){
	    modulelist[i].invalidate();
	} else { 
	    continue;
	}
    }
}

/*
 * public functions
 */
int
dscudaNredundancy(void) {
    VirDev *vdev = St.Vdev + Vdevid[vdevidIndex()];
    return vdev->nredundancy;
}

void
dscudaSetErrorHandler(void (*handler)(void *), void *handler_arg) {
    errorHandler = handler;
    errorHandlerArg = handler_arg;
}

/*
 * Obtain a mangled symbol name of a function, whose
 * interface is given by 'funcif' and is defined somewhere in 'ptxdata'.
 * The obtained symbol name is returned to 'name'.
 *
 * eg) funcif  : void dscudavecAdd(dim3, dim3, size_t, CUstream_st*, float*, float*, float*)
 *     ptxdata : .version 1.4
 *               .target sm_10, map_f64_to_f32
 *               ...
 *               .entry _Z6vecAddPfS_S_ (
 *               ...
 *               } // _Z6vecMulPfS_fS_iPi
 */
void
dscudaGetMangledFunctionName(char *name, const char *funcif, const char *ptxdata)
{
    WARN(10, "<---Entering %s()\n", __func__);
    static char mangler[256] = {0, };
    char cmd[4096];
    FILE *outpipe;
    FILE *tmpfp;
    char ptxfile[1024];

    WARN(4, "getMangledFunctionName(%p, %p, %p)  funcif:\"%s\"\n",
         name, funcif, ptxdata, funcif);

    // create a tmporary file that contains 'ptxdata'.
    system("/bin/mkdir /tmp/dscuda 1> /dev/null  2> /dev/null");
    // do not use >& since /bin/sh on some distro does not recognize it.

    sprintf(ptxfile, "/tmp/dscuda/mgl%d", getpid());
    tmpfp = fopen(ptxfile, "w");
    fprintf(tmpfp, "%s", ptxdata);
    fclose(tmpfp);

    // exec 'ptx2symbol' to obtain the mangled name.
    // command output is stored to name.
    if (!mangler[0]) {
        sprintf(mangler, "%s/bin/ptx2symbol", St.dscuda_path);
    }
    sprintf(cmd, "%s %s << EOF\n%s\nEOF", mangler, ptxfile, funcif);
    outpipe = popen(cmd, "r");
    if (!outpipe) {
        perror("getMangledFunctionName()");
        exit(1);
    }
    fgets(name, 256, outpipe);
    pclose(outpipe);
    if (!strlen(name)) {
        WARN(0, "getMangledFunctionName() : %s returned an error. "
             "it could not found any entry, or found multiple candidates. "
             "set DSCUDA_WARNLEVEL 4 or higher and try again to see "
             "error messages from %s.\n", mangler, mangler);
        exit(1);
    }
    WARN(10, "--->Exiting %s()\n", __func__);
}


static pthread_mutex_t LoadModuleMutex = PTHREAD_MUTEX_INITIALIZER;
/*
 * Load a cuda module from a .ptx file, and then, send it to the server.
 * returns id for the module.
 * the module is cached and sent only once for a certain period.
 */

int
dscudaLoadModule(char *name, char *strdata) { // 'strdata' must be NULL terminated.
    WARN(5, "dscudaLoadModule( name=%p(%s), strdata=%p ) {\n", name, name, strdata);
    int idx = vdevidIndex();
    VirDev *vdev = St.Vdev + Vdevid[idx];
    int module_index;

    module_index = vdev->loadModule(name, strdata);
    
    //printModuleList();
    WARN(5, "} //dscudaLoadModule() returned %d.\n", module_index);
    WARN(5, "\n");
    return module_index;
}

#if 0 // backup
int*
dscudaLoadModule(char *name, char *strdata) {// 'strdata' must be NULL terminated.
    int i, j, mid;
    ClientModule *mp;
    int idx;

    if (name != NULL) {
	WARN(5, "dscudaLoadModule(%p) modulename:%s  ...\n", name, name);
#if RC_CACHE_MODULE
	// look for modulename in the module list.
	for (i=0, mp=CltModulelist; i < RC_NKMODULEMAX; i++, mp++) {
	    if ( mp->isInvalid() ) {
		continue;
	    }
	    
	    idx = vdevidIndex();
	    if (mp->vdevid != Vdevid[idx]) {
		continue;
	    }
	    if ( !strcmp(name, mp->name) ) {
		if ( mp->isAlive() ) {
		    WARN(5, "done. found a cached one. id:%d  age:%d  name:%s\n",
			 mp->id[i], time(NULL) - mp->sent_time, mp->name);
		    return mp->id; // module found. i.e, it's already loaded.
		} else {
		    WARN(5, "found a cached one with id:%d, but it is too old (age:%d). resend it.\n",
			 mp->id[i], time(NULL) - mp->sent_time);
		    mp->invalidate(); // invalidate the cache.
		}
	    }
	} //for
#endif // RC_CACHE_MODULE
    } else {
	WARN(5, "dscudaLoadModule(%p) modulename:-\n", name);
    }

    //<---
    char *strdata_found = NULL;
    char *name_found=NULL;
    if (name==NULL && strdata==NULL) {
        for (i=0, mp=CltModulelist; i<RC_NKMODULEMAX; i++, mp++) {
	    WARN(10, "i=%d\n", i);
	    if (mp->isInvalid()) continue;
	    idx = vdevidIndex();
	    if (mp->vdevid != Vdevid[idx]) continue;
	    if (!strcmp(name, mp->name)) {     /* matched */
		strdata_found = mp->ptx_image;
		name_found = mp->name;
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
    idx = vdevidIndex();
    VirDev *vdev = St.Vdev + Vdevid[idx];
    PhyDev *sp = vdev->server;

    for (i=0; i < vdev->nredundancy; i++) {
	// mid = dscudaLoadModuleLocal(St.getIpAddress(), getpid(), name, strdata, Vdevid[vi], i);
	// mid = dscudaLoadModuleLocal(St.getIpAddress(), getpid(), name_found, strdata_found, Vdevid[idx], i);
	mid = sp[i].loadModule(St.getIpAddress(), getpid(), name_found, strdata_found);
        WARN(3, "(info) dscudaLoadModuleLocal() returns mid=%d as Vdevid[%d], Redun[%d].\n", mid, idx, i);

        // register a new module into the list,
        // and then, return a module id assigned by the server.
        if (i==0) {
            for (j=0; j<RC_NKMODULEMAX; j++) { /* Search vacant sheet. */
                if( CltModulelist[j].isInvalid() ) break;
            }
	    if( j >= RC_NKMODULEMAX ) {
		WARN(0, "\n\n### (+_+) ERROR in DS-CUDA!\n");
		WARN(0,     "### (+_+) module send buffer is full. and exit.\n");
		WARN(0,     "### (+_+) Check if the array length of CltModulelist[%d].\n\n\n", j);
		exit(1);
	    }
            CltModulelist[j].validate();
            CltModulelist[j].sent_time = time(NULL);
//            CltModulelist[j].setPtxPath(name_found);
//	    CltModulelist[j].setPtxImage(strdata_found);
	    CltModulelist[j].linkPtxData(name_found, strdata_found, &PtxStore);
	    
            WARN(5, "New client module item was registered. id:%d\n", mid);
        }
        CltModulelist[j].id[i] = mid;
    }
    CltModulelist[j].vdevid = Vdevid[idx];
    printModuleList();

    return CltModulelist[j].id; //mp->id;
}
#endif

cudaError_t
dscudaFuncGetAttributesWrapper(int *moduleid, struct cudaFuncAttributes *attr, const char *func)
{
    cudaError_t err = cudaSuccess;
    dscudaFuncGetAttributesResult *rp;

    WARN(3, "dscudaFuncGetAttributesWrapper(%d, 0x%08llx, %s)...",
         moduleid, (unsigned long long)attr, func);
    VirDev *vdev = St.Vdev + Vdevid[vdevidIndex()];
    PhyDev *sp = vdev->server;
    
    for (int i = 0; i < vdev->nredundancy; i++) {
        if (St.isIbv()) {
#warning fill this part in dscudaFuncGetAttributesWrapper().
        } else {
            rp = dscudafuncgetattributesid_1(moduleid[i], (char*)func, sp[i].Clnt);
            checkResult(rp, sp[i]);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            if (i == 0) {
                attr->binaryVersion      = rp->attr.binaryVersion;
                attr->constSizeBytes     = rp->attr.constSizeBytes;
                attr->localSizeBytes     = rp->attr.localSizeBytes;
                attr->maxThreadsPerBlock = rp->attr.maxThreadsPerBlock;
                attr->numRegs            = rp->attr.numRegs;
                attr->ptxVersion         = rp->attr.ptxVersion;
                attr->sharedSizeBytes    = rp->attr.sharedSizeBytes;
            }
            xdr_free((xdrproc_t)xdr_dscudaFuncGetAttributesResult, (char *)rp);
        }
    }

    WARN(3, "done.\n");
    WARN(3, "  attr->binaryVersion: %d\n", attr->binaryVersion);
    WARN(3, "  attr->constSizeBytes: %zu\n", attr->constSizeBytes);
    WARN(3, "  attr->localSizeBytes: %zu\n", attr->localSizeBytes);
    WARN(3, "  attr->maxThreadsPerBlock: %d\n", attr->maxThreadsPerBlock);
    WARN(3, "  attr->numRegs: %d\n", attr->numRegs);
    WARN(3, "  attr->ptxVersion: %d\n", attr->ptxVersion);
    WARN(3, "  attr->sharedSizeBytes: %zu\n", attr->sharedSizeBytes);

    return err;
}

cudaError_t
dscudaMemcpyToSymbolWrapper(int *moduleid, const char *symbol, const void *src,
                           size_t count, size_t offset, enum cudaMemcpyKind kind) {
    cudaError_t err = cudaSuccess;
    int nredundancy;

    WARN(3, "dscudaMemcpyToSymbolWrapper(%p, 0x%08lx, 0x%08lx, %zu, %zu, %s)"
         "symbol:%s  ...",
         moduleid, (unsigned long)symbol, (unsigned long)src,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    nredundancy = (St.Vdev + Vdevid[vdevidIndex()])->nredundancy;
    switch (kind) {
      case cudaMemcpyHostToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolH2D(moduleid[i], (char *)symbol, src, count, offset, Vdevid[vdevidIndex()], i);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolD2D(moduleid[i], (char *)symbol, src, count, offset, Vdevid[vdevidIndex()], i);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    if ((St.Vdev + Vdevid[vdevidIndex()])->isRecording() &&
	(kind==cudaMemcpyHostToDevice || kind==cudaMemcpyDeviceToDevice)) {
        CudaMemcpyToSymbolArgs args;
        args.moduleid = moduleid;
        args.symbol = (char *)symbol;
        args.src = (void *)src;
        args.count = count;
        args.offset = offset;
        args.kind = kind;
        //HISTREC.add(dscudaMemcpyToSymbolH2DId, (void *)&args);
    }

    return err;
}//dscudaMemcpyToSymbolWrapper(int *moduleid, const char *symbol, const void *src,

cudaError_t
dscudaMemcpyFromSymbolWrapper(int *moduleid, void *dst, const char *symbol,
                             size_t count, size_t offset,
                             enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    int nredundancy;
    void *dstbuf;

    WARN(3, "dscudaMemcpyFromSymbolWrapper(%p, %p, %p, %zu, %zu, %s)"
         "symbol:%s  ...",
         moduleid, dst, symbol, count, offset, dscudaMemcpyKindName(kind), symbol);

    nredundancy = (St.Vdev + Vdevid[vdevidIndex()])->nredundancy;
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        if (St.isIbv()) {
            dstbuf = calloc(1, count);
            if (!dstbuf) {
                WARN(0, "dscudaMemcpyFromSymbolWrapper:calloc() failed.\n");
                exit(1);
            }
        }

        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolD2H(moduleid[i], &dstbuf, (char *)symbol, count, offset, Vdevid[vdevidIndex()], i);
            if (i == 0) {
                memcpy(dst, dstbuf, count);
            } else if (bcmp(dst, dstbuf, count) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            } else {
                WARN(3, "cudaMemcpyFromSymbol() data copied from device%d matched with that from device0.\n", i);
            }
        }
        if (St.isIbv()) {
            free(dstbuf);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolD2D(moduleid[i], dst, (char *)symbol, count, offset, Vdevid[vdevidIndex()], i);
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
dscudaMemcpyToSymbolAsyncWrapper(int *moduleid, const char *symbol, const void *src,
                                 size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    RCstreamArray *st;
    int nredundancy;

    WARN(3, "sym:%s\n", symbol);
    WARN(3, "dscudaMemcpyToSymbolAsyncWrapper(%p, 0x%08lx, 0x%08lx, %zu, %zu, %s, 0x%08lx) "
         "symbol:%s  ...",
         moduleid, (unsigned long)symbol, (unsigned long)src,
         count, offset, dscudaMemcpyKindName(kind), (unsigned long)stream, symbol);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : %p\n", stream);
        exit(1);
    }
    nredundancy = (St.Vdev + Vdevid[vdevidIndex()])->nredundancy;
    switch (kind) {
      case cudaMemcpyHostToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolAsyncH2D(moduleid[i], (char *)symbol, src, count, offset,
                                               (RCstream)st->s[i], Vdevid[vdevidIndex()], i);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyToSymbolAsyncD2D(moduleid[i], (char *)symbol, src, count, offset,
                                               (RCstream)st->s[i], Vdevid[vdevidIndex()], i);
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
dscudaMemcpyFromSymbolAsyncWrapper(int *moduleid, void *dst, const char *symbol,
                                   size_t count, size_t offset, enum cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;
    RCstreamArray *st;
    int nredundancy;
    void *dstbuf;

    WARN(3, "dscudaMemcpyFromSymbolAsyncWrapper(%d, 0x%08lx, 0x%08lx, %zu, %zu, %s, 0x%08lx)"
         " symbol:%s  ...",
         moduleid, (unsigned long)dst, (unsigned long)symbol,
         count, offset, dscudaMemcpyKindName(kind), (unsigned long)stream, symbol);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : %p\n", stream);
        exit(1);
    }
    nredundancy = (St.Vdev + Vdevid[vdevidIndex()])->nredundancy;
    switch (kind) {
      case cudaMemcpyDeviceToHost:
        if (St.isIbv()) {
            dstbuf = calloc(1, count);
            if (!dstbuf) {
                WARN(0, "dscudaMemcpyFromSymbolAsyncWrapper:calloc() failed.\n");
                exit(1);
            }
        }
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolAsyncD2H(moduleid[i], &dstbuf, (char *)symbol, count, offset,
                                                 (RCstream)st->s[i], Vdevid[vdevidIndex()], i);
            if (i == 0) {
                memcpy(dst, dstbuf, count);
            } else if (bcmp(dst, dstbuf, count) != 0) {
                if (errorHandler) {
                    errorHandler(errorHandlerArg);
                }
            } else {
                WARN(3, "cudaMemcpyFromSymbol() data copied from device%d matched with that from device0.\n", i);
            }
        }
        if (St.isIbv()) {
            free(dstbuf);
        }
        break;
      case cudaMemcpyDeviceToDevice:
        for (int i = 0; i < nredundancy; i++) {
            err = dscudaMemcpyFromSymbolAsyncD2D(moduleid[i], dst, (char *)symbol, count, offset,
                                                 (RCstream)st->s[i], Vdevid[vdevidIndex()], i);
        }
        break;
      default:
        WARN(0, "Unsupported value for cudaMemcpyKind : %s\n", dscudaMemcpyKindName(kind));
        exit(1);
    }
    WARN(3, "done.\n");

    return err;
}

static void
setTextureParams(RCtexture *texbufp, const struct textureReference *tex, const struct cudaChannelFormatDesc *desc)
{
    texbufp->normalized = tex->normalized;
    texbufp->filterMode = tex->filterMode;
    texbufp->addressMode[0] = tex->addressMode[0];
    texbufp->addressMode[1] = tex->addressMode[1];
    texbufp->addressMode[2] = tex->addressMode[2];
    if (desc) {
        texbufp->x = desc->x;
        texbufp->y = desc->y;
        texbufp->z = desc->z;
        texbufp->w = desc->w;
        texbufp->f = desc->f;
    } else {
        texbufp->x = tex->channelDesc.x;
        texbufp->y = tex->channelDesc.y;
        texbufp->z = tex->channelDesc.z;
        texbufp->w = tex->channelDesc.w;
        texbufp->f = tex->channelDesc.f;
    }
}

cudaError_t
dscudaBindTextureWrapper(int *moduleid, char *texname,
                        size_t *offset,
                        const struct textureReference *tex,
                        const void *devPtr,
                        const struct cudaChannelFormatDesc *desc,
                        size_t size)
{
    cudaError_t err = cudaSuccess;
    dscudaBindTextureResult *rp;
    RCtexture texbuf;

    WARN(3, "dscudaBindTextureWrapper(%p, %s, %p, %p, %p, %p, %zu)...",
         moduleid, texname,
         offset, tex, devPtr, desc, size);

    setTextureParams(&texbuf, tex, desc);

    VirDev *vdev = St.Vdev + Vdevid[vdevidIndex()];
    PhyDev *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        if (St.isIbv()) {

#warning fill this part in dscudaBindTextureWrapper().
        }
        else {
            rp = dscudabindtextureid_1(moduleid[i], texname,
                                       (RCadr)devPtr, size, (RCtexture)texbuf, sp[i].Clnt);
            checkResult(rp, sp[i]);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            if (i == 0) {
                if (offset) {
                    *offset = rp->offset;
                }
            }
            xdr_free((xdrproc_t)xdr_dscudaBindTextureResult, (char *)rp);
        }
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
dscudaBindTexture2DWrapper(int *moduleid, char *texname,
                          size_t *offset,
                          const struct textureReference *tex,
                          const void *devPtr,
                          const struct cudaChannelFormatDesc *desc,
                          size_t width, size_t height, size_t pitch)
{
    cudaError_t err = cudaSuccess;
    dscudaBindTexture2DResult *rp;
    RCtexture texbuf;

    WARN(3, "dscudaBindTexture2DWrapper(%p, %s, %p, %p, %p, %p, %zu, %zu, %zu)...",
         moduleid, texname,
         offset, tex, devPtr, desc, width, height, pitch);

    setTextureParams(&texbuf, tex, desc);

    VirDev *vdev = St.Vdev + Vdevid[vdevidIndex()];
    PhyDev *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        if (St.isIbv()) {

#warning fill this part in dscudaBindTexture2DWrapper().
        } else {

            rp = dscudabindtexture2did_1(moduleid[i], texname,
                                         (RCadr)devPtr, width, height, pitch, (RCtexture)texbuf, sp[i].Clnt);
            checkResult(rp, sp[i]);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            if (i == 0) {
                if (offset) {
                    *offset = rp->offset;
                }
            }
            xdr_free((xdrproc_t)xdr_dscudaBindTexture2DResult, (char *)rp);
        }
    }

    WARN(3, "done.\n");
    return err;
}

cudaError_t
dscudaBindTextureToArrayWrapper(int *moduleid, char *texname,
                               const struct textureReference *tex,
                               const struct cudaArray *array,
                               const struct cudaChannelFormatDesc *desc) {
    cudaError_t err = cudaSuccess;
    dscudaResult *rp;
    RCtexture texbuf;
    RCcuarrayArray *ca;

    WARN(3, "dscudaBindTextureToArrayWrapper(%p, %s, %p, %p)...", moduleid, texname, array, desc);

    setTextureParams(&texbuf, tex, desc);

    ca = RCcuarrayArrayQuery((cudaArray *)array);
    if (!ca) {
        WARN(0, "invalid cudaArray : %p\n", array);
        exit(1);
    }

    VirDev *vdev = St.Vdev + Vdevid[vdevidIndex()];
    PhyDev *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++) {
        if (St.isIbv()) {

#warning fill this part in dscudaBindTextureToArrayWrapper().
        } else {

            rp = dscudabindtexturetoarrayid_1(moduleid[i], texname, (RCadr)ca->ap[i], (RCtexture)texbuf, sp[i].Clnt);
            checkResult(rp, sp[i]);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
        }
    }
    WARN(3, "done.\n");
    return err;
}

cudaError_t
cudaGetDevice(int *device) {
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaGetDevice(%p)...", device);
    *device = Vdevid[vdevidIndex()];
    WARN(3, "done.\n");

    return err;
}
cudaError_t
cudaSetDevice_clnt(int device, int errcheck) {
    cudaError_t cuerr = cudaSuccess;
    int         vi    = vdevidIndex();
    
    if (0 <= device && device < St.Nvdev ) {
        Vdevid[vi] = device;
    }
    else {
        cuerr = cudaErrorInvalidDevice;
	if (errcheck != 0) {
	    fprintf( stderr, "%s(): failed.\n", __func__);
	    exit(1);
	}
    }
    return cuerr;
}

cudaError_t
cudaSetDevice(int device) {
    cudaError_t cuerr    = cudaSuccess;
    int         errcheck = 0; 
    WARN(3, "%s(%d) {\n", __func__, device);
    
#if 0
    // cudaSetDevice() is not needed on CUDA called record.
    // active target device is recorded on each cuda*() funcitons.
    if (HISTREC.rec_en > 0) {
        CudaSetDeviceArgs args;
        args.device = device;
        //HISTREC.add(dscudaSetDeviceId, (void *)&args);
    }
#endif
    
    cuerr = cudaSetDevice_clnt( device, errcheck );
    WARN(3, "}\n");
    return cuerr;
}

cudaError_t
cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaChooseDevice(%p, %p)...", device, prop);
    *device = 0;
    WARN(3, "done.\n");
    WARN(3, "Note : The current implementation always returns device 0.\n");

    return err;
}

cudaError_t
cudaGetDeviceCount(int *count) {
    cudaError_t err = cudaSuccess;

    *count = St.Nvdev;
    WARN(3, "cudaGetDeviceCount(%p)  count:%d ...", count, *count);
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice) {
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceCanAccessPeer(%p, %d, %d)...", canAccessPeer, device, peerDevice);
    if (device < 0 || St.Nvdev <= device) {
        err = cudaErrorInvalidDevice;
    }
    if (peerDevice < 0 || St.Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceEnablePeer(%d, %d)...", peerDevice, flags);
    if (peerDevice < 0 || St.Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t
cudaDeviceDisablePeerAccess(int peerDevice) {
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceDisablePeer(%d)...", peerDevice);
    if (peerDevice < 0 || St.Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}

/*
 * MEMO: BkupMemList_t::reallocDeviceRegion(PhyDev *svr)
 */
void
VirDev::remallocRegionsGPU(int num_svr) {
    BkupMem *mem = memlist.headPtr();
    //int     verb = St.isAutoVerb();
    int     copy_count = 0;
    int     i = 0;
    
    WARN(1, "%s(PhyDev *sp).\n", __func__);
    //WARN(1, "Num. of realloc region = %d\n", BKUPMEM.length );
    //St.unsetAutoVerb();
    while ( mem != NULL ) {
	/* TODO: select migrateded virtual device, not all region. */
	WARN(5, "mem[%d]->dst = %p, size= %d\n", i, mem->d_region, mem->size);
	//dscudaVerbMalloc(&mem->d_region, mem->size, svr);
	mem = mem->next;
	i++;
    }
    //St.setAutoVerb(verb);
    WARN(1, "+--- Done.\n");
}
