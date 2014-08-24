//                             -*- Mode: C++ -*-
// Filename         : libdscuda.cu
// Description      : DS-CUDA client node library for Remote Procedure Call.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-08-24 18:15:38
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
/*
 * This file is included into the bottom of ...
 *     -> "libdscuda_ibv.cu"
 *     -> "libdscuda_rpc.cu"
 */
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
#include "dscuda.h"
#include "libdscuda.h"
#include "dscudaverb.h"

ClientModule_t::ClientModule_t(void) {
    WARN( 5, "The constructor %s() called.\n", __func__ );
    valid  = -1;
    vdevid = -1;
    for (int i=0; i<RC_NREDUNDANCYMAX; i++) id[i] = -1;
    strncpy(name, "init", RC_KMODULENAMELEN);
    strncpy(ptx_image, "init", RC_KMODULEIMAGELEN);
}

static int   VdevidIndexMax = 0;            // # of pthreads which utilize virtual devices.
const  char *DEFAULT_SVRIP = "localhost";
static char Dscudapath[512];

static pthread_mutex_t VdevidMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t       VdevidIndex2ptid[RC_NPTHREADMAX]; // convert an Vdevid index into pthread id.
// CheckPointing mutual exclusion
       pthread_mutex_t cudaMemcpyD2H_mutex = PTHREAD_MUTEX_INITIALIZER;
       pthread_mutex_t cudaMemcpyH2D_mutex = PTHREAD_MUTEX_INITIALIZER;
       pthread_mutex_t cudaKernelRun_mutex = PTHREAD_MUTEX_INITIALIZER;

       RCmappedMem    *RCmappedMemListTop     = NULL;
       RCmappedMem    *RCmappedMemListTail    = NULL;

static RCstreamArray  *RCstreamArrayListTop   = NULL;
static RCstreamArray  *RCstreamArrayListTail  = NULL;

static RCeventArray   *RCeventArrayListTop    = NULL;
static RCeventArray   *RCeventArrayListTail   = NULL;

static RCcuarrayArray *RCcuarrayArrayListTop  = NULL;
static RCcuarrayArray *RCcuarrayArrayListTail = NULL;

static RCuva          *RCuvaListTop           = NULL;
static RCuva          *RCuvaListTail          = NULL;

int    Vdevid[RC_NPTHREADMAX] = {0};   // the virtual device currently in use.

/*
 * Physical GPU device server
 */
SvrList_t SvrCand;
SvrList_t SvrSpare;
SvrList_t SvrBroken;

void (*errorHandler)(void *arg) = NULL;
void *errorHandlerArg = NULL;
CLIENT *Clnt[RC_NVDEVMAX][RC_NREDUNDANCYMAX]; /* RPC clients */
struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX];

ClientModule CltModulelist[RC_NKMODULEMAX]; /* is Singleton.*/

struct ClientState_t St; // is Singleton

char *ClientState_t::getFtModeString( void ) {
    static char s[80];
    switch ( ft_mode ) {
    case FT_PLAIN: strncpy( s, "FT_PLAIN", 70);	break;
    case FT_REDUN: strncpy( s, "FT_REDUN", 70); break;
    case FT_MIGRA: strncpy( s, "FT_MIGRA", 70); break;
    case FT_BOTH:  strncpy( s, "FT_BOTH",  70); break;
    default:
	WARN( 0, "%s(): confused.\n", __func__ );
	exit( EXIT_FAILURE );
    }
    return s;
}

void dscudaRecordHistOn(void) {
    HISTREC.rec_en = 1;
}
void dscudaRecordHistOff(void) {
    HISTREC.rec_en = 0;
}

void dscudaAutoVerbOn(void) {
    St.setAutoVerb();
}
void dscudaAutoVerbOff(void) {
    St.unsetAutoVerb();
}

int requestDaemonForDevice(char *ipaddr,  // ex.)"192.168.0.101"
			   int devid, int useibv) {
    int dsock; // socket for side-band communication with the daemon & server.
    int sport; // port number of the server. given by the daemon.
    char msg[256];
    struct sockaddr_in sockaddr;

    sockaddr = setupSockaddr( ipaddr, RC_DAEMON_IP_PORT );
    dsock = socket(AF_INET, SOCK_STREAM, 0);
    if (dsock < 0) {
        perror("socket");
        exit(1);
    }
    
    if ( connect(dsock, (struct sockaddr *)&sockaddr, sizeof(sockaddr)) == -1 ) {
        perror("(;_;) Connect");
	WARN(0, "+--- Program terminated at %s:L%d\n", __FILE__, __LINE__ );
	WARN(0, "+--- Maybe DS-CUDA daemon is not running...\n" );
        exit(1);
    }
    sprintf(msg, "deviceid:%d", devid);
    sendMsgBySocket(dsock, msg);

    memset(msg, 0, strlen(msg));
    recvMsgBySocket(dsock, msg, sizeof(msg));
    sscanf(msg, "sport:%d", &sport);

    if (sport < 0) {
        WARN(0, "max possible ports on %s already in use.\n", ipaddr);
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
int vdevidIndex(void) {
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

void RCmappedMemRegister(void *pHost, void* pDevice, size_t size) {
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

RCmappedMem* RCmappedMemQuery(void *pHost) {
    RCmappedMem *mem = RCmappedMemListTop;
    while (mem) {
        if (mem->pHost == pHost) {
            return mem;
        }
        mem = mem->next;
    }
    return NULL; // pHost not found in the list.
}

void RCmappedMemUnregister(void *pHost) {
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

/*
 * Register a stream array. each component is associated to a stream
 * on each Server[]. User see only the 1st element, streams[0].
 * Others, i.e., streams[1..Nredunddancy-1], are used by this library
 * to handle redundant calculation mechanism.
 */
static
void RCstreamArrayRegister(cudaStream_t *streams) {
    RCstreamArray *st = (RCstreamArray *)malloc(sizeof(RCstreamArray));
    if (!st) {
        perror("RCstreamArrayRegister");
    }
    for (int i = 0; i < RC_NREDUNDANCYMAX; i++) {
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
static
void showsta(void) {
    RCstreamArray *st = RCstreamArrayListTop;
    while (st) {
        fprintf(stderr, ">>> 0x%08llx    prev:%p  next:%p\n", st, st->prev, st->next);
        st = st->next;
    }
}
#endif

RCstreamArray* RCstreamArrayQuery(cudaStream_t stream0) {
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

static
void RCstreamArrayUnregister(cudaStream_t stream0) {
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



/*
 * Register a cudaArray array. each component is associated to a cudaArray
 * on each Server[]. User see only the 1st element, cuarrays[0].
 * Others, i.e., cuarrays[1..Nredunddancy-1], are used by this library
 * to handle redundant calculation mechanism.
 */
void RCcuarrayArrayRegister(cudaArray **cuarrays)
{
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

RCcuarrayArray* RCcuarrayArrayQuery(cudaArray *cuarray0)
{
    RCcuarrayArray *ca = RCcuarrayArrayListTop;
    while (ca) {
        if (ca->ap[0] == cuarray0) {
            return ca;
        }
        ca = ca->next;
    }
    return NULL;
}

void RCcuarrayArrayUnregister(cudaArray *cuarray0)
{
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
void RCeventArrayRegister(cudaEvent_t *events)
{
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

RCeventArray* RCeventArrayQuery(cudaEvent_t event0)
{
    RCeventArray *ev = RCeventArrayListTop;
    while (ev) {
        if (ev->e[0] == event0) {
            return ev;
        }
        ev = ev->next;
    }
    return NULL;
}

void RCeventArrayUnregister(cudaEvent_t event0)
{
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
void* dscudaUvaOfAdr( void *adr, int devid ) {
    DscudaUva_t adri = (DscudaUva_t)adr;
#if __LP64__
    adri |= ((DscudaUva_t)devid << 48);
#endif
    return (void *)adri;
}
/*====================================================================
 * Get GPU deviceID from UVA.
 */
int dscudaDevidOfUva( void *adr ) {
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
void *dscudaAdrOfUva( void *adr ) {
    DscudaUva_t adri = (DscudaUva_t)adr;
#if __LP64__
    adri &= 0x0000ffffffffffffLL;
#endif
    return (void *)adri;
}

/*
 * Register an array of UVA. each component is associated to an UVA
 * on each Server[i]. User see only the 1st element, [0].
 * Others, i.e., Server[1..Nredunddancy-1], are hidden to the user,
 * and used by this library to handle redundant calculation mechanism.
 */
void RCuvaRegister(int devid, void *adr[], size_t size) {
    int i;
    int nredundancy = dscudaNredundancy();
    RCuva *uva = (RCuva *)malloc(sizeof(RCuva));

    if (!uva) {
        perror("RCuvaRegister");
    }
    for (i = 0; i < nredundancy; i++) {
        uva->adr[i] = adr[i];
    }
    uva->devid = devid;
    uva->size = size;
    uva->prev = RCuvaListTail;
    uva->next = NULL;
    if (!RCuvaListTop) { // uva will be the 1st entry.
        RCuvaListTop = uva;
    } else {
        RCuvaListTail->next = uva;
    }
    RCuvaListTail = uva;
}

RCuva *
RCuvaQuery(void *adr)
{
    RCuva *uva = RCuvaListTop;
    unsigned long ladr = (unsigned long)dscudaAdrOfUva(adr);
    int devid = dscudaDevidOfUva(adr);

    //    fprintf(stderr, ">>>> adr:0x%016llx  ladr:0x%016llx  devid:%d\n", adr, ladr, devid);

    while (uva) {
        if ((unsigned long)uva->adr[0] <= ladr &&
            ladr < (unsigned long)uva->adr[0] + uva->size &&
            uva->devid == devid) {
            return uva;
        }
        uva = uva->next;
    }
    return NULL; // uva not found in the list.
}

void
RCuvaUnregister(void *adr)
{
    RCuva *uva = RCuvaQuery(adr);

    if (!uva) return;

    if (uva->prev) { // reconnect the linked list.
        uva->prev->next = uva->next;
    } else { // uva was the 1st entry.
        RCuvaListTop = uva->next;
        if (uva->next) {
            uva->next->prev = NULL;
        }
    }
    if (!uva->next) { // uva was the last entry.
        RCuvaListTail = uva->prev;
    }
    free(uva);
}

static char*
readServerConf(char *fname)
{
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

static void resetServerUniqID(void) {
    for (int i=0; i<RC_NVDEVMAX; i++) { /* Vdev[*] */
	for (int j=0; j<RC_NREDUNDANCYMAX; j++) {
	    St.Vdev[i].server[j].uniq = RC_UNIQ_INVALID;
	}
    }
    for (int j=0; j<RC_NVDEVMAX; j++) { /* svrCand[*] */
	SvrCand.svr[j].uniq = RC_UNIQ_INVALID;
    }
    for (int j=0; j<RC_NVDEVMAX; j++) { /* svrSpare[*] */
	SvrSpare.svr[j].uniq = RC_UNIQ_INVALID;
    }
}

/*
 *
 */
void printVirtualDeviceList( void ) {
    Vdev_t     *pVdev;
    RCServer_t *pSvr;
    int         i,j;
    
    WARN( 0, "/*** Virtual Device Information. (Nvdev=%d) ***/\n", St.Nvdev);
    for ( i=0, pVdev=St.Vdev; i<St.Nvdev; i++, pVdev++ ) {
	if ( i >= RC_NVDEVMAX ) {
	    WARN(0, "(;_;) Too many virtual devices. %s().\nexit.", __func__);
	    exit( EXIT_FAILURE );
	}
	
	if ( pVdev->nredundancy == 1 ) {
	    WARN( 0, "Virtual[%d] (MONO)\n", i );
	} else if ( pVdev->nredundancy > 1 ) {
	    WARN( 0, "Virtual[%d] (POLY:%d)\n", i, pVdev->nredundancy );
	} else {
	    WARN( 0, "Virtual[%d] (????:%d)\n", i, pVdev->nredundancy );
	}
	
	for (j=0, pSvr=pVdev->server; j<pVdev->nredundancy; j++, pSvr++) {
	    if (j >= RC_NREDUNDANCYMAX) {
		WARN(0, "(;_;) Too many redundant devices %d. %s().\nexit.\n", __func__);
		exit( EXIT_FAILURE );
	    }
	    WARN( 0, "    + Physical[%d]: id=%d, cid=%d, IP=%s(%s), uniq=%d.\n", j,
		   pSvr->id, pSvr->cid, pSvr->ip, pSvr->hostname, pSvr->uniq);
	}
    }

    if ( St.ft_mode==FT_MIGRA || St.ft_mode==FT_BOTH ) {
	/*
	 * Device Candidates
	 */
	WARN( 0, "*** Physical Device Candidates. (Ncand=%d)\n", SvrCand.num );
	for( i=0, pSvr=SvrCand.svr; i < SvrCand.num; i++, pSvr++ ){
	    if (i >= RC_NVDEVMAX) {
		WARN(0, "(;_;) Too many candidate devices. %s().\nexit.", __func__);
		exit( EXIT_FAILURE );
	    }
	    WARN( 0, "    - Cand[%2d]: id=%d, cid=%d, IP=%s, uniq=%d.\n", i,
		  pSvr->id, pSvr->cid, pSvr->ip, pSvr->uniq);
	}
	/*
	 * Alternate Devices
	 */
	WARN( 0, " *** Spare Server Info.(Nspare=%d)\n", SvrSpare.num);
	for( i=0, pSvr=SvrSpare.svr; i < SvrSpare.num; i++, pSvr++ ){
	    if (i >= RC_NVDEVMAX) {
		WARN(0, "(;_;) Too many spare devices. %s().\nexit.", __func__);
		exit( EXIT_FAILURE );
	    }
	    WARN( 0, "    - Spare[%d]: id=%d, cid=%d, IP=%s, uniq=%d.\n", i,
		  pSvr->id, pSvr->cid, pSvr->ip, pSvr->uniq);
	}
    }
    return;
}

void printModuleList(void) {
    const int len = 256;
    char printbuf[len];
    int valid_cnt = 0;
    
    WARN( 5, "%s(): ====================================================\n", __func__);
    WARN( 5, "%s(): RC_NKMODULEMAX= %d\n", __func__, RC_NKMODULEMAX);
    
    for (int i=0; i<RC_NKMODULEMAX; i++) {
	if( CltModulelist[i].isValid() ){
	    WARN( 5, "%s(): CltModulelist[%d]:\n", __func__, i);
	    WARN( 5, "%s():    + vdevid= %d\n", __func__, CltModulelist[i].vdevid);
	    for (int j=0; j<4; j++) {
		WARN( 5, "%s():    + id[%d]= %d\n", __func__, j, CltModulelist[i].id[j]);
	    }
	    WARN( 5, "%s():    + name= %s\n", __func__, CltModulelist[i].name);
	    //printf("    + send_time= \n", sent_time., sent_time.);
	    strncpy(printbuf, CltModulelist[i].ptx_image, len - 1 );
	    printbuf[255]='\0';
	    //printf("# %s():    + ptx_image=\n%s\n", __func__, printbuf);
	    valid_cnt++;
	}
    }
    WARN( 5, "%s(): %d valid modules registered.\n", __func__, valid_cnt);
    WARN( 5, "%s(): ====================================================\n", __func__);
}

static
int dscudaSearchDaemon(void) {
    int sendsock;
    int recvsock;

    char sendbuf[SEARCH_BUFLEN_TX];
    char recvbuf[SEARCH_BUFLEN_RX];
    
    int recvlen;
    int num_daemon = 0;
    int num_device = 0;
    int num_ignore = 0;

    int val = 1;
    unsigned int adr, mask;
    socklen_t sin_size;
    
    int setsockopt_ret;
    int bind_ret;
    int close_ret;

    struct sockaddr_in addr, svr;
    struct ifreq ifr[2];
    struct ifconf ifc;
    struct passwd *pwd;
    
    WARN(2, "RC_DAEMON_IP_PORT = %d\n", RC_DAEMON_IP_PORT);
    sendsock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    recvsock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if ( sendsock == -1 || recvsock == -1 ) {
	perror("dscudaSearchDaemon: socket()");
	//return -1;
	exit(1);
    }
    
    setsockopt_ret = setsockopt(sendsock, SOL_SOCKET, SO_BROADCAST, &val, sizeof(val));
    if ( setsockopt_ret != 0 ) {
	perror("dscudaSearchDaemon: setsockopt()");
	exit(1);
    }

    ifc.ifc_len = sizeof(ifr) * 2;
    ifc.ifc_ifcu.ifcu_buf = (char *)ifr;
    ioctl(sendsock, SIOCGIFCONF, &ifc);

    ifr[1].ifr_addr.sa_family = AF_INET;
    ioctl(sendsock, SIOCGIFADDR, &ifr[1]);
    adr = ((struct sockaddr_in *)(&ifr[1].ifr_addr))->sin_addr.s_addr;
    ioctl(sendsock, SIOCGIFNETMASK, &ifr[1]);
    mask = ((struct sockaddr_in *)(&ifr[1].ifr_netmask))->sin_addr.s_addr;

    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(RC_DAEMON_IP_PORT - 1);
    addr.sin_addr.s_addr = adr | ~mask;

    strncpy( sendbuf, SEARCH_PING, SEARCH_BUFLEN_TX - 1 );
    sendto( sendsock, sendbuf, SEARCH_BUFLEN_TX, 0, (struct sockaddr *)&addr, sizeof(addr));
    WARN(2, "Broadcast message \"%s\"...\n", SEARCH_PING);
    sin_size = sizeof(struct sockaddr_in);

    svr.sin_family      = AF_INET;
    svr.sin_port        = htons(RC_DAEMON_IP_PORT - 2);
    svr.sin_addr.s_addr = htonl(INADDR_ANY);
    
    // Set timeout for recvsock.
    struct timeval tout;
    tout.tv_sec  = RC_SEARCH_DAEMON_TIMEOUT ;
    tout.tv_usec = 0;
    setsockopt_ret = setsockopt(recvsock, SOL_SOCKET, SO_RCVTIMEO, (char *)&tout, sizeof(tout));
    if ( setsockopt_ret != 0 ) {
	perror("dscudaSearchDaemon: setsockopt(recvsock)");
	exit(1);
    }
    
    bind_ret = bind( recvsock, (struct sockaddr *)&svr, sizeof(svr) );
    if( bind_ret != 0 ) {
	fprintf(stderr, "Error: bind() returned %d. recvsock=%d, port=%d\n",
		bind_ret, recvsock, svr.sin_port); //port:38655
	perror("dscudaSearchDaemon: bind()");
	return -1;
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
	WARN(2, "    + Recieved ACK \"%s\" ", recvbuf);
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
	    WARN0(2, "from server \"%s\" ", ipaddr );
	    if ( strcmp( magic_word, SEARCH_ACK   )==0 &&
		 strcmp( user_name,  pwd->pw_name )==0 ) { /* Found */
		WARN0(2, "valid.\n");
		/*
		 * Updata SvrCand;
		 */
		for (int d=0; d<num_eachdev; d++) {
		    SvrCand.cat( ipaddr, d, host_name );
		}
		num_daemon += 1;
		num_device += num_eachdev;
	    } else {
		WARN0(2, "ignored.\n");
		num_ignore++;
	    }
	}
	memset( recvbuf, 0, SEARCH_BUFLEN_RX );
    }
    
    close_ret = close( sendsock );
    if ( close_ret != 0 ) {
	WARN(0, "close(sendsock) failed.\n");
	exit(1);
    }
    
    close_ret = close( recvsock );
    if ( close_ret != 0 ) {
	WARN(0, "close(recvsock) failed.\n");
	exit(1);
    }

    if ( num_daemon > 0 ) {
	WARN( 2, "Found %d valid DSCUDA daemon%s. (%d ignored).\n",
	      num_daemon, (num_daemon>1)? "s":"", num_ignore );
    } else {
	/* Terminate program and exit. */
	if ( num_daemon == 0 ) {
	    WARN( 0, "%s(): Not found DS-CUDA daemon in this network.\n", __func__ );
	    WARN( 0, "%s(): Terminate this application.\n", __func__ );
	} else {
	    WARN( 0, "%s(): Detected unexpected trouble; num_daemon=%d\n", __func__, num_daemon );
	}
	exit(-1);
    }

    return num_daemon;
}

/*
 *
 */
static
void initVirtualServerList(const char *env) {
    char *ip, *hostname, ips[RC_NVDEVMAX][256];
    char buf[1024*RC_NVDEVMAX];
    RCServer_t *sp;
    Vdev_t *pvdev;
    
    if (env) {
	if (sizeof(buf) < strlen(env)) {
	    WARN(0, "initEnv:evironment variable DSCUDA_SERVER too long.\n");
	    exit(1);
	}
	strncpy(buf, env, sizeof(buf));
	St.Nvdev = 0;
	ip = strtok(buf, DELIM_VDEV); // a list of IPs which consist a single vdev.
	while (ip != NULL) {
	    strcpy(ips[St.Nvdev], ip);
	    St.Nvdev++; /* counts Nvdev */
	    if (RC_NVDEVMAX < St.Nvdev) {
		WARN(0, "initEnv:number of devices exceeds the limit, RC_NVDEVMAX (=%d).\n",
		     RC_NVDEVMAX);
		exit(1);
	    }
	    ip = strtok(NULL, DELIM_VDEV);
	}
	for ( int i=0; i<St.Nvdev; i++ ) {
	    int nred=0;
	    int uniq=0; // begin with 0.
	    pvdev = St.Vdev + i;  /* vdev = &Vdev[i]  */
	    ip = strtok(ips[i], DELIM_REDUN); // an IP (optionally with devid preceded by a colon) of
	    // a single element of the vdev.
	    while (ip != NULL) {
		strcpy(pvdev->server[nred].ip, ip);
		nred++;
		ip = strtok(NULL, DELIM_REDUN);
	    }
	    /*
	     * Update Vdev.nredundancy.
	     */
	    pvdev->nredundancy = nred;
	    /*
	     * update Vdev.info.
	     */
	    if ( nred == 1 ) {
		pvdev->conf = VDEV_MONO;
		sprintf( pvdev->info, "MONO" );

	    } else if ( nred > 1 ) {
		pvdev->conf = VDEV_POLY;
		sprintf( pvdev->info, "POLY%d", pvdev->nredundancy );
	    } else {
		WARN( 0, "detected invalid nredundancy = %d.\n", nred );
		exit( EXIT_FAILURE );
	    }
	    
	    sp = pvdev->server;
	    for (int ired=0; ired<nred; ired++) {
		strncpy(buf, sp->ip, sizeof(buf));
		ip = strtok(buf, ":");
		strcpy(sp->ip, ip);
		ip = strtok(NULL, ":");
		sp->id  = ired;
		sp->cid = ip ? atoi(ip) : 0;
		sp->uniq = uniq;
		uniq++;
		sp++;
	    }
	}
	/* convert hostname to ip address. */
	int  det_abc;
	char letter;
	char *ip_ref;
	struct hostent *hostent0;
	for ( int i=0; i<St.Nvdev; i++ ) {
	    for (int j=0; j < St.Vdev[i].nredundancy; j++) {
		ip = St.Vdev[i].server[j].ip;
		hostname = St.Vdev[i].server[j].hostname;
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
		if ( det_abc == 1 ) {
		    strcpy( hostname, ip );
		    hostent0 = gethostbyname( hostname );
		    if ( hostent0 == NULL ) {
			//herror("initVirtualServerList():");
			WARN( 0, "May be set invalid hostname \"%s\" to DSCUDA_SERVER or something.\n", hostname );
			WARN( 0, "Program terminated.\n\n\n\n" );
			exit( EXIT_FAILURE );
		    } else {
			ip_ref = inet_ntoa( *(struct in_addr*)hostent0->h_addr_list[0] );
			strcpy( ip, ip_ref );
		    }
		}
	    }
	}
    } else {
	St.Nvdev = 1;
	St.Vdev[0].nredundancy = 1;
	sp = St.Vdev[0].server;
	sp->id = 0;
	strncpy(sp->ip, DEFAULT_SVRIP, sizeof(sp->ip));
    }
}
static
void updateSpareServerList(void) //TODO: need more good algorithm.
{
    int         spare_count = 0;;
    Vdev_t     *pVdev;
    RCServer_t *pSvr;
    int i, j, k;

    for ( i=0; i < SvrCand.num; i++ ) {
	int found = 0;
	pVdev = St.Vdev;
	for ( j=0; j<St.Nvdev; j++) {
	    pSvr = pVdev->server;
	    for ( k=0; k < pVdev->nredundancy; k++) {
		if ( strcmp( SvrCand.svr[i].ip,  pSvr->ip  )==0 &&
		     SvrCand.svr[i].cid==pSvr->cid ) { /* check same IP */
		    found=1;
		}
		pSvr++;
	    }
	    pVdev++;
	}
	if ( found==0 ) { /* not found */
	    SvrSpare.svr[spare_count].id   = SvrCand.svr[i].id;
	    SvrSpare.svr[spare_count].cid  = SvrCand.svr[i].cid;
	    SvrSpare.svr[spare_count].uniq = SvrCand.svr[i].uniq;
	    strcpy(SvrSpare.svr[spare_count].ip, SvrCand.svr[i].ip);
	    spare_count++;
	}
    }
    SvrSpare.num = spare_count;
}
static
void copyServer(RCServer_t *dst, RCServer_t *src)
{
    //dst->id   = src->id;   // id must be the same.
    //dst->cid  = src->cid;  // cid too.
    dst->uniq = src->uniq;
    strcpy(dst->ip, src->ip);
}
static
void swapServer(RCServer_t *s0, RCServer_t *s1)
{
    RCServer_t buf;
    copyServer(&buf, s0);
    copyServer(s0,   s1);
    copyServer(s1, &buf);
}

void replaceBrokenServer(RCServer_t *broken, RCServer_t *spare)
{
    RCServer_t tmp;

    if ( SvrSpare.num < 1) {  //redundant check?
	WARN(0, "(+_+) Not found any spare server.\n");
	exit(1);
    } else {
	swapServer(broken, spare);
    }
}

static void printVersion(void)
{
    WARN(0, "Found DSCUDA_VERSION= %s\n", RC_DSCUDA_VER);
}
static void updateWarnLevel(void)
{
    char *env = getenv("DSCUDA_WARNLEVEL");
    int val;
    if (env) {
        val = atoi(strtok(env, " "));
        if (val >= 0) {
	    dscudaSetWarnLevel(val);
	} else {
	    WARN(0, "(;_;) Invalid DSCUDA_WARNLEVEL(%d), set 0 or positive integer.\n", val);
	    exit(1);
	}
    } else {
	dscudaSetWarnLevel(RC_WARNLEVEL_DEFAULT);
    }
    WARN(1, "Found DSCUDA_WARNLEVEL= %d\n", dscudaWarnLevel());
}

static
void updateDscudaPath(void) {
    char *env = getenv("DSCUDA_PATH");
    if (env) {
	strncpy(Dscudapath, env, sizeof(Dscudapath)); //"Dscudapath" has global scape.
    } else {
        fprintf(stderr, "(;_;) An environment variable 'DSCUDA_PATH' not found.\n");
        exit(1);
    }
    WARN(2, "Found DSCUDA_PATH= %s\n", Dscudapath);
}

/*
 *
 */
void ClientState_t::setFaultTolerantMode(void) {
    char *env;

    env = getenv( "DSCUDA_USEDAEMON" );
    if ( env == NULL ) {
	daemon = 0;
    } else {
	daemon = atoi( env );
    }
    
    env = getenv( "DSCUDA_AUTOVERB" );
    if ( env == NULL ) {
	autoverb = 0;
    } else {
	autoverb = atoi( env );
    }

    env = getenv( "DSCUDA_MIGRATION" );
    if ( env == NULL ) {
	migration = 0;
    } else {
	migration = atoi( env );
    }

    if ( autoverb == 0 ) {
	if ( migration == 0 ) {
	    ft_mode = FT_PLAIN;
	} else if ( migration > 0 ) {
	    ft_mode = FT_MIGRA;
	} else {
	    WARN( 0, "Found invalid setting of DSCUDA_MIGRATION=%d\n", migration );
	    exit( EXIT_FAILURE );
	}
    } else if ( autoverb > 0 ) {
	if ( migration == 0 ) {
	    ft_mode = FT_REDUN;
	} else if ( migration > 0 ) {
	    ft_mode = FT_BOTH;
	} else {
	    WARN( 0, "Found invalid setting of DSCUDA_MIGRATION=%d\n", migration );
	    exit( EXIT_FAILURE );
	}
    } else {
	WARN( 0, "Found invalid setting of DSCUDA_AUTOVERB=%d\n", autoverb );
	exit( EXIT_FAILURE );
    }
    
    WARN( 2, "Found DSCUDA_USEDAEMON=%d\n", daemon    );
    WARN( 2, "Found DSCUDA_AUTOVERB=%d\n",  autoverb  );
    WARN( 2, "Found DSCUDA_MIGRATION=%d\n", migration );
    WARN( 2, "*****************************************************\n");
    WARN( 2, "***  Configured Fault Tolerant Mode as ");
    switch ( ft_mode ) {
    case FT_PLAIN: WARN0( 2, "\"FT_PLAIN\" ***\n"); break;
    case FT_REDUN: WARN0( 2, "\"FT_REDUN\" ***\n"); break;
    case FT_MIGRA: WARN0( 2, "\"FT_MIGRA\" ***\n"); break;
    case FT_BOTH:  WARN0( 2, "\"FT_BOTH\"  ***\n"); break;
    default:
	WARN0( 0, "(UNKNOWN).\n");
	exit( EXIT_FAILURE );
    }
    WARN( 2, "*****************************************************\n");
    return;
}

/*
 * 
 */
void ClientState_t::initEnv(void) {
    char *sconfname;
    char *env;

    printVersion();
    updateDscudaPath();      /* set "Dscudapath[]" from DSCUDA_PATH */
    updateWarnLevel();       /* set "" from DSCUDA_WARNLEVEL */
    // DSCUDA_SERVER
    if (sconfname = getenv("DSCUDA_SERVER_CONF")) {
        env = readServerConf(sconfname);
    } else {
        env = getenv("DSCUDA_SERVER");
    }

    setFaultTolerantMode();
    
    resetServerUniqID();      /* set all unique ID to invalid(-1)*/
    
    dscudaSearchDaemon();

    initVirtualServerList(env);  /* Update the list of virtual devices information. */
    updateSpareServerList();
    
    printVirtualDeviceList(); /* Print result to screen. */

    WARN(2, "method of remote procedure call: ");
    switch ( dscudaRemoteCallType() ) {
    case RC_REMOTECALL_TYPE_RPC:
	WARN(2, "RPC\n");
	break;
    case RC_REMOTECALL_TYPE_IBV:
	WARN(2, "InfiniBand Verbs\n");
	break;
    default:
	WARN(0, "(Unkown)\n"); exit(1);
    }

    /*
     * Create a thread of checkpointing.
     */
    if ( ft_mode==FT_REDUN || ft_mode== FT_MIGRA || ft_mode==FT_BOTH ) {
	pthread_create( &tid, NULL, periodicCheckpoint, NULL);
    }
    
    return;
}

void ClientState_t::initProgress( ClntInitStat stat ) {
    switch(stat) {
    case ORIGIN:
	init_stat = ORIGIN;
	break;
    case INITIALIZED:
	if (init_stat==ORIGIN) {
	    WARN(5, "init_stat is set to INITIALIZED from ORIGIN.\n")
	    init_stat = INITIALIZED;
	} else {
	    fprintf(stderr, "%s(): unexpected state transition to INITIALIZED.\n",
		    __func__);
	    exit(1);
	}
	break;
    case CUDA_CALLED:
	if (init_stat==INITIALIZED || init_stat==CUDA_CALLED) {
	    WARN(5, "init_stat is set to CUDA_CALLED from %d.\n", init_stat)
	    init_stat = CUDA_CALLED;
	} else {
	    fprintf(stderr, "%s(): unexpected state transition from %d to CUDA_CALLED.\n",
		    __func__, init_stat);
	    exit(1);
	}
	break;
    default:
	fprintf(stderr, "%s(): unexpected state transition to UNKNOWN.\n", __func__);
	exit(1);
    }
}

/*
 * Take the data backups of each virtualized GPU to client's host memory
 * after verifying between redundant physical GPUs every specified wall clock
 * time period. The period is defined in second.
 */
void*
ClientState_t::periodicCheckpoint( void *arg ) {
    int devid, d, i, j, m, n;
    int errcheck = 1;
    cudaError_t cuerr;
    int pmem_devid;
    BkupMem *pmem;
    int  pmem_count;
    void *lsrc;
    void *ldst;
    int  redun;
    int  size;
    
    int  cmp_result[RC_NREDUNDANCYMAX][RC_NREDUNDANCYMAX]; //verify
    int  regional_match;
    int  snapshot_match = 1;
    int  snapshot_count = 0;
    void *dst_cand[RC_NREDUNDANCYMAX];
    int  dst_color[RC_NREDUNDANCYMAX], next_color;

    dscudaMemcpyD2HResult *rp;

    //
    // Wait for all connections established, invoked 1st call of initClient().
    //
    WARN( 10, "%s() thread starts.\n", __func__ );

    int passflag = 0;
    while ( init_stat == ORIGIN ) {
	if ( passflag == 0 ) {
	    WARN(10, "%s() thread wait for INITIALIZED.\n", __func__);
	}
     	sleep(1);
	passflag = 1;
    }
    WARN(10, "%s() thread detected INITIALIZED.\n", __func__);
    
    for (;;) { /* infinite loop */
	WARN( 10, "%s\n", __func__ );
#if 1
	for ( d=0; d < Nvdev; d++ ) { /* Sweep all virtual GPUs */
	    pmem = Vdev[d].bkupmemlist.head ;
	    while ( pmem != NULL ) {
		cudaSetDevice_clnt( d, errcheck );
		
		// mutex locks
		pthread_mutex_lock( &cudaMemcpyD2H_mutex );
		pthread_mutex_lock( &cudaMemcpyH2D_mutex );
		pthread_mutex_lock( &cudaKernelRun_mutex );
		WARN(3, "mutex_lock:%s(),cudaMemcpyD2H\n", __func__);

		for ( i=0; i < Vdev[d].nredundancy; i++ ) {
		    
		}

		// mutex unlocks
		pthread_mutex_unlock( &cudaMemcpyD2H_mutex );
		pthread_mutex_unlock( &cudaMemcpyH2D_mutex );
		pthread_mutex_unlock( &cudaKernelRun_mutex );

	    }

	}//for
#else	
	for ( devid=0; devid < Nvdev; devid++ ) { /* All virtual GPUs */
	    pmem = BKUPMEM.head;
	    while ( pmem != NULL ) { /* sweep all registered regions */
		pmem_devid = dscudaDevidOfUva( pmem->d_region );
		size = pmem->size;
		if ( devid == pmem_devid ) {
		    cudaSetDevice_clnt( devid, errcheck );
		    redun = Vdev[devid].nredundancy;
		    //<-- Mutex lock
		    pthread_mutex_lock( &cudaMemcpyD2H_mutex );
		    pthread_mutex_lock( &cudaMemcpyH2D_mutex );
		    pthread_mutex_lock( &cudaKernelRun_mutex );
		    WARN(3, "mutex_lock:%s(),cudaMemcpyD2H\n", __func__);
		    for ( i=0; i < redun; i++ ) {
			dst_cand[i] = malloc( size );
			if ( dst_cand[i] == NULL ) {
			    fprintf(stderr, "%s():malloc() failed.\n", __func__ );
			    exit( EXIT_FAILURE );
			}
			cudaMemcpyD2H_redundant( dst_cand[i], pmem->d_region, size, i );
		    }
		    pthread_mutex_unlock( &cudaMemcpyD2H_mutex );/*mutex-unlock*/
		    pthread_mutex_unlock( &cudaMemcpyH2D_mutex );/*mutex-unlock*/
		    pthread_mutex_unlock( &cudaKernelRun_mutex );/*mutex-unlock*/
		    //--> Mutex lock
		    WARN(3, "mutex_unlock:%s(),cudaMemcpyD2H\n", __func__);
		    /**************************
		     * compare redundant data.
		     **************************/
		    regional_match = 1;

		    for ( i=0; i<RC_NREDUNDANCYMAX; i++) {
			dst_color[i] = -1;
		    }
		    next_color = 0;
		    dst_color[0] = next_color;
		    for ( m=1; m<redun; m++ ) {
			for ( n=0; n<m; n++ ) {
			    WARN(3, "memcmp(%3d <--> %3d, %d Byte)... ", m, n, size);
			    cmp_result[m][n] = memcmp(dst_cand[m], dst_cand[n], size);
			    if ( cmp_result[m][n] == 0 ) {
				WARN0(3, "Matched.\n");
				dst_color[m] = dst_color[n];
				break;
			    } else {
				WARN0(3, "*** Unmatched. ***\n");
				regional_match = 0;
				snapshot_match = 0;
				if ( n == (m - 1) ) {
				    next_color++;
				    dst_color[m] = next_color;
				}
			    }//if ( cmp_result...
			}//for (n...
		    }//for (m...
		    /********************
		     * Verify
		     ********************/
		    for ( i=0; i<redun; i++ ) {
			WARN(3, "redundant pattern: dst_color[%d]=%d\n", i, dst_color[i]);
		    }
		    if ( regional_match == 1 ) {    /* completely matched data */
			WARN(3, "(^_^) matched all redundants(%d).\n", redun);
			memcpy( pmem->h_region, dst_cand[0], size );
		    } else {                   /* unmatch exist */
			fprintf(stderr, "%s(): unmatched data.\n", __func__ );
			exit(1);
		    }
		    /*
		     * free 
		     */
		    for ( i=0; i<redun; i++ ) {
			free( dst_cand[i] );
		    }
		} 
		pmem = pmem->next;
	    }
	}//for ( devid=0; ...
	
#endif //replacing new code
	
	if (snapshot_match == 1) {
	    /*****************************************
	     * Update snapshot memories, and storage.
	     */
	    WARN(3, "***********************************\n");
	    WARN(3, "*** Update checkpointing data(%d).\n", snapshot_count);
	    WARN(3, "***********************************\n");
	    pmem = BKUPMEM.head;
	    pmem_count = 0;
	    while ( pmem != NULL ) { /* sweep all registered regions */
		pmem->updateSafeRegion();
		pmem_count++;	
		pmem = pmem->next;
	    }
	    WARN(3, "*** Made checkpointing of all %d regions.\n", pmem_count);
	    WARN(3, "***********************************\n");
	    WARN(3, "*** Clear following cuda API called history(%d).\n", snapshot_count);
	    WARN(3, "***********************************\n");
	    HISTREC.print();
	    HISTREC.clear();
	    snapshot_count++;	    
	} else {
	    /*
	     * Rollback and retry cuda sequence.
	     */
	    WARN(3, "Can not update checkpointing data, then Rollback retry.\n");
	}
	sleep(2);
	pthread_testcancel();/* cancelation available */
    }//for (;;)
}


/*
 * Client initializer.
 * This function may be executed in parallel threads, so need mutex lock.    
 */
ClientState_t::ClientState_t(void) {
    int i, k;
    
    start_time = time( NULL );
    WARN( 1, "[ERRORSTATICS] start.\n" );
    
    initProgress( ORIGIN );
	
    WARN( 5, "The constructor %s() called.\n", __func__ );

    ip_addr     = 0;
    use_ibv     = 0;
    autoverb    = 0;
    migration   = 0;
    daemon      = 0;
    historical_calling = 0;
    
    initEnv();
    
    for ( i=0; i < Nvdev; i++ ) {
	for (k=0; k < Vdev[i].nredundancy; k++ ) {
            setupConnection( i, &Vdev[i].server[k] );
        }
    }
    struct sockaddr_in addrin;
    get_myaddress(&addrin);
    setIpAddress(addrin.sin_addr.s_addr);

    WARN(2, "Client IP address : %s\n", dscudaGetIpaddrString(St.getIpAddress()));
    
    initProgress( INITIALIZED );    
    WARN( 5, "The constructor %s() ends.\n", __func__);
}

ClientState_t::~ClientState_t(void) {
    RCServer *svr;
    time_t exe_time;
    char my_tfmt[64];	      
    struct tm *my_local;
    
    stop_time = time( NULL );
    exe_time = stop_time - start_time;

    WARN( 1, "[ERRORSTATICS] stop.\n" );
    WARN( 1, "[ERRORSTATICS] ************** Summary *******************************\n" );
    
    my_local = localtime( &start_time );
    strftime( my_tfmt, 64, "%c", my_local );
    WARN( 1, "[ERRORSTATICS]  Start_time: %s\n", my_tfmt );
    
    my_local = localtime( &stop_time );
    strftime( my_tfmt, 64, "%c", my_local );
    WARN( 1, "[ERRORSTATICS]  Stop_time:  %s\n", my_tfmt );

    my_local = localtime( &exe_time );
    strftime( my_tfmt, 64, "%s", my_local );
    WARN( 1, "[ERRORSTATICS]  Run_time:   %s (sec)\n", my_tfmt );
    for ( int i=0; i<Nvdev; i++ ) {

	WARN( 1, "[ERRORSTATICS]  Virtual[%2d]\n", i );
	for ( int j=0; j<Vdev[i].nredundancy; j++ ) {
	    svr = &Vdev[i].server[j];
	    WARN( 1, "[ERRORSTATICS]  + Physical[%2d]:%s:%s: ErrorCount= %d\n",
		  j, svr->ip, svr->hostname, svr->errcount );
	}
    }
    WARN( 1, "[ERRORSTATICS] ******************************************************\n" );
}

void invalidateModuleCache(void) {
    for (int i=0; i<RC_NKMODULEMAX; i++) {
        if( CltModulelist[i].isValid() ){
	    CltModulelist[i].invalidate();
	} else { 
	    continue;
	}
    }
}

/*
 * public functions
 */

int dscudaNredundancy(void)
{
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    return vdev->nredundancy;
}

void dscudaSetErrorHandler(void (*handler)(void *), void *handler_arg)
{
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
dscudaGetMangledFunctionName(char *name, const char *funcif, const char *ptxdata) {
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
        sprintf(mangler, "%s/bin/ptx2symbol", Dscudapath);
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
int* dscudaLoadModule(char *name, char *strdata) // 'strdata' must be NULL terminated.
{
    int i, j, mid;
    ClientModule *mp;
    int idx;

    if (name != NULL) {
	WARN(5, "dscudaLoadModule(%p) modulename:%s  ...\n", name, name);
#if RC_CACHE_MODULE
	// look for modulename in the module list.
	for (i=0, mp=CltModulelist; i < RC_NKMODULEMAX; i++, mp++) {
	    if ( mp->isInvalid() ) continue;
	    idx = vdevidIndex();
	    if (mp->vdevid != Vdevid[idx]) continue;
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
    Vdev_t *vdev = St.Vdev + Vdevid[idx];

    for (i=0; i < vdev->nredundancy; i++) {
	// mid = dscudaLoadModuleLocal(St.getIpAddress(), getpid(), name, strdata, Vdevid[vi], i);
	mid = dscudaLoadModuleLocal(St.getIpAddress(), getpid(), name_found, strdata_found, Vdevid[idx], i);
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
            CltModulelist[j].setPtxPath(name_found);
	    CltModulelist[j].setPtxImage(strdata_found);
            WARN(5, "New client module item was registered. id:%d\n", mid);
        }
        CltModulelist[j].id[i] = mid;
    }
    CltModulelist[j].vdevid = Vdevid[idx];
    printModuleList();

    return CltModulelist[j].id; //mp->id;
}

cudaError_t
dscudaFuncGetAttributesWrapper(int *moduleid, struct cudaFuncAttributes *attr, const char *func)
{
    cudaError_t err = cudaSuccess;
    dscudaFuncGetAttributesResult *rp;

    WARN(3, "dscudaFuncGetAttributesWrapper(%d, 0x%08llx, %s)...",
         moduleid, (unsigned long long)attr, func);
    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        if (St.isIbv()) {
#warning fill this part in dscudaFuncGetAttributesWrapper().
        } else {
            rp = dscudafuncgetattributesid_1(moduleid[i], (char*)func, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(rp, sp);
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

    if (St.isAutoVerb() &&
	(kind==cudaMemcpyHostToDevice || kind==cudaMemcpyDeviceToDevice)) {
        cudaMemcpyToSymbolArgs args;
        args.moduleid = moduleid;
        args.symbol = (char *)symbol;
        args.src = (void *)src;
        args.count = count;
        args.offset = offset;
        args.kind = kind;
        HISTREC.add(dscudaMemcpyToSymbolH2DId, (void *)&args);
    }

    return err;
}

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

    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        if (St.isIbv()) {

#warning fill this part in dscudaBindTextureWrapper().
        }
        else {
            rp = dscudabindtextureid_1(moduleid[i], texname,
                                       (RCadr)devPtr, size, (RCtexture)texbuf, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(rp, sp);
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

    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        if (St.isIbv()) {

#warning fill this part in dscudaBindTexture2DWrapper().
        } else {

            rp = dscudabindtexture2did_1(moduleid[i], texname,
                                         (RCadr)devPtr, width, height, pitch, (RCtexture)texbuf, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(rp, sp);
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
                               const struct cudaChannelFormatDesc *desc)
{
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

    Vdev_t *vdev = St.Vdev + Vdevid[vdevidIndex()];
    RCServer_t *sp = vdev->server;
    for (int i = 0; i < vdev->nredundancy; i++, sp++) {
        if (St.isIbv()) {

#warning fill this part in dscudaBindTextureToArrayWrapper().
        } else {

            rp = dscudabindtexturetoarrayid_1(moduleid[i], texname, (RCadr)ca->ap[i], (RCtexture)texbuf, Clnt[Vdevid[vdevidIndex()]][sp->id]);
            checkResult(rp, sp);
            if (rp->err != cudaSuccess) {
                err = (cudaError_t)rp->err;
            }
            xdr_free((xdrproc_t)xdr_dscudaResult, (char *)rp);
        }
    }
    WARN(3, "done.\n");
    return err;
}


cudaError_t cudaGetDevice(int *device) {
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaGetDevice(%p)...", device);
    *device = Vdevid[vdevidIndex()];
    WARN(3, "done.\n");

    return err;
}

/*********************************************************************
 * cudaSetDevice()
 *    
 */
cudaError_t cudaSetDevice_clnt( int device, int errcheck ) {
    cudaError_t cuerr = cudaSuccess;
    int vi = vdevidIndex();
    
    WARN(3, "%s(%d), verb=%d, history=%d...\n", __func__, device, St.isAutoVerb(), HISTREC.rec_en);
    if ( 0 <= device && device < St.Nvdev ) {
        Vdevid[vi] = device;
    } else {
        cuerr = cudaErrorInvalidDevice;
	if ( errcheck != 0 ) {
	    fprintf( stderr, "%s(): failed.\n", __func__);
	    exit(1);
	}
    }
    WARN(3, "+--- done.\n");
    return cuerr;
}
cudaError_t cudaSetDevice( int device ) {
    cudaError_t cuerr = cudaSuccess;
    int errcheck = 0; 

    St.cudaCalled();
    WARN(3, "%s(%d), verb=%d, history=%d...\n", __func__, device, St.isAutoVerb(), HISTREC.rec_en);
    if ( HISTREC.rec_en > 0 ) {
        cudaSetDeviceArgs args;
        args.device = device;
        HISTREC.add(dscudaSetDeviceId, (void *)&args);
    }
    cuerr = cudaSetDevice_clnt( device, errcheck );
    WARN(3, "+--- done.\n");
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

cudaError_t cudaGetDeviceCount(int *count)
{
    cudaError_t err = cudaSuccess;

    St.cudaCalled();
    *count = St.Nvdev;
    WARN(3, "cudaGetDeviceCount(%p)  count:%d ...", count, *count);
    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
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

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceEnablePeer(%d, %d)...", peerDevice, flags);
    if (peerDevice < 0 || St.Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)
{
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceDisablePeer(%d)...", peerDevice);
    if (peerDevice < 0 || St.Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}
