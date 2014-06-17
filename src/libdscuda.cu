//                             -*- Mode: C++ -*-
// Filename         : libdscuda.cu
// Description      : DS-CUDA client node library for Remote Procedure Call.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-02-12 20:57:57
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
/*
 * This file is included into the bottom of ...
 *     -> "libdscuda_ibv.cu"
 *     -> "libdscuda_rpc.cu"
 */
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <ctype.h>
#include <pwd.h>
#include "dscuda.h"
#include "libdscuda.h"
#include "dscudaverb.h"

static int   VdevidIndexMax = 0;            // # of pthreads which utilize virtual devices.
const  char *DEFAULT_SVRIP = "localhost";
static char Dscudapath[512];

static pthread_mutex_t VdevidMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t       VdevidIndex2ptid[RC_NPTHREADMAX];   // convert an Vdevid index into pthread id.

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

int       Vdevid[RC_NPTHREADMAX] = { 0 };           // the virtual device currently in use.

static int Nvdev;                   /* # of virtual devices available. */
Vdev_t     Vdev[RC_NVDEVMAX];       /* a list of virtual devices. */

SvrList_t SvrCand;
SvrList_t SvrSpare;
SvrList_t SvrBroken;

void (*errorHandler)(void *arg) = NULL;
void *errorHandlerArg = NULL;
CLIENT *Clnt[RC_NVDEVMAX][RC_NREDUNDANCYMAX]; /* RPC clients */
struct rdma_cm_id *Cmid[RC_NVDEVMAX][RC_NREDUNDANCYMAX];

ClientModule CltModulelist[RC_NKMODULEMAX]; /* is Singleton.*/
struct ClientState_t St; // is Singleton

void dscudaRecordHistOn(void)
{
    St.setRecordHist();
}
void dscudaRecordHistOff(void)
{
    St.unsetRecordHist();
}

void dscudaAutoVerbOn(void)
{
    St.setAutoVerb();
}
void dscudaAutoVerbOff(void)
{
    St.unsetAutoVerb();
}

int requestDaemonForDevice(char *ipaddr, int devid, int useibv)
{
    int dsock; // socket for side-band communication with the daemon & server.
    int sport; // port number of the server. given by the daemon.
    char msg[256];
    struct sockaddr_in sockaddr;

    sockaddr = setupSockaddr(ipaddr, RC_DAEMON_IP_PORT);
    dsock = socket(AF_INET, SOCK_STREAM, 0);
    if (dsock < 0) {
        perror("socket");
        exit(1);
    }
    if (connect(dsock, (struct sockaddr *)&sockaddr, sizeof(sockaddr)) == -1) {
        perror("(;_;) Connect");
	fprintf( stderr, "(;_;) Program terminated at %s:L%d\n", __FILE__, __LINE__ );
	fprintf( stderr, "(;_;) Maybe DS-CUDA daemon is not running...\n" );
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
int vdevidIndex(void)
{
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

void RCmappedMemRegister(void *pHost, void* pDevice, size_t size)
{
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

RCmappedMem* RCmappedMemQuery(void *pHost)
{
    RCmappedMem *mem = RCmappedMemListTop;
    while (mem) {
        if (mem->pHost == pHost) {
            return mem;
        }
        mem = mem->next;
    }
    return NULL; // pHost not found in the list.
}

void RCmappedMemUnregister(void *pHost)
{
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
void RCstreamArrayRegister(cudaStream_t *streams)
{
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
void showsta(void)
{
    RCstreamArray *st = RCstreamArrayListTop;
    while (st) {
        fprintf(stderr, ">>> 0x%08llx    prev:%p  next:%p\n", st, st->prev, st->next);
        st = st->next;
    }
}
#endif

RCstreamArray* RCstreamArrayQuery(cudaStream_t stream0)
{
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
void RCstreamArrayUnregister(cudaStream_t stream0)
{
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
void* dscudaUvaOfAdr( void *adr, int devid )
{
    DscudaUva_t adri = (DscudaUva_t)adr;
#if __LP64__
    adri |= ((DscudaUva_t)devid << 48);
#endif
    return (void *)adri;
}
/*
 * Get GPU deviceID from UVA.
 */
int dscudaDevidOfUva( void *adr )
{
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
void *dscudaAdrOfUva( void *adr )
{
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
void RCuvaRegister(int devid, void *adr[], size_t size)
{
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

static void
resetServerUniqID(void)
{
    int i,j;
    for (i=0; i<RC_NVDEVMAX; i++) { /* Vdev[*] */
	for (j=0; j<RC_NREDUNDANCYMAX; j++) {
	    Vdev[i].server[j].uniq = RC_UNIQ_INVALID;
	}
    }
    for (j=0; j<RC_NVDEVMAX; j++) { /* svrCand[*] */
	SvrCand.svr[j].uniq = RC_UNIQ_INVALID;
    }
    for (j=0; j<RC_NVDEVMAX; j++) { /* svrSpare[*] */
	SvrSpare.svr[j].uniq = RC_UNIQ_INVALID;
    }
}

void printVirtualDeviceList(void)
{
    Vdev_t     *pVdev;
    RCServer_t *pSvr;
    int         i,j;
    printf("#(info.) *** Virtual Device Info.(Nvdev=%d)\n", Nvdev);
    for( i=0, pVdev=Vdev; i<Nvdev; i++, pVdev++ ){
	if (i >= RC_NVDEVMAX) {
	    WARN(0, "(;_;) Too many virtual devices. %s().\nexit.", __func__);
	    exit(1);
	}
	printf("#(info.) Virtual Device[%d] (Redundancy: %d)\n", i, pVdev->nredundancy);
	for (j=0, pSvr=pVdev->server; j<pVdev->nredundancy; j++, pSvr++) {
	    if (j >= RC_NREDUNDANCYMAX) {
		WARN(0, "(;_;) Too many redundant devices %d. %s().\nexit.\n", __func__);
		exit(1);
	    }
	    printf("#(info.)    + Raid[%d]: id=%d, cid=%d, IP=%s(%s), uniq=%d.\n", j,
		   pSvr->id, pSvr->cid, pSvr->ip, pSvr->hostname, pSvr->uniq);
	}
    }
    /*            */
    printf("#(info.) *** Candidate Server Info.(Ncand=%d)\n",SvrCand.num);
    for( i=0, pSvr=SvrCand.svr; i < SvrCand.num; i++, pSvr++ ){
	if (i >= RC_NVDEVMAX) {
	    WARN(0, "(;_;) Too many candidate devices. %s().\nexit.", __func__);
	    exit(1);
	}
	printf("#(info.)    - Cand[%d]: id=%d, cid=%d, IP=%s, uniq=%d.\n", i,
	       pSvr->id, pSvr->cid, pSvr->ip, pSvr->uniq);
    }
    /*                 */
    printf("#(info.) *** Spare Server Info.(Nspare=%d)\n", SvrSpare.num);
    for( i=0, pSvr=SvrSpare.svr; i < SvrSpare.num; i++, pSvr++ ){
	if (i >= RC_NVDEVMAX) {
	    WARN(0, "(;_;) Too many spare devices. %s().\nexit.", __func__);
	    exit(1);
	}
	printf("#(info.)    - Spare[%d]: id=%d, cid=%d, IP=%s, uniq=%d.\n", i,
	       pSvr->id, pSvr->cid, pSvr->ip, pSvr->uniq);
    }
    fflush(stdout);
}

void printModuleList(void) {
    const int len = 256;
    char printbuf[len];
    int valid_cnt = 0;
    printf("# %s(): ====================================================\n", __func__);
    printf("# %s(): RC_NKMODULEMAX= %d\n", __func__, RC_NKMODULEMAX);
    
    for (int i=0; i<RC_NKMODULEMAX; i++) {
	if( CltModulelist[i].isValid() ){
	    printf("# %s(): CltModulelist[%d]:\n", __func__, i);
	    printf("# %s():    + vdevid= %d\n", __func__, CltModulelist[i].vdevid);
	    for (int j=0; j<4; j++) {
		printf("# %s():    + id[%d]= %d\n", __func__, j, CltModulelist[i].id[j]);
	    }
	    printf("# %s():    + name= %s\n", __func__, CltModulelist[i].name);
	    //printf("    + send_time= \n", sent_time., sent_time.);
	    strncpy(printbuf, CltModulelist[i].ptx_image, len - 1 );
	    printbuf[256]='\0';
	    //printf("# %s():    + ptx_image=\n%s\n", __func__, printbuf);
	    valid_cnt++;
	}
    }
    printf("# %s(): %d valid modules registered.\n", __func__, valid_cnt);
    printf("# %s(): ====================================================\n", __func__);
    fflush(stdout);
}

static
int dscudaSearchDaemon(char *ips, int size)
{
    char sendbuf[SEARCH_BUFLEN];
    char recvbuf[SEARCH_BUFLEN];
    char *magic_word, *user_name;
    int num_svr = 0; // # of dscuda daemons found.
    int num_ignore = 0;
    int sock, recvsock, val = 1;
    unsigned int adr, mask;
    socklen_t sin_size;

    struct sockaddr_in addr, svr;
    struct ifreq ifr[2];
    struct ifconf ifc;
    struct passwd *pwd;

    WARN(2, "#(info) Searching DSCUDA daemons.\n");
    sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    recvsock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if( sock == -1 || recvsock == -1 ) {
	perror("dscudaSearchDaemon: socket()");
	return -1;
    }
    setsockopt(sock, SOL_SOCKET, SO_BROADCAST, &val, sizeof(val));

    ifc.ifc_len = sizeof(ifr) * 2;
    ifc.ifc_ifcu.ifcu_buf = (char *)ifr;
    ioctl(sock, SIOCGIFCONF, &ifc);

    ifr[1].ifr_addr.sa_family = AF_INET;
    ioctl(sock, SIOCGIFADDR, &ifr[1]);
    adr = ((struct sockaddr_in *)(&ifr[1].ifr_addr))->sin_addr.s_addr;
    ioctl(sock, SIOCGIFNETMASK, &ifr[1]);
    mask = ((struct sockaddr_in *)(&ifr[1].ifr_netmask))->sin_addr.s_addr;

    addr.sin_family = AF_INET;
    addr.sin_port = htons(RC_DAEMON_IP_PORT - 1);
    addr.sin_addr.s_addr = adr | ~mask;

    strncpy( sendbuf, SEARCH_PING, SEARCH_BUFLEN - 1 );
    sendto(sock, sendbuf, SEARCH_BUFLEN, 0, (struct sockaddr *)&addr, sizeof(addr));
    WARN(2, "#(info) +--- Sent message \"%s\"...\n", SEARCH_PING);
    sin_size = sizeof(struct sockaddr_in);
    memset(ips, 0, size);

    svr.sin_family = AF_INET;
    svr.sin_port = htons(RC_DAEMON_IP_PORT - 2);
    svr.sin_addr.s_addr = htonl(INADDR_ANY);
    ioctl(recvsock, FIONBIO, &val);

    if( bind(recvsock, (struct sockaddr *)&svr, sizeof(svr)) != 0 ) {
	perror("dscudaSearchDaemon: bind()");
	return -1;
    }
    
    pwd = getpwuid( getuid() );
    
    sleep(RC_SEARCH_DAEMON_TIMEOUT);
    memset( recvbuf, 0, SEARCH_BUFLEN );
    while( 0 < recvfrom(recvsock, recvbuf, SEARCH_BUFLEN - 1, 0, (struct sockaddr *)&svr, &sin_size) ) {
	WARN(2, "#(info) +--- Recieved ACK \"%s\" ", recvbuf);
	magic_word = strtok(recvbuf, SEARCH_DELIM);
	user_name  = strtok(NULL,    SEARCH_DELIM);
	if ( magic_word==NULL ) {
	    WARN(0, "\n\n###(ERROR) Unexpected token in %s().\n\n", __func__);
	    exit(1);
	} else {
	    WARN(2, "from server \"%s\" ", inet_ntoa(svr.sin_addr));
	    if ( strcmp( magic_word, SEARCH_ACK   )==0 &&
		 strcmp( user_name,  pwd->pw_name )==0 ) { /* Found */
		WARN(2, "valid.\n");
		strcat(ips, inet_ntoa(svr.sin_addr));
		strcat(ips, " ");
		num_svr++;
	    } else {
		WARN(2, "ignored.\n");
		num_ignore++;
	    }
	}
	memset( recvbuf, 0, SEARCH_BUFLEN );
    }
    close( sock );
    close( recvsock );

    if ( num_svr < 0 ) {
	WARN(0, "#(ERROR) Unexpected trouble occur in %s(), num_svr=%d\n", __func__, num_svr );
	exit(-1);
    } else if ( num_svr == 0 ) {
	WARN(0, "#(info) No DSCUDA daemons found.%s()\n", __func__ );
	WARN(0, "#(info) Program terminated.\n");
	exit(-1);
    } else {
	WARN( 2, "#(info) +--- %d valid DSCUDA daemon%s found. (%d ignored).\n",
	      num_svr, (num_svr>1)? "s":"", num_ignore );
    }
    return num_svr;
}

static
void initCandServerList(const char *env)
{
    char *ip;
    char buf[1024 * RC_NVDEVMAX];
    int nsvr;

    nsvr = dscudaSearchDaemon( buf, 1024 * RC_NVDEVMAX );
    if ( nsvr <= 0 ) {
	fprintf(stderr, "(+_+) Not found DS-CUDA daemons in this cluster.\n");
	fprintf(stderr, "(+_+) Program terminated..\n\n\n");
	exit(1);
    }

    int uniq = RC_UNIQ_CANDBASE; // begin with
    SvrCand.clear();
    if ( env != NULL ) { // always true?
	ip = strtok( buf, DELIM_CAND );
	while ( ip != NULL ) {
	    SvrCand.cat( ip );
	    ip = strtok(NULL, DELIM_CAND);
	}
	for (int i=0; i < SvrCand.num; i++) {
	    strncpy(buf, SvrCand.svr[i].ip, sizeof(buf));
	    ip = strtok(buf, ":");
	    strcpy(SvrCand.svr[i].ip, ip);
	    ip = strtok(NULL, ":");
	    //sp->cid = ip ? atoi(ip) : 0; /* Yoshikawa's original */
	    if (ip != NULL) { SvrCand.svr[i].cid = atoi(ip); }
	    else            { SvrCand.svr[i].cid = 0;        }
	    //sp->id will be set on server reconnecting
	    SvrCand.svr[i].uniq = uniq;
	    uniq++;
	}
    } else {
	setenv("DSCUDA_SERVER", buf, 1);
	env = getenv("DSCUDA_SERVER");
    }
}

static
void initVirtualServerList(const char *env)
{
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
	Nvdev = 0;
	ip = strtok(buf, DELIM_VDEV); // a list of IPs which consist a single vdev.
	while (ip != NULL) {
	    strcpy(ips[Nvdev], ip);
	    Nvdev++; /* counts Nvdev */
	    if (RC_NVDEVMAX < Nvdev) {
		WARN(0, "initEnv:number of devices exceeds the limit, RC_NVDEVMAX (=%d).\n",
		     RC_NVDEVMAX);
		exit(1);
	    }
	    ip = strtok(NULL, DELIM_VDEV);
	}
	for (int i=0; i<Nvdev; i++) {
	    int nred=0;
	    int uniq=0; // begin with 0.
	    pvdev = Vdev + i;  /* vdev = &Vdev[i]  */
	    ip = strtok(ips[i], DELIM_REDUN); // an IP (optionally with devid preceded by a colon) of
	    // a single element of the vdev.
	    while (ip != NULL) {
		strcpy(pvdev->server[nred].ip, ip);
		nred++;
		ip = strtok(NULL, DELIM_REDUN);
	    }
	    pvdev->nredundancy = nred;
	    
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
	int det_abc;
	char letter;
	char *ip_ref;
	struct hostent *hostent0;
	for (int i=0; i<Nvdev; i++) {
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
		if (det_abc==1) {
		    strcpy(hostname, ip);
		    hostent0 = gethostbyname(hostname);
		    if (hostent0==NULL) {
			fprintf(stderr, "gethostbyname() returned NULL.\n");
			exit(1);
		    } else {
			ip_ref = inet_ntoa(*(struct in_addr*)hostent0->h_addr_list[0]);
			strcpy(ip, ip_ref);
		    }
		}
	    }
	}
    } else {
	Nvdev = 1;
	Vdev[0].nredundancy = 1;
	sp = Vdev[0].server;
	sp->id = 0;
	strncpy(sp->ip, DEFAULT_SVRIP, sizeof(sp->ip));
    }
}
static
void updateSpareServerList() //TODO: need more good algorithm.
{
    int         spare_count = 0;;
    Vdev_t     *pVdev;
    RCServer_t *pSvr;

    for ( int i=0; i < SvrCand.num; i++ ) {
	int found = 0;
	pVdev = Vdev;
	for (int j=0; j<Nvdev; j++) {
	    pSvr = pVdev->server;
	    for (int k=0; k < pVdev->nredundancy; k++) {
		if (strcmp(SvrCand.svr[i].ip, pSvr->ip)==0) { /* check same IP */
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
    WARN(0, "#(info.) DSCUDA_VERSION= %s\n", RC_DSCUDA_VER);
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
    WARN(1, "#(info.) DSCUDA_WARNLEVEL= %d\n", dscudaWarnLevel());
}

static void
updateDscudaPath(void) {
    char *env = getenv("DSCUDA_PATH");
    if (env) {
	strncpy(Dscudapath, env, sizeof(Dscudapath)); //"Dscudapath" has global scape.
    } else {
        fprintf(stderr, "(;_;) An environment variable 'DSCUDA_PATH' not found.\n");
        exit(1);
    }
    WARN(1, "#(info.) DSCUDA_PATH= %s\n", Dscudapath);
}

static
void initEnv(void)
{
/*
 * set "int Ndev",  "Vdev_t     Vdev[RC_NVDEVMAX]",
 *     "int Ncand", "RCServer_t svrCand[RC_NVDEVMAX]
 *     "autoVerb", "UseDaemon"
 */
    static int firstcall=1;
    char *sconfname, *env;

    if (!firstcall) return;
    firstcall = 0;

    printVersion();
    updateDscudaPath();      /* set "Dscudapath[]" from DSCUDA_PATH */
    updateWarnLevel();       /* set "" from DSCUDA_WARNLEVEL */
    // DSCUDA_SERVER
    if (sconfname = getenv("DSCUDA_SERVER_CONF")) {
        env = readServerConf(sconfname);
    } else {
        env = getenv("DSCUDA_SERVER");
    }

    resetServerUniqID();      /* set all unique ID to invalid(-1)*/
    initCandServerList(env);  /* search servers in which dscudad is running. */
    initVirtualServerList(env);  /* Update the list of virtual devices information. */

    updateSpareServerList();
    
    printVirtualDeviceList(); /* Print result to screen. */

    WARN(2, "method of remote procedure call: ");
    switch (dscudaRemoteCallType()) {
      case RC_REMOTECALL_TYPE_RPC: WARN(2, "RPC\n");              break;
      case RC_REMOTECALL_TYPE_IBV: WARN(2, "InfiniBand Verbs\n"); break;
      default:                     WARN(0, "(Unkown)\n"); exit(1);
    }

    // DSCUDA_AUTOVERB
    env = getenv("DSCUDA_AUTOVERB");

    if (env) {
        St.setAutoVerb();
        dscudaVerbInit();
	WARN(2, "#(info.) Automatic data recovery: on.\n");
    } else {
	WARN(2, "#(info.) Automatic data recovery: off.\n");
    }

    // DSCUDA_USEDAEMON
    env = getenv("DSCUDA_USEDAEMON");
    if (env && atoi(env)) {
        WARN(3, "#(info.) Connect to the server via daemon.\n");
        St.setUseDaemon();
    } else {
        WARN(3, "#(info.) Do not use daemon. connect to the server directly.\n");
        St.unsetUseDaemon();
    }

    // MIGRATE_DEVICE
    env = getenv("DSCUDA_MIGRATION");
    if (env && atoi(env)) {
        WARN(3, "#(info.) Device Migrataion is enabled.\n");
        St.setMigrateDevice();
    } else {
        WARN(3, "#(info.) Device Migration is disabled.\n");
        St.unsetMigrateDevice();
    }

}

static pthread_mutex_t InitClientMutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * Client initialize.
 */
void initClient( void )
{
    static int firstcall = 1;
    int idev;
    int ired;
    Vdev_t *vdev;
    RCServer_t *sp;

    pthread_mutex_lock( &InitClientMutex );

    if ( !firstcall ) {
        pthread_mutex_unlock( &InitClientMutex );
        return;
    }

    initEnv();

    for ( idev = 0; idev < Nvdev; idev++ ) {
        vdev = Vdev + idev;
        sp = vdev->server;
        for ( ired = 0; ired < vdev->nredundancy; ired++, sp++ ) {
            setupConnection(idev, sp);
        }
    }
    struct sockaddr_in addrin;
    get_myaddress(&addrin);
    St.setIpAddress(addrin.sin_addr.s_addr);
    WARN(2, "Client IP address : %s\n", dscudaGetIpaddrString(St.getIpAddress()));
    firstcall = 0;
    pthread_mutex_unlock( &InitClientMutex );
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
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
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

    WARN(4, "getMangledFunctionName(%08llx, %08llx, %08llx)  funcif:\"%s\"\n",
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
    Vdev_t *vdev = Vdev + Vdevid[idx];

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

    initClient();
    WARN(3, "dscudaFuncGetAttributesWrapper(%d, 0x%08llx, %s)...",
         moduleid, (unsigned long)attr, func);
    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
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
    WARN(3, "  attr->constSizeBytes: %d\n", attr->constSizeBytes);
    WARN(3, "  attr->localSizeBytes: %d\n", attr->localSizeBytes);
    WARN(3, "  attr->maxThreadsPerBlock: %d\n", attr->maxThreadsPerBlock);
    WARN(3, "  attr->numRegs: %d\n", attr->numRegs);
    WARN(3, "  attr->ptxVersion: %d\n", attr->ptxVersion);
    WARN(3, "  attr->sharedSizeBytes: %d\n", attr->sharedSizeBytes);

    return err;
}

cudaError_t
dscudaMemcpyToSymbolWrapper(int *moduleid, const char *symbol, const void *src,
                           size_t count, size_t offset, enum cudaMemcpyKind kind)
{
    cudaError_t err = cudaSuccess;
    int nredundancy;

    initClient();

    WARN(3, "dscudaMemcpyToSymbolWrapper(%d, 0x%08llx, 0x%08llx, %d, %d, %s)"
         "symbol:%s  ...",
         moduleid, (unsigned long)symbol, (unsigned long)src,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;
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

    if (St.isAutoVerb() && St.isRecordHist() &&
	(kind == cudaMemcpyHostToDevice || kind == cudaMemcpyDeviceToDevice)) {
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

    initClient();

    WARN(3, "dscudaMemcpyFromSymbolWrapper(0x%08llx, 0x%08llx, 0x%08llx, %d, %d, %s)"
         "symbol:%s  ...",
         moduleid, (unsigned long)dst, (unsigned long)symbol,
         count, offset, dscudaMemcpyKindName(kind), symbol);

    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;
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

    initClient();

    WARN(3, "sym:%s\n", symbol);
    WARN(3, "dscudaMemcpyToSymbolAsyncWrapper(%d, 0x%08lx, 0x%08lx, %d, %d, %s, 0x%08lx) "
         "symbol:%s  ...",
         moduleid, (unsigned long)symbol, (unsigned long)src,
         count, offset, dscudaMemcpyKindName(kind), (unsigned long)stream, symbol);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;
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

    initClient();

    WARN(3, "dscudaMemcpyFromSymbolAsyncWrapper(%d, 0x%08lx, 0x%08lx, %d, %d, %s, 0x%08lx)"
         " symbol:%s  ...",
         moduleid, (unsigned long)dst, (unsigned long)symbol,
         count, offset, dscudaMemcpyKindName(kind), (unsigned long)stream, symbol);
    st = RCstreamArrayQuery(stream);
    if (!st) {
        WARN(0, "invalid stream : 0x%08llx\n", stream);
        exit(1);
    }
    nredundancy = (Vdev + Vdevid[vdevidIndex()])->nredundancy;
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

    initClient();

    WARN(3, "dscudaBindTextureWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx, 0x%08llx, 0x%08llx, %d)...",
         moduleid, texname,
         offset, tex, devPtr, desc, size);

    setTextureParams(&texbuf, tex, desc);

    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
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

    initClient();

    WARN(3, "dscudaBindTexture2DWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx, 0x%08llx, 0x%08llx, %d, %d, %d)...",
         moduleid, texname,
         offset, tex, devPtr, desc, width, height, pitch);

    setTextureParams(&texbuf, tex, desc);

    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
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

    initClient();

    WARN(3, "dscudaBindTextureToArrayWrapper(0x%08llx, %s, 0x%08llx, 0x%08llx)...",
         moduleid, texname, (unsigned long)array, (unsigned long)desc);

    setTextureParams(&texbuf, tex, desc);

    ca = RCcuarrayArrayQuery((cudaArray *)array);
    if (!ca) {
        WARN(0, "invalid cudaArray : 0x%08llx\n", array);
        exit(1);
    }

    Vdev_t *vdev = Vdev + Vdevid[vdevidIndex()];
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

    initClient();
    WARN(3, "cudaGetDevice(0x%08llx)...", (unsigned long)device);
    *device = Vdevid[vdevidIndex()];
    WARN(3, "done.\n");

    return err;
}
/*
 * cudaSetDevice()
 */
cudaError_t cudaSetDevice(int device) {
    cudaError_t err = cudaSuccess;
    int         vi  = vdevidIndex();

    initClient();
    WARN(3, "(WARN-3) %s(%d), verb=%d, history=%d...\n", __func__, device,
	 St.isAutoVerb(), St.isRecordHist());

    if ( 0 <= device && device < Nvdev ) {
        Vdevid[vi] = device;
    } else {
        err = cudaErrorInvalidDevice;
    }

    if (St.isAutoVerb() && St.isRecordHist()) {
        cudaSetDeviceArgs args;
        args.device = device;
        HISTREC.add(dscudaSetDeviceId, (void *)&args);
    }
    WARN(3, "(WARN-3) +--- done.\n");
    return err;
}

cudaError_t
cudaChooseDevice(int *device, const struct cudaDeviceProp *prop) {
    cudaError_t err = cudaSuccess;

    initClient();
    WARN(3, "cudaChooseDevice(0x%08llx, 0x%08llx)...",
         (unsigned long)device, (unsigned long)prop);
    *device = 0;
    WARN(3, "done.\n");
    WARN(3, "Note : The current implementation always returns device 0.\n");

    return err;
}

cudaError_t cudaGetDeviceCount(int *count)
{
    cudaError_t err = cudaSuccess;

    initClient();
    *count = Nvdev;
    WARN(3, "cudaGetDeviceCount(0x%08llx)  count:%d ...",
    (unsigned long)count, *count);
    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device, int peerDevice)
{
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceCanAccessPeer(0x%08lx, %d, %d)...",
         canAccessPeer, device, peerDevice);
    if (device < 0 || Nvdev <= device) {
        err = cudaErrorInvalidDevice;
    }
    if (peerDevice < 0 || Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceEnablePeer(%d, %d)...", peerDevice, flags);
    if (peerDevice < 0 || Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice)
{
    cudaError_t err = cudaSuccess;

    WARN(3, "cudaDeviceDisablePeer(%d)...", peerDevice);
    if (peerDevice < 0 || Nvdev <= peerDevice) {
        err = cudaErrorInvalidDevice;
    }
    WARN(3, "done.\n");

    return err;
}
