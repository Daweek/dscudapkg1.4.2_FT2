//                             -*- Mode: C++ -*-
// Filename         : dscudad.cu
// Description      : DS-CUDA server daemon.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-09 16:10:33
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <pwd.h>
#include <time.h>
#include "dscuda.h"
#include "dscudadefs.h"
#include "dscudamacros.h"
#include "sockutil.h"

#define NBACKLOG 1   // max # of servers can be put into the listening queue.

static int WarnLevel = 2;
static int CallFaultServer = 0;

typedef struct Server_t {
    pid_t pid;
    int   port;
    struct Server_t *prev;
    struct Server_t *next;
} Server;

static void signal_from_child(int sig);
static Server *server_with_pid(pid_t pid);
static int unused_server_port(int devid);
static void* response_to_search( void *arg );

static int Daemonize = 0;
static int Nserver = 0;
static Server *ServerListTop = NULL;
static Server *ServerListTail = NULL;
static char LogFileName[1024] = "dscudad.log";

static
int create_daemon_socket(in_port_t port, int backlog)
{
    struct sockaddr_in me;
    int sock;

    memset((char *)&me, 0, sizeof(me));
    me.sin_family      = AF_INET;
    me.sin_addr.s_addr = htonl( INADDR_ANY );
    me.sin_port        = htons( port );

    sock = socket( AF_INET, SOCK_STREAM, IPPROTO_TCP );
    if (sock < 0) {
	perror("dscudad:socket");
	return -1;
    }

    /* <-- For avoiding TIME_WAIT status on TCP port. */
    bool yes=1;
    setsockopt( sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));
    /* --> For avoiding TIME_WAIT status on TCP port. */
    
    if (bind(sock, (struct sockaddr *)&me, sizeof(me)) == -1) {
        perror("dscudad:bind");
        return -1;
    }

    if (listen(sock, backlog) == -1) {
        perror("dscudad:listen");
        return -1;
    }
    DWARN(3, "socket for port %d successfully setup.\n", port);

    return sock;
}

static
void register_server(pid_t pid, int port)
{
    DWARN(3, "register_server(%d, %d).\n", pid, port);
    Server *svr = (Server *)xmalloc(sizeof(Server));

    svr->pid  = pid;
    svr->port = port;
    svr->prev = ServerListTail;
    svr->next = NULL;
    if (!ServerListTop) { // svr will be the 1st entry.
        ServerListTop = svr;
    } else {
        ServerListTail->next = svr;
    }
    ServerListTail = svr;
    DWARN(3, "register_server done.\n");
}

static
void unregister_server(pid_t pid)
{
    DWARN(3, "unregister_server(%d).\n", pid);
    Server *svr = server_with_pid(pid);
    if (!svr) {
        DWARN(0, "server with pid %d not found. "
             "unregister operation not performed.\n", pid);
        return;
    }

    if (svr->prev) { // reconnect the linked list.
        svr->prev->next = svr->next;
    } else { // svr was the 1st entry.
        ServerListTop = svr->next;
        if (svr->next) {
            svr->next->prev = NULL;
        }
    }
    if (!svr->next) { // svr was the last entry.
        ServerListTail = svr->prev;
    }
    DWARN(3, "unregister_server done. port:%d released.\n", svr->port);
    DWARN(3, "###     ###   #   #  #####\n");
    DWARN(3, "#   #  #   #  ##  #  #    \n");
    DWARN(3, "#   #  #   #  # # #  #### \n");
    DWARN(3, "#   #  #   #  #  ##  #    \n");
    DWARN(3, "###     ###   #   #  #####\n");
    DWARN(3, "\n");
    DWARN(3, "==============================================================================\n");
    xfree(svr);
}

static
Server *server_with_pid(pid_t pid)
{
    Server *svr = ServerListTop;
    while (svr) {
        if (svr->pid == pid) {
            return svr;
        }
        svr = svr->next;
    }
    return NULL; // server with pid not found in the list.
}

static
int unused_server_port(int devid)
{
    int inuse;
    Server *s;

    DWARN(3, "unused_server_port().\n");

//  for (int p=RC_SERVER_IP_PORT; p<RC_SERVER_IP_PORT + RC_NSERVERMAX; p++) {
    for (int p=RC_SERVER_IP_PORT+devid; p<RC_SERVER_IP_PORT + RC_NSERVERMAX; p++) {
        inuse = 0;
        for (s = ServerListTop; s; s = s->next) {
            if (p == s->port) {
                inuse = 1;
                break;
            }
        }
        if (!inuse) {
            DWARN(3, "unused_server_port: port found:%d\n", p);
            return p;
        }
    }

    DWARN(3, "unused_server_port: all ports in use.\n");
    return -1;
}

static
void spawn_server(int listening_sock)
{
    int len, dev, sock, sport;
    pid_t pid;
    char *argv[16];
    char msg[256];
    char portstr[128], devstr[128], sockstr[128];

    DWARN(3, "listening request from client...\n");
    sock = accept(listening_sock, NULL, NULL);
    if (sock == -1) {
        DWARN(0, "accept() error\n");
        exit(1);
    }
    recvMsgBySocket(sock, msg, sizeof(msg));
    sscanf(msg, "deviceid:%d", &dev); // deviceid to be handled by the server.
    DWARN(3, "---> Recv message \"%s\"\n", msg);

    sport = unused_server_port(dev);
    sprintf(msg, "sport:%d", sport); // server port to be connected by the client.
    sendMsgBySocket(sock, msg);
    DWARN(3, "<--- Send message \"%s\"\n", msg);        

    if (sport < 0) {
        DWARN(0, "spawn_server: max possible ports already in use.\n");
        close(sock);
        return;
    }

    Nserver++;
    pid = fork();
    if ( pid ) { // parent
        signal( SIGCHLD, signal_from_child );
        DWARN( 3, "spawn a server with sock: %d\n", sock );
        register_server( pid, sport );
        close( sock );
    } else { // child
#if RPC_ONLY
        argv[0] = "dscudasvr_rpc";
#else
	if ( CallFaultServer == 0 ) {
	    argv[0] = "dscudasvr";
	} else {
	    argv[0] = "dscudasvr_fault";
	}
#endif
        sprintf( portstr, "-p%d", sport );
        argv[1] = portstr;
	
        sprintf( devstr, "-d%d", dev );
        argv[2] = devstr;
	
        sprintf( sockstr, "-S%d", sock );
        argv[3] = sockstr;
	
        argv[4] = (char *)NULL;
        DWARN( 3, "exec %s %s %s %s\n", argv[0], argv[1], argv[2], argv[3] );
	
        execvp( argv[0], (char **)argv );
        perror( argv[0] );
        DWARN( 0, "execvp() failed.\n" );
        DWARN( 0, "%s may not be in the PATH?\n", argv[0] );
        exit( EXIT_FAILURE );
    }
}

/*
 *
 */
static void signal_from_child( int sig )
{
    int status;
    int pid = waitpid(-1, &status, WNOHANG);

    switch (pid) {
    case -1:
        DWARN(0, "signal_from_child:waitpid failed.\n");
        exit(1);
        break;
    case 0:
        DWARN(0, "no child has exited.\n");
        break;
    default:
        DWARN(2, "exited a child (pid:%d).\n", pid);
	
        if (WIFEXITED(status)) {
            DWARN(2, "exit status:%d\n", WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            DWARN(2, "terminated by signal %d.\n", WTERMSIG(status));
        }
        Nserver--;
        unregister_server(pid);
    }
}

/*
 * Run in separated thread by pthread_create();
 */
static void *response_to_search( void *arg )
{
    char sendbuf[ SEARCH_BUFLEN_TX ];
    char recvbuf[ SEARCH_BUFLEN_RX ];

    int       dev_count;
    int       sock;
    socklen_t sin_size;
    uid_t     uid;
    char      host[256];
    struct sockaddr_in addr, clt;
    struct passwd *pwd = NULL;
    int       retval;
    cudaError_t cuerr;

    cuerr = cudaGetDeviceCount( &dev_count );
    if ( cuerr != cudaSuccess ) {
	DWARN( 0, "cudaGetDeviceCount() failed.\n");
	return NULL;
    } else if ( dev_count == 0 ) {
	DWARN( 0, "cudaGetDeviceCount() returned 0.\n");
	return NULL;
    }
    
    sock = socket( AF_INET, SOCK_DGRAM, IPPROTO_UDP );
    if ( sock == -1 ) {
	perror("response_to_search:socket()");
	return NULL;
    }

    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(RC_DAEMON_IP_PORT - 1);
    addr.sin_addr.s_addr = htonl( INADDR_ANY );

    /* <-- For avoiding TIME_WAIT status on TCP port. */
    bool yes=1;
    setsockopt( sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&yes, sizeof(yes));
    /* --> For avoiding TIME_WAIT status on TCP port. */
    
    if ( bind(sock, (struct sockaddr *)&addr, sizeof(addr)) == -1 ) {
	perror("response_to_search:bind()");
	return NULL;
    }

    /* Construct send message. */
    uid = getuid(); 
    pwd = getpwuid(uid);
    
    retval = gethostname( host, 256 );
    if ( retval != 0 ) {
	perror("gethostname()");
	exit(1);
    }
    
    if ( pwd == NULL ) {
	sprintf( sendbuf, "%s:NULL@%s", SEARCH_ACK, host);	
    } else {
	sprintf( sendbuf, "%s:%s@%s:Ndev=%d", SEARCH_ACK, pwd->pw_name, host, dev_count );	
    }

    memset( recvbuf, 0, SEARCH_BUFLEN_RX );
    for(;;) {
	sin_size = sizeof( struct sockaddr_in );
	recvfrom( sock, recvbuf, SEARCH_BUFLEN_RX, 0, (struct sockaddr *)&clt, &sin_size );
	if( strcmp( recvbuf, SEARCH_PING ) != 0 ) continue;

	DWARN(2, "Received message \"%s\" from %s\n", SEARCH_PING, inet_ntoa(clt.sin_addr));
	if ( Nserver > 0 ) {
	    DWARN(2, "Invoked %d dscudasvr%c, so I can't reply.\n", Nserver, Nserver?' ':'s');
	    memset( recvbuf, 0, SEARCH_BUFLEN_RX );
	    continue;
	}
	
	clt.sin_family = AF_INET;
	clt.sin_port   = htons( RC_DAEMON_IP_PORT - 2 );
	inet_aton( inet_ntoa( clt.sin_addr ), &(clt.sin_addr) );
	
	sendto( sock, sendbuf, SEARCH_BUFLEN_TX, 0, (struct sockaddr *)&clt, sizeof(struct sockaddr) );
	DWARN(2, "---> Replied message \"%s\" to %s\n", sendbuf, inet_ntoa(clt.sin_addr)); 
	memset( recvbuf, 0, SEARCH_BUFLEN_RX );
    }

    close(sock);
    return NULL;
}

/*
 *
 */
static
void initEnv(void)
{
    static int firstcall = 1;
    char *env;

    if (!firstcall) return;

    firstcall = 0;

    // DSCUDA_WARNLEVEL
    env = getenv("DSCUDA_WARNLEVEL");
    if (env) {
        int tmp;
        tmp = atoi(strtok(env, " "));
        if (0 <= tmp) {
            WarnLevel = tmp;
        }
        DWARN(1, "WarnLevel: %d\n", WarnLevel);
    }
}

/*
 *
 */
void showUsage(char *command)
{
    fprintf(stderr,
            "usage: %s [-d]\n"
            "  -d: daemonize.\n",
            command);
}

extern char *optarg;
extern int optind;
static
void parseArgv( int argc, char **argv )
{
    int c;
    char *param = "dfl:h";

    while ((c = getopt(argc, argv, param)) != EOF) {
        switch (c) {
	case 'd':
            Daemonize = 1;
            break;
	case 'f':
	    CallFaultServer = 1;
	    break;
	case 'l':
            strncpy( LogFileName, optarg, sizeof(LogFileName) );
            break;
	case 'h':
	default:
	    showUsage(argv[0]);
            exit(1);
        }
    }
}

int main(int argc, char **argv)
{
    int sock, nserver0;
    int errfd;
    pthread_t th;
    
    pthread_create( &th, NULL, response_to_search, NULL );
    parseArgv( argc, argv );
    if ( Daemonize ) {
        if ( fork() ) {
            exit(0);
        } else {
            close(2);
            errfd = open( LogFileName, O_RDWR | O_CREAT | O_APPEND, S_IRUSR | S_IWUSR );
            if ( errfd < 0 ) {
		perror( "open:" );
            }
            close(0);
            close(1);
        }
    }
    
    initEnv();
    
    sock = create_daemon_socket( RC_DAEMON_IP_PORT, NBACKLOG );
    if ( sock == -1 ) {
	DWARN(0, "create_daemon_socket() failed\n");
	exit(1);
    }
    nserver0 = Nserver;
    for (;;) {
        if ( Nserver < RC_NSERVERMAX ) {
            spawn_server( sock );

            if ( nserver0 != Nserver ) {
                if ( Nserver < RC_NSERVERMAX ) {
                    DWARN(0, "%d servers active (%d max possible).\n", Nserver, RC_NSERVERMAX);
                } else {
                    DWARN(0, "%d servers active. reached the limit.\n", Nserver);
                }
            }
        }
        sleep(1);
        nserver0 = Nserver;
    }
    DWARN( 0, "%s: cannot be reached.\n", __FILE__ );

    pthread_join( th, NULL );
    exit(1);
}
