#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>

/*
 * utils for communication via TCP socket.
 */

struct sockaddr_in
setupSockaddr( char *ipaddr, int   tcpport ) {
    struct sockaddr_in sockaddr;
    
    memset((char *)&sockaddr, 0, sizeof(sockaddr));
    sockaddr.sin_family      = AF_INET;
    sockaddr.sin_port        = htons( tcpport );
    sockaddr.sin_addr.s_addr = inet_addr( ipaddr ); 
							
    return sockaddr;
}

void sendMsgBySocket(int sock, char *msg) {
    char buf[1024];
    int len = strlen(msg) + 1;

    if (sizeof buf < len) {
        fprintf(stderr, "sendMsgBySocket:message too long.\n");
        exit(1);
    }
    *(int *)buf = htonl(len);
    strcpy(buf + sizeof(int), msg);
    send(sock, buf, sizeof(int) + len, 0);
}

void recvMsgBySocket(int sock, char *msg, int msgbufsize) {
    char buf[1024];
    int len = strlen(msg) + 1;

    recv(sock, buf, sizeof(int), 0);
    len = ntohl(*(int *)buf);
    if (msgbufsize < len) {
        fprintf(stderr, "recvMsgBySocket:message too long.\n");
        exit(1);
    }
    recv(sock, msg, len, 0);
}

