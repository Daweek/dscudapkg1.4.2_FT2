//                             -*- Mode: C++ -*-
// Filename         : dscudautil.h
// Description      : DS-CUDA client node library for Remote Procedure Call.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-02-12 20:57:57
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef DSCUDAUTIL_H
#define DSCUDAUTIL_H

#include <stdio.h>
#include <stdint.h>
#include <time.h>

char       *dscudaMemcpyKindName(cudaMemcpyKind kind);
const char *dscudaGetIpaddrString(unsigned int addr);

namespace dscuda {
    void    setWarnLevel(int level);
    int     getWarnLevel(void);
    int     sprintfDate(char *s, int fmt=0);
    void*   xmalloc(size_t size);
    void    xfree(void *p);
    int      searchDaemon(void);
    uint32_t calcChecksum(void *, size_t);
    double   stopwatch(double *t0);
    double   stopwatch(double *t0, double *min, double *max);
}

extern struct ClientState St;

#define TSTAMP_FORMAT 							\
        struct timeval tv;						\
        gettimeofday( &tv, NULL );					\
	struct tm *local = localtime( &(tv.tv_sec) );			\
	char tfmt[16];							\
	strftime( tfmt, 16, "%T", local );

#define INFO(fmt, args...) {						\
	TSTAMP_FORMAT							\
	fprintf(St.dscuda_stdout, "[%s]info: ", tfmt);			\
	fprintf(St.dscuda_stdout, fmt, ## args);			\
    }

#define INFO0(fmt, args...) {\
	fprintf(St.dscuda_stdout, fmt, ## args);\
    }

#define ERROR(fmt, args...) {\
    	TSTAMP_FORMAT						\
	fprintf(St.dscuda_stdout, "[%s](DSC-ERROR) ", tfmt);	\
	fprintf(St.dscuda_stdout, fmt, ## args);			\
    }

//-- [DSCUDA CLIENT] WARING Message.
#define WARN(lv, fmt, args...) {\
	if (lv <= dscuda::getWarnLevel()) {\
	    TSTAMP_FORMAT					\
            fprintf( St.dscuda_stdout, "[%s](%d) ", tfmt, lv);\
	    fprintf( St.dscuda_stdout, fmt, ## args);\
	    fflush( St.dscuda_stdout );\
	}\
    }

#define WARN0(lv, fmt, args...) {			       \
	if (lv <= dscuda::getWarnLevel()) {		       \
	    fprintf( St.dscuda_stdout, fmt, ## args);	       \
	    fflush( St.dscuda_stdout );			       \
	}						       \
    }

#define WARNONCE(lv, fmt, args...) if (lv <= dscuda::getWarnLevel()) {	\
      static int firstcall = 1;					  \
      if (firstcall) {						  \
	 firstcall = 0;						  \
	 fprintf(stderr, fmt, ## args);				  \
      }								  \
   }

//-- [DSCUDA CLIENT] Foult Tolerant's checkpointing
#define WARN_CP(lv, fmt, args...) {\
	if (lv <= dscuda::getWarnLevel()) {	\
	    TSTAMP_FORMAT\
	    fprintf( St.dscuda_chkpnt, "[%s.%03d](%d) ", tfmt, tv.tv_usec/1000, lv); \
	    fprintf( St.dscuda_chkpnt, fmt, ## args);\
	    fprintf( St.dscuda_stdout, "[%s.%03d](%d)CP: ", tfmt, tv.tv_usec/1000, lv); \
	    fprintf( St.dscuda_stdout, fmt, ## args);\
            fflush ( St.dscuda_chkpnt );\
	    fflush ( St.dscuda_stdout );\
	}\
    }

#define WARN_CP0(lv, fmt, args...) {\
	if (lv <= dscuda::getWarnLevel()) {\
	    fprintf( St.dscuda_chkpnt, fmt, ## args);\
	    fflush ( St.dscuda_chkpnt );\
	    fprintf( St.dscuda_stdout, fmt, ## args);\
	    fflush ( St.dscuda_stdout );\
	}\
    }

//-- [DSCUDA SERVER] WARNING Message.
#define SWARN(lv, fmt, args...)						\
    if ( lv <= dscuda::getWarnLevel() ) {				\
	TSTAMP_FORMAT							\
	fprintf(stderr, "[%s]", tfmt);					\
	fprintf(stderr, "(SVR[%d]-%d) " fmt, TcpPort - RC_SERVER_IP_PORT, lv, ## args); \
    }
#define SWARN0(lv, fmt, args...) {			       \
	if (lv <= dscuda::getWarnLevel()) {		       \
	    fprintf(stderr, fmt, ## args);		       \
	}						       \
    }

#define check_cuda_error(err) {						\
	if (cudaSuccess != err) {					\
	    fprintf(stderr,						\
		    "%s(%i) : check_cuda_error() Runtime API error : %s.\n" \
		    "You may need to restart dscudasvr.\n",		\
		    __FILE__, __LINE__, cudaGetErrorString(err));	\
	}								\
    }

#define fatal_error(exitcode) {					\
	fprintf(stderr,						\
		"%s(%i) : fatal_error().\n"			\
		"Probably you need to restart dscudasvr.\n",	\
		__FILE__, __LINE__);				\
	exit(exitcode);						\
    }								

#define ALIGN_UP(off, align) (off) = ((off) + (align) - 1) & ~((align) - 1)

#endif // DSCUDAUTIL_H
