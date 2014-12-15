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
#include <time.h>

char       *dscudaMemcpyKindName(cudaMemcpyKind kind);
const char *dscudaGetIpaddrString(unsigned int addr);
double      RCgetCputime(double *t0);

namespace dscuda {
    void    setWarnLevel(int level);
    int     getWarnLevel(void);
    int     sprintfDate(char *s, int fmt=0);
    void*   xmalloc(size_t size);
    void    xfree(void *p);
}

extern struct ClientState_t St;

#define MACRO_TSTAMP_FORMAT						\
	time_t now = time(NULL);					\
	struct tm *local = localtime( &now );				\
	char tfmt[16];							\
	strftime( tfmt, 16, "%T", local );

#define INFO(fmt, args...) {						\
	MACRO_TSTAMP_FORMAT						\
	fprintf(St.dscuda_stdout, "[%s]info: ", tfmt);			\
	fprintf(St.dscuda_stdout, fmt, ## args);			\
    }

#define INFO0(fmt, args...) {\
	fprintf(St.dscuda_stdout, fmt, ## args);\
    }

#define ERROR(fmt, args...) {\
    	MACRO_TSTAMP_FORMAT\
	fprintf(St.dscuda_stdout, "[%s](DSC-ERROR) ", tfmt);	\
	fprintf(St.dscuda_stdout, fmt, ## args);			\
    }

//-- [DSCUDA CLIENT] WARING Message.
#define WARN(lv, fmt, args...) {\
	if (lv <= dscuda::getWarnLevel()) {\
	    MACRO_TSTAMP_FORMAT\
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
	if (lv <= dscuda::getWarnLevel()) {\
	    MACRO_TSTAMP_FORMAT\
            fprintf( St.dscuda_stdout, "[%s](%d)CP: ", tfmt, lv);\
	    fprintf( St.dscuda_stdout, fmt, ## args);\
	    fflush( St.dscuda_stdout );\
	}\
    }


//-- [DSCUDA SERVER] WARNING Message.
#define SWARN(lv, fmt, args...)						\
    if ( lv <= dscuda::getWarnLevel() ) {				\
	MACRO_TSTAMP_FORMAT						\
	fprintf(stderr, "[%s]", tfmt);					\
	fprintf(stderr, "(SVR[%d]-%d) " fmt, TcpPort - RC_SERVER_IP_PORT, lv, ## args); \
    }
#define SWARN0(lv, fmt, args...) {			       \
	if (lv <= dscuda::getWarnLevel()) {		       \
	    fprintf(stderr, fmt, ## args);		       \
	}						       \
    }

//-- [DSCUDA DAEMON] WARNING Message.
#define DWARN(lv, fmt, args...) {					\
	if (lv <= dscuda::getWarnLevel()) {				\
	    MACRO_TSTAMP_FORMAT						\
	    fprintf( stderr, "[%s]", tfmt);				\
	    fprintf( stderr, "(DAEMON-%d) " fmt, lv, ## args);		\
	}								\
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
