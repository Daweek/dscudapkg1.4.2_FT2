#ifndef DSCUDA_MACROS_H
#define DSCUDA_MACROS_H

#include "dscudautil.h"

#define MACRO_TSTAMP_FORMAT						\
	time_t now = time(NULL);					\
	struct tm *local = localtime( &now );				\
	char tfmt[16];							\
	strftime( tfmt, 16, "%T", local );

#define INFO(fmt, args...) {						\
	MACRO_TSTAMP_FORMAT						\
	fprintf(stderr, "[%s](DSC-INFO) ", tfmt);			\
	fprintf(stderr, fmt, ## args);					\
    }

#define ERROR(fmt, args...) {						\
    	MACRO_TSTAMP_FORMAT						\
	fprintf(stderr, "[%s](DSC-ERROR) ", tfmt);			\
	fprintf(stderr, fmt, ## args);					\
    }

//-- [DSCUDA CLIENT] WARING Message.
#define WARN(lv, fmt, args...) {					\
	if (lv <= dscudaWarnLevel()) {					\
	    MACRO_TSTAMP_FORMAT						\
	    fprintf( St.dscuda_stdout, "[%s](DSC-%d) ", tfmt, lv);	\
	    fprintf( St.dscuda_stdout, fmt, ## args);			\
	}								\
    }

#define WARN0(lv, fmt, args...) {			       \
	if (lv <= dscudaWarnLevel()) {			       \
	    fprintf( stderr, fmt, ## args);	       \
	}						       \
    }

#define WARNONCE(lv, fmt, args...) if (lv <= dscudaWarnLevel()) { \
      static int firstcall = 1;					  \
      if (firstcall) {						  \
	 firstcall = 0;						  \
	 fprintf(stderr, fmt, ## args);				  \
      }								  \
   }

//-- [DSCUDA SERVER] WARNING Message.
#define SWARN(lv, fmt, args...)						\
    if ( lv <= dscudaWarnLevel() ) {					\
	MACRO_TSTAMP_FORMAT						\
	fprintf(stderr, "[%s]", tfmt);					\
	fprintf(stderr, "(SVR[%d]-%d) " fmt, TcpPort - RC_SERVER_IP_PORT, lv, ## args); \
    }

//-- [DSCUDA DAEMON] WARNING Message.
#define DWARN(lv, fmt, args...) {					\
	if (lv <= WarnLevel) {						\
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

#endif //DSCUDA_MACROS_H
