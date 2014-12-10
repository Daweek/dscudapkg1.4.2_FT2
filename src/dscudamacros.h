#ifndef DSCUDA_MACROS_H
#define DSCUDA_MACROS_H

#define MACRO_TSTAMP_FORMAT						\
	time_t now = time(NULL);					\
	struct tm *local = localtime( &now );				\
	char tfmt[16];							\
	strftime( tfmt, 16, "%T", local );

#define TTY_RED       "\x1b[31m"				
#define TTY_YELLOW    "\x1b[33m"
#define TTY_BLUE      "\x1b[34m"	
#define TTY_DEFAULT   "\x1b[39m"

#define INFO(fmt, args...) {						\
	MACRO_TSTAMP_FORMAT						\
	int stderr2tty = isatty( fileno( stderr));			\
	if (stderr2tty) {						\
	    fprintf( stderr, TTY_BLUE);					\
	}								\
	fprintf(stderr, "[%s](DSC-INFO) ", tfmt);			\
	fprintf(stderr, fmt, ## args);					\
	if (stderr2tty) {						\
	    fprintf( stderr, TTY_DEFAULT);				\
	}								\
    }

#define ERROR(fmt, args...) {						\
    	MACRO_TSTAMP_FORMAT						\
	int stderr2tty = isatty( fileno( stderr));			\
	if (stderr2tty) {						\
	    fprintf( stderr, TTY_RED);					\
	}								\
	fprintf(stderr, "[%s](DSC-ERROR) ", tfmt);			\
	fprintf(stderr, fmt, ## args);					\
	if (stderr2tty) {						\
	    fprintf( stderr, TTY_DEFAULT);				\
	}								\
    }

#define WARN(lv, fmt, args...) {					\
	if (lv <= dscudaWarnLevel()) {					\
	    MACRO_TSTAMP_FORMAT						\
	    int stderr2tty = isatty( fileno( stderr));			\
	    if (stderr2tty) {						\
		fprintf( stderr, TTY_YELLOW);				\
	    }								\
	    fprintf(stderr, "[%s](DSC-%d) ", tfmt, lv);			\
	    fprintf(stderr, fmt, ## args);				\
	    if (stderr2tty) {						\
		fprintf( stderr, TTY_DEFAULT);				\
	    }								\
	}								\
    }

#define WARN0(lv, fmt, args...) if (lv <= dscudaWarnLevel()) { \
	fprintf(stderr, fmt, ## args);			       \
    };

#define WARNONCE(lv, fmt, args...) if (lv <= dscudaWarnLevel()) { \
      static int firstcall = 1;					  \
      if (firstcall) {						  \
	 firstcall = 0;						  \
	 fprintf(stderr, fmt, ## args);				  \
      }								  \
   }

#define ALIGN_UP(off, align) (off) = ((off) + (align) - 1) & ~((align) - 1)

int  dscudaWarnLevel(void);
void dscudaSetWarnLevel(int level);
int  dscudaGetFaultInjection(void);
void dscudaSetFaultInjection(int pattern);

#endif //DSCUDA_MACROS_H
