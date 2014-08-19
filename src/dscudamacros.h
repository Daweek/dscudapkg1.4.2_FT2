#ifndef DSCUDA_MACROS_H
#define DSCUDA_MACROS_H

#define WARN(lv, fmt, args...) if (lv <= dscudaWarnLevel()) {		\
        time_t now = time(NULL);					\
	struct tm *local = localtime( &now );				\
	char tfmt[16];							\
	strftime( tfmt, 16, "%T", local );				\
	fprintf(stderr, "[%s](DSC-%d) ", tfmt, lv);			\
	fprintf(stderr, fmt, ## args);					\
    };

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
int dscudaWarnLevel(void);
void dscudaSetWarnLevel(int level);
int dscudaGetFaultInjection(void);
void dscudaSetFaultInjection(int pattern);

#endif // DSCUDA_MACROS_H
