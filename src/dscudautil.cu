//                             -*- Mode: C++ -*-
// Filename         : dscudautil.cu
// Description      : DS-CUDA utility.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-11 12:53:07
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <stdio.h>
#include <sys/time.h>
#include <driver_types.h>
#include "dscudautil.h"

static int WarnLevel = 2; /* warning message output level. the higher the more verbose.
                             0: no warning (may cause wrong result with g7pkg/scripts/check.csh)
                             1: minimum
                             2: default
                             >= 3: for debugging purpose
                          */
int
dscuda::getWarnLevel(void) {
    return WarnLevel;
}
void
dscuda::setWarnLevel(int level) {
    WarnLevel = level;
}
char*
dscudaMemcpyKindName(cudaMemcpyKind kind) {
    static char *name;

    switch (kind) {
      case cudaMemcpyHostToHost:
        name = "cudaMemcpyHostToHost";
        break;
      case cudaMemcpyHostToDevice:
        name = "cudaMemcpyHostToDevice";
        break;
      case cudaMemcpyDeviceToHost:
        name = "cudaMemcpyDeviceToHost";
        break;
      case cudaMemcpyDeviceToDevice:
        name = "cudaMemcpyDeviceToDevice";
        break;
      case cudaMemcpyDefault:
        name = "cudaMemcpyDefault";
        break;
      default:
        name = "Invalid cudaMemcpyKind";
    }
    return name;
}
const char*
dscudaGetIpaddrString(unsigned int addr) {
    static char buf[128];
    char *p = (char *)&addr;
    sprintf(buf, "%hhu.%hhu.%hhu.%hhu", p[0], p[1], p[2], p[3]);
    return buf;
}

/*
 *
 * t0 : time of day (in second) the last time this function is called.
 * returns the number of seconds passed since *t0.
 */
double
RCgetCputime(double *t0) {
    struct timeval t;
    double tnow, dt;

    gettimeofday(&t, NULL);
    tnow = t.tv_sec + t.tv_usec/1000000.0;
    dt = tnow - *t0;
    *t0 = tnow;
    return dt;
}
//--
//--
//--
int
dscuda::sprintfDate(char *s, int fmt) {
    time_t now;
    struct tm *local;
    //--
    now   = time(NULL);
    local = localtime(&now);
    switch (fmt) {
    case 0: // "MMDD_hhmmss"
	strftime(s, 32, "%m%d_%H%M%S", local);
	break;
    case 1:
	strftime(s, 32, "%H%M%S", local);
	break;
    default:
	fprintf(stderr, "%s():error undefined switch branch.\n");
	exit(1);
    }
    return 0;
}
void*
dscuda::xmalloc(size_t size) {
    void *p;
    size_t sz=size;
    if (size == 0) {
	sz=1;
    }
    p = malloc( size ) ;
    if (p == NULL) {
	fprintf(stderr, "dscuda:xmalloc() out of memory. requested size was %d bytes.\n", size);
	exit(1);
    }
    return p;
}

void
dscuda::xfree(void *p) {
    if (p == NULL) {
	fprintf(stderr, "xfree() called with NULL. \n");
	exit(1);
    }
    free(p);
    p = NULL;
}

