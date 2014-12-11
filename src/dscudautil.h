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
int         sprintfDate(char *s, int fmt=0);
int         dscudaWarnLevel(void);
void        dscudaSetWarnLevel(int level);
char       *dscudaMemcpyKindName(cudaMemcpyKind kind);
const char *dscudaGetIpaddrString(unsigned int addr);
double      RCgetCputime(double *t0);
#endif // DSCUDAUTIL_H
