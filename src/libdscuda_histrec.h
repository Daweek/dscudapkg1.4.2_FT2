//                             -*- Mode: C++ -*-
// Filename         : libdscuda_histrec.h
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-08-26 09:39:13
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef __LIBDSCUDA_HISTREC_H__
#define __LIBDSCUDA_HISTREC_H__

/*** 
 *** Each argument types and lists for historical recall.
 *** If you need to memorize another function into history, add new one.
 ***/
typedef struct HistRec_t {
    int      funcID;   // Recorded cuda*() function.
    void    *args;     // And its arguments.
    int      dev_id;   // The Device ID, set by last cudaSetDevice().
} HistRec;

typedef struct HistRecList_t {

    HistRec *histrec;
    int      length;    /* # of recorded function calls to be recalled */
    int      max_len;   /* Upper bound of "verbHistNum", extensible */
    // Constructor.
    HistRecord_t(void);
    //
    void add(int funcID, void *argp); /* Add */
    void clear(void);           /* Clear */
    void print(void);           /* Print to stdout */
    int  recall(void);          /* Recall */
private:
    static const int EXTEND_LEN = 32; // Size of growing of "max_len"
    int      recall_flag;
    void     setRecallFlag(void);
    void     clrRecallFlag(void);
    void     extendLen(void);

} HistRecList;

#endif //__LIBDSCUDA_HISTREC_H__
