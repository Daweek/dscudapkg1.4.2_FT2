//                             -*- Mode: C++ -*-
// Filename         : libdscuda_bkupmem.h
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-08-26 09:51:03
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#ifndef __DSCUDA_BKUPMEM_H__
#define __DSCUDA_BKUPMEM_H__
#include <pthread.h>
#include "libdscuda.h"
/*
 * Breif:
 *    Backup memory region of devices allocated by cudaMemcpy().
 * Description:
 *    mirroring of a global memory region to client memory region.
 *    In case when device memory region was corrupted, restore with
 *    clean data to device memory.
 *    In case when using device don't response from client request,
 *    migrate to another device and restore with clean data.
 */
typedef struct BkupMem_t
{
    void  *d_region;        // server device memory space (UVA).
    void  *h_region;        //
    int    size;            // in Byte.
    int    update_rdy;      // 1:"*dst" has valid data, 0:invalid.
    struct BkupMem_t *next; // For double-linked-list prev.
    struct BkupMem_t *prev; // For double-linked-list next.
    //--- methods
    void   init( void *uva_ptr, int isize );
    int    isHead( void );
    int    isTail( void );
    void   updateSafeRegion( void );
    void   restoreSafeRegion( void );
    /*constructor/destructor.*/
    BkupMem_t( void );
} BkupMem;

typedef struct BkupMemList_t
{
private:
    pthread_t tid;        /* thread ID of Checkpointing */
    static void* periodicCheckpoint( void *arg );
public:
    BkupMem *head;        /* pointer to 1st  BkupMem */
    BkupMem *tail;        /* pointer to last BkupMem */
    int      length;       /* Counts of allocated memory region */
    long     total_size;   /* Total size of backuped memory in Byte */
    //--- construct/destruct
    BkupMemList_t(void);
    ~BkupMemList_t(void);
    //--- methods ---------------
    void     add(void *dst, int size); // verbAllocatedMemRegister()
    void     remove( void *dst );        // verbAllocatedMemUnregister()
    int      isEmpty(void);
    int      getLen(void);
    long     getTotalSize(void); // get total size of allocated memory.
    int      countRegion(void);
    int      checkSumRegion(void *targ, int size );
    BkupMem* queryRegion(void *dst );
    void*    searchUpdateRegion(void *dst );
    void     updateRegion(void *dst, void *src, int size );
    void     reallocDeviceRegion( RCServer_t *svr );  /* ReLoad backups */
    void     restructDeviceRegion(void);              /* ReLoad backups */
} BkupMemList;

#endif //__DSCUDA_BKUPMEM_H__
