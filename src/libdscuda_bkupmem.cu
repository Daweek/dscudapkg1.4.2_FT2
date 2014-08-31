//                             -*- Mode: C++ -*-
// Filename         : libdscuda_bkupmem.h
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-08-31 10:32:00
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include "dscudadefs.h"
#include "dscudamacros.h"
#include "libdscuda_bkupmem.h"

/*
 * Constructor of "Bkupmem_t" class.
 */
BkupMem_t::BkupMem_t( void ) {
    v_region      = NULL;
    d_region      = NULL;
    h_region      = NULL;
    size          = 0;
    update_rdy    = 0;
}
/*
 * Is this BkupMem_t is the "head" of list?
 */
int BkupMem_t::isHead( void ) {
    if ( prev == NULL ) { return 1; }
    else                { return 0; }
}
/*
 * Is this BkupMem_t is the "tail" of list?
 */
int BkupMem_t::isTail( void ) {
    if ( next == NULL ) { return 1; }
    else                { return 0; }
}
/*
 * 
 */
void BkupMem_t::init( void *uva_ptr,   // Device pointer in UVA.
		      int   isize )    // Size of region in Byte.
{
    d_region   = uva_ptr;
    size       = isize;
    update_rdy = 0;
    
    h_region_tmp = (void *)malloc( isize );
    if ( h_region_tmp == NULL ) {
	perror("BkupMem_t.init()");
	exit( EXIT_FAILURE );
    }

    prev = NULL;
    next = NULL;
}

//========================================================================
/*
 * Constuctor of "BkupMemList_t" class.
 */
BkupMemList_t::BkupMemList_t(void) {
    char *env;
    int   autoverb;

    head       = NULL;
    tail       = NULL;
    length     = 0;
    total_size = 0;
    WARN( 5, "The constructor %s() called.\n", __func__);
}

/*
 * Destructor of "BkupMemList_t" class.
 */
BkupMemList_t::~BkupMemList_t(void) {
    pthread_cancel( tid );
}

int BkupMemList_t::getLen(void) {
    return length;
}

long BkupMemList_t::getTotalSize(void) {
    return total_size;
}

int BkupMemList_t::isEmpty( void ) {
    if      ( head==NULL && tail==NULL ) return 1;
    else if ( head!=NULL && tail!=NULL ) return 0;
    else {
	fprintf( stderr, "Unexpected error in %s().\n", __func__ );
	exit( EXIT_FAILURE );
    }
}

int BkupMemList_t::countRegion( void ) {
    BkupMem *mem = head;
    int count = 0;
    while ( mem != NULL ) {
	mem = mem->next;
	count++;
    }
    return count;
}

int
BkupMemList_t::checkSumRegion( void *targ, int size ) {
    int sum=0;
    int  *ptr = (int *)targ;
    
    for (int s=0; s < size; s+=sizeof(int)) {
	sum += *ptr;
	ptr++;
    }
    return sum;
}
/* Class: "BkupMemList_t"
 * Method: queryRegion()
 *
 */
BkupMem*
BkupMemList_t::queryRegion( void *dst ) {
    BkupMem *mem = head;
    int i = 0;
    while ( mem != NULL ) { /* Search */
	if ( mem->d_region == dst ) { /* tagged by its address on GPU */
	    WARN(10, "---> %s(%p): return %p\n", __func__, dst, mem);
	    return mem;
	}
	WARN(10, "%s(): search %p, check[%d]= %p\n", __func__, dst, i, mem->d_region);
	mem = mem->next;
	i++;
    }
    return NULL;
}
/*
 * Add the BkupMem_t cell at the tail of List.
 */
void BkupMemList_t::add( void *dst, int size )
{
    BkupMem *mem;
    
    mem = (BkupMem *)malloc( sizeof(BkupMem) );
    if ( mem == NULL ) {
	perror( "BkupMemLit_t::add()");
	exit( EXIT_FAILURE );
    }
    
    mem->init( dst, size );

    if ( isEmpty() ) {
	head = mem;
	mem->prev = NULL;
    } else {
	tail->next = mem;
	mem->prev = tail;
    }
    tail = mem;
    length++;
    total_size += size;

    WARN( 5, "+--- add BkupMemList[%d]: p_dev=%p, size=%d\n", length - 1, dst, size );
    if ( getLen() < 0 ) {
	fprintf( stderr, "(+_+) Unexpected error in %s()\n", __func__ );
	exit( EXIT_FAILURE );
    }
}
/*
 * Class: "BkupMemlist_t"
 * Method: removeRegion()
 */
void
BkupMemList_t::remove( void *dst )
{
    BkupMem *mem = queryRegion(dst);
    BkupMem *p_list = head;
    int i;
    if ( mem == NULL ) {
	WARN(0, "%s(): not found requested memory region.\n", __func__);
	WARN(0, "mem. list length= %d \n", countRegion());
	i = 0;
	while ( p_list != NULL ) {
	    WARN(0, "mem-list[%d] = %p\n", i, p_list->d_region);
	    p_list = p_list->next;
	    i++;
	}
	return;
    } else if ( mem->isHead() ) { // remove head, begin with 2nd.
	head = mem->next;
	if ( head != NULL ) {
	    head->prev = NULL;
	}
    } else if ( mem->isTail() ) {
	tail = mem->prev;
    } else {
	mem->prev->next = mem->next;
    }
    
    total_size -= mem->size;    
    free( mem->h_region_tmp );
    free( mem );
    length--;
    if ( getLen() < 0 ) {
	fprintf( stderr, "(+_+) Unexpected error in %s()\n", __func__ );
	exit( EXIT_FAILURE );
    }
}

void*
BkupMemList_t::searchUpdateRegion(void *dst) {
    BkupMem *mem = head;
    char *d_targ  = (char *)dst;
    char *d_begin;
    char *h_begin;
    char *h_p     = NULL;
    int   i = 0;
    
    while (mem) { /* Search */
	d_begin = (char *)mem->d_region;
	h_begin = (char *)mem->h_region_tmp;
	
	if (d_targ >= d_begin &&
	    d_targ < (d_begin + mem->size)) {
	    h_p = h_begin + (d_targ - d_begin);
	    break;
	}
	mem = mem->next;
	i++;
    }
    return (void *)h_p;
}

void
BkupMemList_t::updateRegion( void *dst, void *src, int size ) {
// dst : GPU device memory region
// src : HOST memory region
    BkupMem *mem;
    void    *src_mirrored;
    
    if ( src == NULL ) {
	WARN(0, "(+_+) not found backup target memory region (%p).\n", dst);
	exit(1);
    } else {
	//mem = BkupMem.queryRegion(dst);
	//src_mirrored = mem->src;
	src_mirrored = searchUpdateRegion(dst);
	memcpy(src_mirrored, src, size); // update historical memory region.
	WARN(3, "+--- Also copied to backup region (%p), checksum=%d.\n",
	     dst, checkSumRegion(src, size));
    }
}


void
BkupMemList_t::reallocDeviceRegion(RCServer_t *svr) {
    BkupMem *mem = head;
    int     verb = St.isAutoVerb();
    int     copy_count = 0;
    int     i = 0;
    
    WARN(1, "%s(RCServer_t *sp).\n", __func__);
    WARN(1, "Num. of realloc region = %d\n", BKUPMEM.length );
    St.unsetAutoVerb();
    while ( mem != NULL ) {
	/* TODO: select migrateded virtual device, not all region. */
	WARN(5, "mem[%d]->dst = %p, size= %d\n", i, mem->d_regio, mem->size);
	dscudaVerbMalloc(&mem->d_region, mem->size, svr);
	mem = mem->next;
	i++;
    }
    St.setAutoVerb(verb);
    WARN(1, "+--- Done.\n");
}
/* 
 * Resore the all data of a GPU device with backup data on client node.
 */
void
BkupMemList_t::restructDeviceRegion(void) {
    BkupMem *mem = head;
    int      verb = St.isAutoVerb();
    int      copy_count = 0;
    unsigned char    *mon;
    float            *fmon;
    int              *imon;

    WARN(2, "%s(void).\n", __func__);
    St.unsetAutoVerb();
    while (mem != NULL) {
	WARN(1, "###   + region[%d] (dst=%p, src=%p, size=%d) . checksum=0x%08x\n",
	     copy_count++, mem->d_region, mem->h_region_tmp, mem->size, checkSum(mem->h_region_tmp, mem->size));
	mem->restoreSafeRegion();
	mem = mem->next;
    }
    St.setAutoVerb( verb );
    WARN(2, "+--- done.\n");
}

