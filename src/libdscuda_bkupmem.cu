//                             -*- Mode: C++ -*-
// Filename         : libdscuda_bkupmem.h
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-06 13:20:20
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//--------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include "dscudarpc.h"
#include "dscudadefs.h"
#include "dscudamacros.h"
#include "libdscuda.h"

/*
 * Constructor of "Bkupmem_t" class.
 */
BkupMem_t::BkupMem_t(void) {
    v_region      = NULL;
    d_region      = NULL;
    h_region      = NULL;
    size          = 0;
    update_rdy    = 0;
}
/*
 * Is this BkupMem_t is the "head" of list?
 */
int BkupMem_t::isHead(void) {
    if (prev == NULL) { return 1; }
    else              { return 0; }
}
/*
 * Is this BkupMem_t is the "tail" of list?
 */
int BkupMem_t::isTail( void ) {
    if (next == NULL) { return 1; }
    else              { return 0; }
}
void *BkupMem_t::translateAddrVtoD(const void *v_ptr) {
    char *v_cptr       = (char *)v_ptr;
    char *v_region_end = (char *)v_region + size;
    long  d_offset;
    char *d_ret;
    if ((v_cptr >= v_region) && (v_cptr < v_region_end)) {
	d_offset = v_cptr - (char *)v_region;
	d_ret = (char *)d_region + d_offset;
	return (void *)d_ret;
    } else {
	return NULL;
    }
}
void *BkupMem_t::translateAddrVtoH(const void *v_ptr) {
    char *v_cptr       = (char *)v_ptr;
    char *v_region_end = (char *)v_region + size;
    long  h_offset;
    char *h_ret;
    if ((v_cptr >= v_region) && (v_cptr < v_region_end)) {
	h_offset = v_cptr - (char *)v_region;
	h_ret = (char *)h_region + h_offset;
	return (void *)h_ret;
    } else {
	return NULL;
    }
}
/*
 * 
 */
void BkupMem_t::init( void *uva_ptr, void *d_ptr, int sz) {
    v_region   = uva_ptr;
    d_region   = d_ptr;
    h_region = (void *)malloc( sz );
    if (h_region == NULL) {
	perror("BkupMem_t.init()");
	exit(EXIT_FAILURE);
    }
    WARN(10, "%s():v_region=%p, d_region=%p, h_region=%p\n",
	 __func__, v_region, d_region, h_region);

    size       = sz;
    update_rdy = 0;

    prev = NULL;
    next = NULL;
}
int BkupMem_t::calcSum(void) {
    int sum=0;
    int *ptr = (int *)h_region;

    for (int i=0; i<size; i+=sizeof(int)) {
	sum += *ptr;
	ptr++;
    }
    return sum;
}
//========================================================================
/*
 * Constuctor of "BkupMemList_t" class.
 */
BkupMemList_t::BkupMemList_t(void) {
    int   autoverb;

    head       = NULL;
    tail       = NULL;
    length     = 0;
    total_size = 0;
    //WARN( 5, "The constructor %s() called.\n", __func__);
}

/*
 * Destructor of "BkupMemList_t" class.
 */
BkupMemList_t::~BkupMemList_t(void) {

}

void BkupMemList_t::print(void) {
    BkupMem *mem_ptr = head;
    int i=0;

    while (mem_ptr != NULL) {
	WARN(0, "BkupMemList[%d]: v= %p, d= %p, h=%p\n",
	     i, mem_ptr->v_region, mem_ptr->d_region, mem_ptr->h_region);
	mem_ptr = mem_ptr->next;
	i++;
    }
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
 * Method: query()
 *
 */
BkupMem*
BkupMemList_t::query(void *uva_ptr) {
    BkupMem *mem = head;
    int i = 0;
    while (mem != NULL) { // Search the target from head to tail in the list.
	if (mem->v_region == uva_ptr) { /* tagged by its address on GPU */
	    WARN(10, "---> %s(%p): return %p\n", __func__, uva_ptr, mem);
	    return mem;
	}
	WARN(10, "%s(): search %p, check[%d]= %p\n", __func__, uva_ptr, i, mem->v_region);
	mem = mem->next;
	i++;
    }
    return NULL;
}
/*
 * Add the BkupMem_t cell at the tail of List.
 */
void BkupMemList_t::add(void *uva_ptr, void *d_ptr, int size) {
    BkupMem *mem;
    
    mem = (BkupMem *)malloc( sizeof(BkupMem) );
    if (mem == NULL) {
	perror( "BkupMemLit_t::add()");
	exit(EXIT_FAILURE);
    }
    
    mem->init(uva_ptr, d_ptr, size);

    if (isEmpty()) { // add as head cell.
	head = mem;
	mem->prev = NULL;
    } else { // cat to tail cell.
	tail->next = mem;
	mem->prev = tail;
    }
    tail = mem;
    length++;
    total_size += size;

    WARN(5, "      add BkupMemList[%d]: uva_ptr=%p d_ptr=%p, size=%d\n",
	 length - 1, uva_ptr, d_ptr, size);
    if (getLen() < 0) {
	WARN(0, "(+_+) Unexpected error in %s()\n", __func__);
	exit(EXIT_FAILURE);
    }
}
/*
 * Class: "BkupMemlist_t"
 * Method: removeRegion()
 */
void BkupMemList_t::remove(void *uva_ptr) {
    BkupMem *mem = query(uva_ptr);
    
    if (mem == NULL) { // not found.
	WARN(0, "%s(): not found requested memory region.\n", __func__);
	WARN(0, "mem. list length= %d \n", countRegion());
	this->print();
	return;
    } else if (mem->isHead()) { // remove head cell.
	head = mem->next;
	if (head != NULL) {
	    head->prev = NULL;
	}
    } else if (mem->isTail()) { // remove tail cell.
	tail = mem->prev;
    } else { // remove a intermediate cell.
	mem->prev->next = mem->next;
    }

    // delete removed cell.
    total_size -= mem->size;    
    free(mem->h_region);
    free(mem);
    length--;
    if (getLen() < 0) {
	fprintf(stderr, "(+_+) Unexpected error in %s()\n", __func__);
	exit(EXIT_FAILURE);
    }
}

void *BkupMemList_t::queryHostPtr(const void *v_ptr) {
    BkupMem *mem = head;
    void *h_ptr = NULL;
    int   i = 0;
    
    while (mem) { /* Search */
	h_ptr = mem->translateAddrVtoH(v_ptr);
	if (h_ptr != NULL) {
	    return h_ptr;
	}
	mem = mem->next;
	i++;
    }
    WARN(0, "%s():not found host pointer.\n", __func__);
    return NULL;
}

void *BkupMemList_t::queryDevicePtr(const void *v_ptr) {
    BkupMem *mem = head;
    void *d_ptr = NULL;
    int   i = 0;
    
    while (mem) { /* Search */
	d_ptr = mem->translateAddrVtoD(v_ptr);
	if (d_ptr != NULL) {
	    return d_ptr;
	}
	mem = mem->next;
	i++;
    }
    WARN(0, "%s():not found device pointer.\n", __func__);
    return NULL;
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
	     copy_count++, mem->d_region, mem->h_region, mem->size, mem->calcSum());
	//mem->restoreSafeRegion();
	mem = mem->next;
    }
    St.setAutoVerb( verb );
    WARN(2, "+--- done.\n");
}

