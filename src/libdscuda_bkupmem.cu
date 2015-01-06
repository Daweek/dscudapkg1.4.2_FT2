//                             -*- Mode: C++ -*-
// Filename         : libdscuda_bkupmem.h
// Description      : DS-CUDA verb function.
// Author           : A.Kawai, K.Yoshikawa, T.Narumi
// Created On       : 2011-01-01 00:00:00
// Last Modified By : M.Oikawa
// Last Modified On : 2014-09-07 17:09:34
// Update Count     : 0.1
// Status           : Unknown, Use with caution!
//--------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "dscudarpc.h"
#include "dscudadefs.h"
#include "dscudautil.h"
#include "libdscuda.h"

/*
 * Constructor of "Bkupmem" class.
 */
BkupMem::BkupMem(void) {
    v_region      = NULL;
    d_region      = NULL;
    h_region      = NULL;
    size          = 0;
    update_rdy    = 0;
}
/*
 * Is this BkupMem is the "head" of list?
 */
bool
BkupMem::isHead(void) {
    if (prev == NULL) return true;
    else              return false;
}
/*
 * Is this BkupMem is the "tail" of list?
 */
bool
BkupMem::isTail(void) {
    if (next == NULL) return true;
    else              return false;
}
void*
BkupMem::translateAddrVtoD(const void *v_ptr) {
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
void*
BkupMem::translateAddrVtoH(const void *v_ptr) {
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
void
BkupMem::init( void *uva_ptr, void *d_ptr, int sz) {
    v_region   = uva_ptr;
    d_region   = d_ptr;
    h_region = (void *)dscuda::xmalloc(sz);
    
#if 0
    WARN(10, "%s():v_region=%p, d_region=%p, h_region=%p\n",
	 __func__, v_region, d_region, h_region);
#endif
    
    size       = sz;
    update_rdy = 0;

    prev = NULL;
    next = NULL;
}
int
BkupMem::calcSum(void) {
    int sum=0;
    int *ptr = (int *)h_region;

    for (int i=0; i<size; i+=sizeof(int)) {
	sum += *ptr;
	ptr++;
    }
    return sum;
}
//========================================================================
// Constuctor of "BkupMemList" class.
//
BkupMemList::BkupMemList(void) {
    int   autoverb;
    head       = NULL;
    tail       = NULL;
    length     = 0;
    total_size = 0;
    //WARN( 5, "The constructor %s() called.\n", __func__);
}

/*
 * Destructor of "BkupMemList" class.
 */
BkupMemList::~BkupMemList(void) {

}
void
BkupMemList::print(void) {
    BkupMem *mem_ptr = head;
    int i=0;

    while (mem_ptr != NULL) {
	WARN(0, "BkupMemList[%d]: v= %p, d= %p, h=%p\n",
	     i, mem_ptr->v_region, mem_ptr->d_region, mem_ptr->h_region);
	mem_ptr = mem_ptr->next;
	i++;
    }
}
BkupMem*
BkupMemList::headPtr(void) {
    return head;
}
int
BkupMemList::getLen(void) {
    return length;
}
long
BkupMemList::getTotalSize(void) {
    return total_size;
}
bool
BkupMemList::isEmpty(void) {
    if      ( head==NULL && tail==NULL ) return true;
    else if ( head!=NULL && tail!=NULL ) return false;
    else {
	fprintf( stderr, "Unexpected error in %s().\n", __func__ );
	exit(EXIT_FAILURE);
    }
}
int
BkupMemList::countRegion( void ) {
    BkupMem *mem = head;
    int count = 0;
    while ( mem != NULL ) {
	mem = mem->next;
	count++;
    }
    return count;
}

int
BkupMemList::checkSumRegion( void *targ, int size ) {
    int sum=0;
    int  *ptr = (int *)targ;
    
    for (int s=0; s < size; s+=sizeof(int)) {
	sum += *ptr;
	ptr++;
    }
    return sum;
}
/* Class: "BkupMemList"
 * Method: query()
 *
 */
BkupMem*
BkupMemList::query(void *uva_ptr) {
    BkupMem *mem = head;
    int i = 0;
    while (mem != NULL) { // Search the target from head to tail in the list.
	if (mem->v_region == uva_ptr) { /* tagged by its address on GPU */
	    //WARN(10, "---> %s(%p): return %p\n", __func__, uva_ptr, mem);
	    return mem;
	}
	WARN(10, "%s(): search %p, check[%d]= %p\n", __func__, uva_ptr, i, mem->v_region);
	mem = mem->next;
	i++;
    }
    return NULL;
}
/*
 * Add the BkupMem cell at the tail of List.
 */
void
BkupMemList::append(void *uva_ptr, void *d_ptr, int size) {
    BkupMem *mem;
    
    mem = (BkupMem *)dscuda::xmalloc( sizeof(BkupMem) );
    
    mem->init(uva_ptr, d_ptr, size);

    if (isEmpty()) { // add as head cell.
	head = mem;
	mem->prev = NULL;
    } else { // cat to tail cell.
	tail->next = mem;
	mem->prev = tail;
    }
    tail = mem;  // Update tail poiter.
    length++;
    total_size += size;
    if (getLen() < 0) {
	WARN(0, "(+_+) Unexpected error in %s()\n", __func__);
	exit(EXIT_FAILURE);
    }
}
/*
 * Class: "BkupMemlist"
 * Method: removeRegion()
 */
void
BkupMemList::remove(void *uva_ptr) {
    BkupMem *mem = query(uva_ptr);
    
    if (mem == NULL) { // not found.
	WARN(0, "%s(): not found requested memory region.\n", __func__);
	WARN(0, "mem. list length= %d \n", countRegion());
	this->print();
	return;
    }
    else if (mem->isHead()) { // remove head cell.
	head = mem->next;
	if (head != NULL) {
	    head->prev = NULL;
	}
    }
    else if (mem->isTail()) { // remove tail cell.
	tail = mem->prev;
    }
    else { // remove a intermediate cell.
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

void*
BkupMemList::queryHostPtr(const void *v_ptr) {
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

void*
BkupMemList::queryDevicePtr(const void *v_ptr) {
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
BkupMemList::restructDeviceRegion(void) {
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

