# ARCH = $(shell uname -m)
# ifeq ($(ARCH), x86_64)

ifeq ($(shell uname -m), x86_64)
	LP64 = 1
else
	LP64 = 0
endif

GCCVER = 4.4
RPCGEN        = rpcgen -N -D__LP64__=$(LP64)
CC           ?= /usr/bin/gcc-$(GCCVER)
CFLAGS       ?= -O0 -g -ffast-math -funroll-loops
CXX          ?= /usr/bin/g++-$(GCCVER)
CXXFLAGS     ?= -O0 -g -ffast-math -funroll-loops
CUDAPATH     ?= /usr/local/cuda
NVCC         ?= $(CUDAPATH)/bin/nvcc -O0 -g -arch=sm_20 --compiler-bindir=g++-$(GCCVER)
CUDASDKPATH  ?= /share02/m_oikawa/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES  = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
CUDALIBS      = -L. -L$(CUDAPATH)/lib64

TARGET         = dscudad dscudasvr dscudasvr_fault libdscuda_rpc.a libcudart.so.3
TARGET_RPCONLY = dscudad_rpc libdscuda_rpc.a libcudart.so.3
DSCUDAIF       = dscudarpc
CLIENTBASE     = libdscuda.o

default :	$(TARGET)

rpconly:	$(TARGET_RPCONLY)

libdscuda_rpc.so :	$(CLIENTBASE) libdscuda_rpc.o $(DSCUDAIF)_clnt.o \
                        $(DSCUDAIF)_xdr.o dscudautil.o sockutil.o dscudaverb.o
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ $^

libdscuda_rpc.a :	$(CLIENTBASE) libdscuda_rpc.o $(DSCUDAIF)_clnt.o \
                        $(DSCUDAIF)_xdr.o dscudautil.o sockutil.o dscudaverb.o
	ar rvu $@ $^
	ranlib $@

libdscuda_rpc.o  : libdscuda_rpc.cu
	$(NVCC) -DRPC_ONLY=1 $(CUDAINCLUDES) -c -o $@ $<

$(CLIENTBASE) : libdscuda.cu
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

# A dummy library to override CUDA runtime library functions.
libcudart.so.3:	cudart_dummy.cu
	$(NVCC) $(CUDAINCLUDES) -shared -Xcompiler -fPIC -o $@ $<

dscudaverb.o:	dscudaverb.cu
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -c -o $@ $<

libdscuda_bkupmem.o:  libdscuda_bkupmem.cu libdscuda.h
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

libdscuda_histrec.o:  libdscuda_histrec.cu libdscuda.h
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

dscudad :	dscudad.cu sockutil.c
	$(NVCC) -o $@ $(CUDAINCLUDES) $^ -lpthread

dscudad_rpc :	dscudad.cu sockutil.c
	$(NVCC) -o $@ $(CUDAINCLUDES) -DRPC_ONLY=1 $^ -lpthread

dummysvr :	dummysvr.c

dummyclient :	dummyclient.c

#
# dscudasvr series
# 
dscudasvr :	   dscudasvr.o $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o sockutil.o
	$(CXX) $(CUDALIBS) -lcufft -lcudart -lcuda -lpthread -o $@ $^

dscudasvr.o:       dscudasvr.cu dscudasvr_rpc.cu dscuda.h dscudarpc.h sockutil.h  
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -c -o $@ $<

dscudasvr_fault :  dscudasvr_fault.o $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o sockutil.o
	$(CXX) $(CUDALIBS) -lcufft -lcudart -lcuda -lpthread -o $@ $^

dscudasvr_fault.o: dscudasvr.cu dscudasvr_rpc.cu dscuda.h dscudarpc.h sockutil.h  
	$(NVCC) -c $(CUDAINCLUDES) -DRPC_ONLY=1 -DFAULT_AT_D2H=1 -o $@ $<

#
#
#
dscudatest :	              dscudatest.cu libdscuda_ibv.so
	$(NVCC) $(CUDAINCLUDES) -o $@ $< -L../lib -ldscuda_ibv -lrdmacm -libverbs -lpthread

$(DSCUDAIF)_clnt.o:	$(DSCUDAIF)_clnt.c
	$(CXX) $(CUDAINCLUDES) -c -o $@ $<

$(DSCUDAIF)_svc.o:	$(DSCUDAIF)_svc.c 
	$(CXX) $(CUDAINCLUDES) -c -o $@ $<

$(DSCUDAIF)_xdr.o:	$(DSCUDAIF)_xdr.c
	$(CXX) $(CUDAINCLUDES) -c -o $@ $<

dscudautil.o:	dscudautil.cu dscudautil.h
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

#ibv_rdma.o:	ibv_rdma.cpp
#	$(CXX) $(CUDAINCLUDES) -c -o $@ $<
###
### source file include dependancy.
###
libdscuda.cu:      dscuda.h libdscuda.h dscudaverb.h
libdscuda_rpc.cu:  dscuda.h libdscuda.h dscudaverb.h
libdscuda_ibv.cu:  dscuda.h libdscuda.h dscudaverb.h ibv_rdma.h
dscudad.cu :       dscudadefs.h sockutil.h dscudarpc.h
dscudaverb.cu:     dscuda.h libdscuda.h dscudaverb.h dscudarpc.h
dscudatest.cu:     dscuda.h ibv_rdma.h
ibv_rdma.cpp:      ibv_rdma.h 
ibv_rdma.h:        dscudadefs.h dscudarpc.h dscudamacros.h
dscuda.h:          dscudadefs.h dscudarpc.h dscudamacros.h ibv_rdma.h
###
### RPCGEN
###
$(DSCUDAIF)_clnt.c :	$(DSCUDAIF).x $(DSCUDAIF).h
	$(RPCGEN) -l $< > $@
$(DSCUDAIF)_svc.c:	$(DSCUDAIF).x $(DSCUDAIF).h
	$(RPCGEN) -m $< > $@
#	$(RPCGEN) -s tcp -s udp $< > $@
$(DSCUDAIF)_xdr.c:	$(DSCUDAIF).x $(DSCUDAIF).h
	$(RPCGEN) -c $< > $@
$(DSCUDAIF).h:		$(DSCUDAIF).x
	$(RPCGEN) -h $< > $@
$(DSCUDAIF).x: dscudadefs.h
###
### Phony target
###
.PHONY: clean install

install:
	ln -fs ../src/dscuda.h       ../include/
	ln -fs ../src/dscudadefs.h   ../include/
	ln -fs ../src/dscudamacros.h ../include/
	ln -fs ../src/dscudarpc.h    ../include/
	ln -fs ../src/dscudaverb.h   ../include/
	ln -fs ../src/dscudautil.h   ../include/
	ln -fs ../src/libdscuda.h    ../include/
	ln -fs ../src/sockutil.h     ../include/
	ln -fs ../src/dscudad        ../bin/
	ln -fs ../src/dscudasvr      ../bin/
	ln -fs ../src/dscudasvr_fault  ../bin/
	ln -fs ../src/libdscuda_rpc.a ../lib/
clean:
	/bin/rm $(TARGET) $(TARGET_RPCONLY) $(DSCUDAIF).h $(DSCUDAIF)_svc.* $(DSCUDAIF)_clnt.* $(DSCUDAIF)_xdr.* *.o *~ *.linkinfo
