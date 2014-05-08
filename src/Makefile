# ARCH = $(shell uname -m)
# ifeq ($(ARCH), x86_64)

ifeq ($(shell uname -m), x86_64)
	LP64 = 1
else
	LP64 = 0
endif

RPCGEN        = rpcgen -N -D__LP64__=$(LP64)
CC            = gcc -O0 -g -ffast-math -funroll-loops
CXX           = g++ -O0 -g -ffast-math -funroll-loops
CUDAPATH     ?= /usr/local/cuda
NVCC          = $(CUDAPATH)/bin/nvcc -O0 -g -arch=sm_30
CUDASDKPATH  ?= /share02/m_oikawa/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES  = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
CUDALIBS      = -L. -L$(CUDAPATH)/lib64

TARGET         = dscudad dscudasvr libdscuda_ibv.a libdscuda_rpc.a libcudart.so.3
TARGET_RPCONLY = dscudad_rpc dscudasvr_rpc libdscuda_rpc.a libcudart.so.3
DSCUDAIF       = dscudarpc
CLIENTBASE     = libdscuda.o

default :	$(TARGET)

rpconly:	$(TARGET_RPCONLY)

libdscuda_ibv.so :	$(CLIENTBASE) libdscuda_ibv.o $(DSCUDAIF)_clnt.o \
                        dscudautil.o ibv_rdma.o sockutil.o dscudaverb.o
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ $^

libdscuda_rpc.so :	$(CLIENTBASE) libdscuda_rpc.o $(DSCUDAIF)_clnt.o \
                        $(DSCUDAIF)_xdr.o dscudautil.o sockutil.o dscudaverb.o
	$(CXX) -shared -Wl,-soname,libdscuda.so -o $@ $^

libdscuda_ibv.a :	$(CLIENTBASE) libdscuda_ibv.o $(DSCUDAIF)_clnt.o \
                        dscudautil.o ibv_rdma.o sockutil.o dscudaverb.o
	ar rvu $@ $^
	ranlib $@

libdscuda_rpc.a :	$(CLIENTBASE) libdscuda_rpc.o $(DSCUDAIF)_clnt.o \
                        $(DSCUDAIF)_xdr.o dscudautil.o sockutil.o dscudaverb.o
	ar rvu $@ $^
	ranlib $@

libdscuda_ibv.o  : libdscuda_ibv.cu
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

libdscuda_rpc.o  : libdscuda_rpc.cu
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

$(CLIENTBASE) : libdscuda.cu
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

# A dummy library to override CUDA runtime library functions.
libcudart.so.3:	cudart_dummy.cu
	$(NVCC) $(CUDAINCLUDES) -shared -Xcompiler -fPIC -o $@ $<

dscudaverb.o:	dscudaverb.cu
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -c -o $@ $<

dscudad :	dscudad.c sockutil.c sockutil.h
	$(CC) -o $@ dscudad.c sockutil.c -lpthread

dscudad_rpc :	dscudad.c sockutil.c sockutil.h
	$(CC) -DRPC_ONLY=1 -o $@ dscudad.c sockutil.c -lpthread

dummysvr :	dummysvr.c

dummyclient :	dummyclient.c

dscudasvr :	dscudasvr.o $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o ibv_rdma.o sockutil.o
	echo "# <--- Link object files."
	$(CXX) $(CUDALIBS) -lcufft -lcudart -lcuda -lrdmacm -libverbs -lpthread -o $@ $^

dscudasvr_rpc :	dscudasvr_rpc_only.o $(DSCUDAIF)_svc.o $(DSCUDAIF)_xdr.o dscudautil.o sockutil.o
	$(CXX) $(CUDALIBS) -lcufft -lcudart -lcuda -lpthread -o $@ $^

dscudasvr.o: dscudasvr.cu dscudasvr_rpc.cu dscudasvr_ibv.cu dscuda.h \
             ibv_rdma.h dscudarpc.h sockutil.h  
	$(NVCC) $(CUDAINCLUDES) -c -o $@ $<

dscudasvr_rpc_only.o: dscudasvr.cu dscudasvr_rpc.cu dscudasvr_ibv.cu dscuda.h \
                      ibv_rdma.h dscudarpc.h sockutil.h  
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -c -o $@ $<

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

ibv_rdma.o:	ibv_rdma.cpp
	$(CXX) $(CUDAINCLUDES) -c -o $@ $<
###
### source file include dependancy.
###
libdscuda.cu:      dscuda.h libdscuda.h dscudaverb.h
libdscuda_rpc.cu:  dscuda.h libdscuda.h dscudaverb.h
libdscuda_ibv.cu:  dscuda.h libdscuda.h dscudaverb.h ibv_rdma.h
dscudad.c :        dscudadefs.h sockutil.h
dscudaverb.cu:     dscuda.h libdscuda.h dscudaverb.h dscudarpc.h
dscudatest.cu:     dscuda.h ibv_rdma.h
ibv_rdma.cpp:      ibv_rdma.h 
ibv_rdma.h:        dscudadefs.h dscudarpc.h dscudamacros.h
dscuda.h:          dscudadefs.h dscudarpc.h dscudamacros.h ibv_rdma.h

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
.PHONY: clean
clean:
	/bin/rm $(TARGET) $(TARGET_RPCONLY) $(DSCUDAIF).h $(DSCUDAIF)_svc.* $(DSCUDAIF)_clnt.* $(DSCUDAIF)_xdr.* *.o *~ *.linkinfo
