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
CFLAGS       ?= -O0 -g 
CXX          ?= /usr/bin/g++-$(GCCVER)
CXXFLAGS     ?= -O0 -g -ffast-math -funroll-loops
CUDAPATH     ?= /usr/local/cuda
NVCC         ?= $(CUDAPATH)/bin/nvcc -O0 -g -arch=sm_20 --compiler-bindir=g++-$(GCCVER)
CUDASDKPATH  ?= /share02/m_oikawa/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES  = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
CUDALIBS      = -L. -L$(CUDAPATH)/lib64

TARGET        = dscudad dscudasvr dscudasvr_fault libdscuda_rpc.a libcudart.so.3

TARGET_RPCONLY = dscudad_rpc libdscuda_rpc.a libcudart.so.3

RPCBASE        = dscudarpc
CLIENTBASE     = libdscuda.o

### Build in Debug mode.
CDEBUG    = -Wall -O0 -g -gp -D_DEBUG
VPATH = ..
###
### Phony target
###
.PHONY: all 
all: $(TARGET)

.PHONY: debug_conf 
debug_conf :
	CFLAGS   += $(CDEBUG)
	CXXFLAGS += $(CDEBUG)

rpconly:	$(TARGET_RPCONLY)

LIBDSCUDA_OBJ = $(CLIENTBASE) libdscuda_rpc.o $(RPCBASE)_clnt.o \
                $(RPCBASE)_xdr.o dscudautil.o sockutil.o \
                libdscuda_bkupmem.o libdscuda_histrec.o
#----
libdscuda_rpc.so: $(LIBDSCUDA_OBJ)
	@echo "//************************"
	@echo "//**  libdscuda_rpc.so  ***"
	@echo "//************************"
	$(CXX) -O0 -g -shared -Wl,-soname,libdscuda.so -o $@ $^
#----
libdscuda_rpc.a: $(LIBDSCUDA_OBJ)
	ar rvu $@ $^
	ranlib $@
#
#---- "libdscuda_rpc.o"
libdscuda_rpc.o  : libdscuda_rpc.cu
	$(NVCC) -DRPC_ONLY=1 $(CUDAINCLUDES) -O0 -g -c -Xcompiler -fPIC -o $@ $<
#
#---- "libdscuda.o"
$(CLIENTBASE) : libdscuda.cu
	$(NVCC) $(CUDAINCLUDES) -O0 -g -c -Xcompiler -fPIC -o $@ $<

#---- A dummy library to override CUDA runtime library functions.
libcudart.so.3:	cudart_dummy.cu
	$(NVCC) $(CUDAINCLUDES) -O0 -g -shared -Xcompiler -fPIC -o $@ $<
#
#---- "fault torelant" routine.
libdscuda_bkupmem.o: libdscuda_bkupmem.cu libdscuda.h
	@echo "//***************************"
	@echo "//**  libdscuda_bkupmem.o  **"
	@echo "//***************************"		
	$(NVCC) $(CUDAINCLUDES) -c -O0 -g -Xcompiler -fPIC -o $@ $<
#
#---- "checkpoint" routine.
libdscuda_histrec.o:  libdscuda_histrec.cu libdscuda.h
	@echo "//***************************"
	@echo "//**  libdscuda_histrec.o  **"
	@echo "//***************************"		
	$(NVCC) $(CUDAINCLUDES) -c -O0 -g -Xcompiler -fPIC -o $@ $<
#---- "dscudad"
dscudad: dscudad.o sockutil.o dscudautil.o 
	@echo "//****************"
	@echo "//**  dscudad  ***"
	@echo "//****************"
	$(NVCC) -o $@ $(CUDAINCLUDES) $^ -lpthread
dscudad.o: dscudad.cu dscuda.h dscudadefs.h dscudautil.h sockutil.h
	$(NVCC) -c -o $@ $(CUDAINCLUDES) $<
#----
dscudad_rpc: dscudad.cu sockutil.o
	$(NVCC) -o $@ $(CUDAINCLUDES) -DRPC_ONLY=1 $^ -lpthread

dummysvr :	dummysvr.c

dummyclient :	dummyclient.c

###
### Generate "dscudasvr"
###
DSCUDASVR_OBJ = $(RPCBASE)_svc.o $(RPCBASE)_xdr.o dscudautil.o sockutil.o
dscudasvr: dscudasvr.o dscudasvr_rpc.o $(DSCUDASVR_OBJ)
	@echo "//******************"
	@echo "//**  dscudasvr  ***"
	@echo "//******************"
	$(CXX) $(CUDALIBS) -lcufft -lcudart -lcuda -lpthread -o $@ $^

dscudasvr_rpc.o: dscudasvr_rpc.cu dscudasvr_rpc.h
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -c -o $@ $<

dscudasvr.o: dscudasvr.cu dscuda.h dscudarpc.h sockutil.h
	@echo "//********************"
	@echo "//**  dscudasvr.o  ***"
	@echo "//********************"
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -c -o $@ $<
#---- FAULT server 
dscudasvr_fault:  dscudasvr_fault.o dscudasvr_rpc_fault.o $(DSCUDASVR_OBJ)
	$(CXX) $(CUDALIBS) -lcufft -lcudart -lcuda -lpthread -o $@ $^

dscudasvr_fault.o: dscudasvr.cu dscuda.h dscudarpc.h sockutil.h  
	$(NVCC) -c $(CUDAINCLUDES) -DRPC_ONLY=1 -DFAULT_AT_D2H=1 -o $@ $<

dscudasvr_rpc_fault.o: dscudasvr_rpc.cu dscudasvr_rpc.h
	$(NVCC) $(CUDAINCLUDES) -DRPC_ONLY=1 -DFAULT_AT_D2H=1 -c -o $@ $<

sockutil.o: sockutil.c sockutil.h
	$(CXX) -c $(CXXFLAGS) -fPIC -o $@ $<
###
### Generate object files for RPC.
###
dscudatest: dscudatest.cu libdscuda_ibv.so
	$(NVCC) $(CUDAINCLUDES) -o $@ $< -L../lib -ldscuda_ibv -lrdmacm -libverbs -lpthread
#---- "RPC partial set"
.PHONY: rpcbase
rpcbase: $(RPCBASE)_clnt.o $(RPCBASE)_svc.o $(RPCBASE)_xdr.o
#----
$(RPCBASE)_clnt.o: $(RPCBASE)_clnt.c
	$(CXX) $(CUDAINCLUDES) -c -fPIC -o $@ $<
#----
$(RPCBASE)_svc.o: $(RPCBASE)_svc.c
	$(CXX) $(CUDAINCLUDES) -c -fPIC -o $@ $<
#----
$(RPCBASE)_xdr.o: $(RPCBASE)_xdr.c
	$(CXX) $(CUDAINCLUDES) -c -fPIC -o $@ $<
#----
dscudautil.o: dscudautil.cu dscudautil.h
	$(NVCC) $(CUDAINCLUDES) -c -Xcompiler -fPIC -o $@ $<
###
### source file include dependancy.
###
libdscuda.cu:      dscuda.h libdscuda.h
libdscuda_rpc.cu:  dscuda.h libdscuda.h
libdscuda_ibv.cu:  dscuda.h libdscuda.h ibv_rdma.h
dscudad.cu :       dscudadefs.h sockutil.h dscudarpc.h
dscudaverb.cu:     dscuda.h libdscuda.h dscudaverb.h dscudarpc.h
dscudatest.cu:     dscuda.h ibv_rdma.h
ibv_rdma.cpp:      ibv_rdma.h 
ibv_rdma.h:        dscudadefs.h dscudarpc.h dscudautil.h
dscuda.h:          dscudadefs.h dscudarpc.h dscudautil.h ibv_rdma.h

.PHONY: install clean
install:
	@echo "//************************************************"
	@echo "//**  Install dscuda files to bin/lib/include.  **"
	@echo "//************************************************"
	ln -fs ../src/dscuda.h       ../include/
	ln -fs ../src/dscudadefs.h   ../include/
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
	@echo "//*********************************"
	@echo "//**  Clean up generated files.  **"
	@echo "//*********************************"
	/bin/rm -fv $(TARGET) $(TARGET_RPCONLY) *.o *.so *~ *.linkinfo
