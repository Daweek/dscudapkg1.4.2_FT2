# ARCH = $(shell uname -m)
# ifeq ($(ARCH), x86_64)

ifeq ($(shell uname -m), x86_64)
	LP64 = 1
else
	LP64 = 0
endif

BUILD_MODE = DEBUG_MODE

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

TARGET        = dscudad dscudasvr dscudasvr_fault libdscuda_rpc.a libcudart.so.3
DBGTARGET     = $(DBGDIR)/$(TARGET)
RELTARGET     = $(RELDIR)/$(TARGET)

TARGET_RPCONLY = dscudad_rpc libdscuda_rpc.a libcudart.so.3

RPCBASE        = dscudarpc
CLIENTBASE     = libdscuda.o

### Build in Debug mode.
DBGDIR    = ./Debug
CDEBUG    = -Wall -O0 -g -gp -D_DEBUG

### Build in Release mode.
RELDIR    = ./Release
CRELEASE  = -Wall -O2 -D_RELEASE

### Build by default.
ODIR    = $(DBGDIR)

.PHONY: rpc_generated
rpc_generated: $(RPCBASE)_clnt.c $(RPCBASE)_svc.c $(RPCBASE)_xdr.c $(RPCBASE).h

###
### Phony target
###
.PHONY: all clean install

.PHONY: debug release
debug: $(DBGTARGET)
	@echo "********************"
	@echo "*** Debug target ***"
	@echo "********************"
	mkdir -p $(DBGDIR)
	cd $(DBGDIR); make

release: $(RELTARGET)
	@echo "**********************"
	@echo "*** Release target ***"
	@echo "**********************"
	mkdir -p $(RELDIR)
	cd $(RELDIR); make

all : debug release

###
### Create RPC source files into current direcotory by "rpcgen".
###
$(RPCBASE)_clnt.c:	$(RPCBASE).x $(RPCBASE).h
	$(RPCGEN) -l $< > $@
$(RPCBASE)_svc.c:	$(RPCBASE).x $(RPCBASE).h
	$(RPCGEN) -m $< > $@
$(RPCBASE)_xdr.c:	$(RPCBASE).x $(RPCBASE).h
	$(RPCGEN) -c $< > $@
$(RPCBASE).h:		$(RPCBASE).x
	$(RPCGEN) -h $< > $@
$(RPCBASE).x: dscudadefs.h

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
	/bin/rm -fv $(TARGET) $(TARGET_RPCONLY) $(rpc_generatedE).h $(RPCBASE)_svc.* $(RPCBASE)_clnt.* $(RPCBASE)_xdr.* *.o *~ *.linkinfo
