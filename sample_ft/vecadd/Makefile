CC            ?= /usr/bin/gcc
CXX           ?= /usr/bin/g++
DSCUDACPP     ?= ../../bin/dscudacpp -arch=sm_20
CUDAINCLUDES  = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
DSCUDAINCLUDE = -I$(DSCUDA_PATH)/include -I$(DSCUDA_PATH)/src
NVCC          ?= $(CUDAPATH)/bin/nvcc
NVCCFLAGS     = -arch=sm_20

TARGET = vecadd
SRC = vecadd.cu

all: $(TARGET)_rpc

$(TARGET)_rpc: $(SRC) ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) $(NVCCFLAGS) -DRPC_ONLY=1 -g -I. -ldscuda_rpc -lpthread -i $< -o $@

$(TARGET).o: $(SRC) ../../lib/libdscuda.a
	$(DSCUDACPP) $(NVCCFLAGS) -c -g -DRPC_ONLY=1 -o $@ -I. -i $< -ldscuda -lpthread

$(TARGET)_local: $(SRC)
	$(NVCC) -o $@ -g $(CUDAINCLUDES) $(DSCUDAINCLUDE) $< -lcuda

clean:
	rm -rf *.o *.ptx $(TARGET)_rpc $(TARGET)_local *.linkinfo ./dscudatmp
