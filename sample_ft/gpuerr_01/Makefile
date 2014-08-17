DSCUDACPP        = ../../bin/dscudacpp -arch=sm_20
CUDAPATH        ?= /usr/local/cuda-4.2/cuda
CUDASDKPATH     ?= /usr/local/cuda-4.2/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
DSCUDAINCLUDE    = -I$(DSCUDA_PATH)/include
NVCC             = $(CUDAPATH)/bin/nvcc
NVCC_OPT         = -use_fast_math -O -g -arch=sm_20 --compiler-bindir=gcc-4.6



TARGET = gpuerr_01
SRC = gpuerr_01.cu

all: $(TARGET)_rpc

$(TARGET).rpc: $(SRC) ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -I. -ldscuda_rpc -lpthread -i $< -o $@

$(TARGET)_rpconly: $(SRC) ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -DRPC_ONLY=1 -o $@ -I. -i $< -ldscuda_rpc

$(TARGET).o: $(SRC) ../../lib/libdscuda.a
	$(DSCUDACPP) -c -o $@ -I. -i $< -ldscuda -libverbs -lrdmacm -lpthread

$(TARGET).local: $(SRC)
	$(NVCC) $(NVCC_OPT) -o $@ $<

clean:
	rm -rf *.o *.ptx $(TARGET)_{ibv,rpc,rpconly,local} *.linkinfo ./dscudatmp
