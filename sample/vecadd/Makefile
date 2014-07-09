DSCUDACPP        = ../../bin/dscudacpp -arch=sm_20
CUDAPATH        ?= /usr/local/cuda-4.2/cuda
CUDASDKPATH     ?= /usr/local/cuda-4.2/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
DSCUDAINCLUDE    = -I$(DSCUDA_PATH)/include
NVCC             = $(CUDAPATH)/bin/nvcc -use_fast_math -O -arch=sm_20 



TARGET = vecadd
SRC = vecadd.cu

all: $(TARGET)_rpc

$(TARGET)_ibv: $(SRC) ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -libverbs -lrdmacm -lpthread

$(TARGET)_rpc: $(SRC) ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -I. -ldscuda_rpc -lpthread -i $< -o $@

$(TARGET)_rpconly: $(SRC) ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -DRPC_ONLY=1 -o $@ -I. -i $< -ldscuda_rpc

$(TARGET).o: $(SRC) ../../lib/libdscuda.a
	$(DSCUDACPP) -c -o $@ -I. -i $< -ldscuda -libverbs -lrdmacm -lpthread

$(TARGET)_local: $(SRC)
	$(NVCC) -o $@ $(CUDAINCLUDES) $(DSCUDAINCLUDE) $< -lcuda

clean:
	rm -rf *.o *.ptx $(TARGET)_{ibv,rpc,rpconly,local} *.linkinfo ./dscudatmp
