DSCUDACPP        = ../../bin/dscudacpp
CUDAPATH        ?= /usr/local/cuda4.1/cuda
CUDASDKPATH     ?= /usr/local/cuda4.1/NVIDIA_GPU_Computing_SDK
CUDAINCLUDES     = -I. -I$(CUDAPATH)/include -I$(CUDASDKPATH)/C/common/inc
NVCC						?= $(CUDAPATH)/bin/nvcc -use_fast_math -O

TARGET = matrixMul

$(TARGET):	matrixMul.cu ../../lib/libdscuda.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda -libverbs -lrdmacm -lpthread

$(TARGET)_rpc:	matrixMul.cu ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -o $@ -I. -i $< -ldscuda_rpc

$(TARGET)_rpconly:	matrixMul.cu ../../lib/libdscuda_rpc.a
	$(DSCUDACPP) -DRPC_ONLY=1 -o $@ -I. -i $< -ldscuda_rpc

$(TARGET)_local: matrixMul.cu
	$(NVCC) -o $@ $(CUDAINCLUDES) $< -lcuda


clean:
	rm -rf *.o *.ptx $(TARGET) $(TARGET)_rpc *.linkinfo ./dscudatmp
