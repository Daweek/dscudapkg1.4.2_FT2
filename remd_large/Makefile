##                            -*- Mode: GNUmakefile -*-
## Filename         : Makefile
## Description      :
## Author           : Minoru Oikawa (m_oikawa@amber.plala.or.jp)
## Created On       : 2013-08-25 11:49:27
## Last Modified By : Minoru Oikawa
## Last Modified On : 2014-03-18 13:18:07
## Update Count     : 0.1
## Status           : Unknown, Use with caution!
##----------------------------------------------------------------------------- 
#CCLIB		= -lpthread -lgomp
CUDAPATH        ?= /usr/local/cuda-4.2/cuda
CUDALIB		= -L$(CUDAPATH)/lib64 -lcudart -lGLU -lGL
DSCUDA_PATH     = /home/m_oikawa/dscudapkg
DSCUDAFLAGS     = -I$(DSCUDA_PATH)/include
DSCUDALIB_RPC   = -ldscuda_rpc

#### compilers selection 
CC = gcc
CXX = g++
NVCC = $(CUDAPATH)/bin/nvcc
DSCUDACPP ?= /bin/dscudacpp

CPPFLAGS = 
NVCCFLAGS = -O2 -g -I${CUDASDKPATH}/CUDALibraries/common/inc -I${HOME} \
	    -arch=compute_20 -code=compute_20 \
   -I/usr/apps/free/NVIDIA_GPU_Computing_SDK/4.1/C/common/inc  --use_fast_math --ptxas-options=-v

OUTPUT_FILE = histogram/histogram*.dat index_replica.dat energy_replica.dat input.txt acceptance_ratio.dat

SRCS = mytools.C mytools.H calc_force.cu calc_force.H comm_save.cu comm_save.H \
       init_cond.cu  init_cond.H integ.cu integ.H \
       remd_top.cu remd_typedef.H switch_float_double.H \
       remd_config.idata config_dev.idata config_hst.idata test_debug.cu

EXES = remd_top

OBJS_HST = remd_top_hst.o init_cond.o integ_hst.o comm_save_hst.o mytools.o

OBJS_DEV = remd_top_dev.o init_cond.o integ_dev.o comm_save_dev.o mytools.o

OBJS_DEV_FJ = remd_top_dev.o init_cond.o integ_dev.fault.o comm_save_dev.o mytools.o

OBJS_DS  = remd_top_ds.o  init_cond.o integ_ds.o  comm_save_ds.o  mytools.o

OBJS_DS_RCUT = remd_top_ds.o  init_cond.o integ_ds_rcut.o  comm_save_ds.o  mytools.o

curr : remd_rpc

all	: remd_hst remd_local remd_rpc

#### CPU version 
remd_hst : $(OBJS_HST)
	@echo "//"
	@echo "// Generate remd_hst"
	@echo "//"
#	$(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -o $@ $^ 
	$(CXX) $(NVCCFLAGS) $(CPPFLAGS) $(CUDALIB) $^ -o $@

#### CUDA version
remd_local : $(OBJS_DEV)
	@echo "//"
	@echo "// Generate remd_local"
	@echo "//"
	$(NVCC) $(NVCCFLAGS) $(DSCUDAFLAGS) -L. -L$(CUDAPATH)/lib64 $^ $(CCLIB) -o $@

remd_local.fault : $(OBJS_DEV_FJ)
	@echo "//"
	@echo "// Generate remd_local"
	@echo "//"
	$(NVCC) $(NVCCFLAGS) $(DSCUDAFLAGS) -L. -L$(CUDAPATH)/lib64 $^ $(CCLIB) -o $@

remd_local_meas_kernel : remd_top_dev.o init_cond.o integ_dev_meas_kernel.o comm_save_dev.o mytools.o
	$(NVCC) $(NVCCFLAGS) -L. -L$(CUDAPATH)/lib64 -DMEAS_CUDA_EVENT $^ $(CCLIB) -o $@
#### DS-CUDA version

remd_rpc : $(OBJS_DS)
	@echo "//"
	@echo "// Generate remd_ds_rpc"
	@echo "//"
	$(NVCC) $(NVCCFLAGS) -L. -L$(DSCUDA_PATH)/lib -L$(CUDAPATH)/lib64 -link -lrt $^ $(DSCUDALIB_RPC) $(CCLIB) -o $@
#	$(CXX) $(NVCCFLAGS) -L. -L$(DSCUDA_PATH)/lib -L$(CUDAPATH)/lib64 $^ $(DSCUDALIB_RPC) $(CCLIB) -o $@

remd_rpc_rcut : $(OBJS_DS_RCUT)
	@echo "//"
	@echo "// Generate remd_ds_rpc"
	@echo "//"
	$(NVCC) $(NVCCFLAGS) -L. -L$(DSCUDA_PATH)/lib -L$(CUDAPATH)/lib64 -link $^ $(DSCUDALIB_RPC) $(CCLIB) -o $@

#
# remd_top.cu
#
remd_top_hst.o : remd_top.cu
	$(NVCC) -c -D"HOST_RUN" $(NVCCFLAGS) $< -o $@
remd_top_dev.o : remd_top.cu
	$(NVCC) -c -D"DEVICE_RUN" $(NVCCFLAGS) $(DSCUDAFLAGS) $< -o $@
remd_top_ds.o  : remd_top.cu
	$(DSCUDACPP) -c -D"DEVICE_RUN" -I. $(NVCCFLAGS) -i $< -o $@

remd_top.cu : init_cond.H remd_typedef.H mytools.H
#
# integ.cu : cpu(*_hst.o) / cuda(*_dev.o) / ds-cuda(*_ds.o)
#
integ_hst.o : integ.cu
	$(NVCC) -c -D"HOST_RUN" $(NVCCFLAGS) $< -o $@
integ_dev.o : integ.cu
	$(NVCC) -c -D"DEVICE_RUN" $(NVCCFLAGS) $(DSCUDAFLAGS) $< -o $@
integ_dev.fault.o : integ.cu
	$(NVCC) -c -D"DEVICE_RUN" -D"FAULT_ON" $(NVCCFLAGS) $(DSCUDAFLAGS) $< -o $@
integ_dev_meas_kernel.o : integ.cu
	$(NVCC) -c -D"DEVICE_RUN" -D"MEAS_CUDA_EVENT" $(NVCCFLAGS)  $< -o $@
integ_ds.o  : integ.cu
	$(DSCUDACPP) -c -D"DEVICE_RUN" -D"__DSCUDA__" -I. $(NVCCFLAGS) -i $< -o $@
integ_ds_rcut.o : integ.cu
	$(DSCUDACPP) -c -D"DEVICE_RUN" -D"RCUT_COUNT" -I. $(NVCCFLAGS) -i $< -o $@
integ.cu : switch_float_double.H remd_typedef.H mytools.H init_cond.H \
           calc_force.cu integ.H
#
# comm_save.cu
#
comm_save_hst.o : comm_save.cu
	$(NVCC) -c -D"HOST_RUN" $(NVCCFLAGS) $< -o $@
comm_save_dev.o : comm_save.cu
	$(NVCC) -c -D"DEVICE_RUN" $(NVCCFLAGS) $< -o $@
comm_save_ds.o  : comm_save.cu
	$(DSCUDACPP) -c -D"DEVICE_RUN" -I. $(NVCCFLAGS) -i $< -o $@
comm_save.cu : switch_float_double.H mytools.H remd_typedef.H init_cond.H \
               integ.H comm_save.H
#
# init_cond.cu
#
init_cond.o : init_cond.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@
init_cond.cu : init_cond.H remd_typedef.H mytools.H
init_cond.H : mytools.H
#
# etc
#
mytools.o : mytools.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@
#---
#--- PHONY
#---
test_hst : remd_hst
	rm -rf ./config_hst.odata/*
	./$< -i config_hst.idata | tee log_hst

test_dev : remd_local
	rm -rf ./config_dev.odata/*
	./$< -i config_dev.idata | tee log_dev

backup :
	./bkup.sh

pack	:
	mkdir output
	cp $(OUTPUT_FILE) output
	tar zcvf ./output.tar.gz ./output
	rm -r output
cdv	:
	tar zcvf ./cdv.tar.gz ./cdv
eps	:
	mkdir eps
	cp ./*.eps ./*.plt ./eps
	tar zcvf ./eps.tar.gz ./eps
	rm -r eps

clean	:
	rm -f *~
	rm -rf remd_hst remd_local remd_ds_rpc *.o dscudatmp
check	:
	./remd_local input.txt > test.log
	diff test.log exact.log
dist :
	tar cvfzh remd_1024.tgz $(SRCS) Makefile remd_config.idata
push_bitbucket:
	hg push https://m_oikawa@bitbucket.org/m_oikawa/remd1024gpu

push_github:
	git push -u https://github.com/moikawa/remd_large.git master
