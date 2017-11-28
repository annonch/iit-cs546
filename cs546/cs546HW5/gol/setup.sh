#!/bin/bash
./clean.sh
#/usr/local/cuda-7.0/bin/nvcc -ccbin g++ -I../../common/inc  -m64    -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -o GoLCuda.o -c GoLCuda.cu
nvcc -o GoLCuda GoLCuda.cu
gcc -o GoLS GoLS.c
sbatch g1.sh
sbatch g2.sh