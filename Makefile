all: BFS.cu
	nvcc  -lcudart -lcuda -lnvgraph -I ../../../common/inc BFSProject.cu KernelFunctions.cu -o BFS
