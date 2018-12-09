all: BFSProject.cpp
	nvcc  -lcudart -lcuda -I ../../../common/inc BFSProject.cpp KernelFunctions.cu -o BFS
