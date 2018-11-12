all: nvgraph_assign.cu
	nvcc  -lcudart -lcuda -lnvgraph -I ../../../common/inc nvgraph_assign.cu -o nvgraph_assign
