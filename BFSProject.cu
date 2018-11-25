// nvgraph_assign.cu -- NVGraph  Programming Assignment
// Adam Piorkowski
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "nvgraph.h"

#include "KernelFunctions.h"

// __host__ allows to run on device and host and void code duplication
// taken from global_memory.cu
__host__ cudaEvent_t get_time(void)
{
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time);
	return time;
}
 
int main(int argc, char** argv)
{
	// read command line arguments
    int verticies  = 8;

    if (argc >= 2) 
    {
		elements = atoi(argv[1]);
	}

    printf("Element test size is %d\n", verticies);

    // Using Advanced Library NVGraph for finding the shortest Path
    float run_time = run_nvgraph_search(verticies);

    printf("Total run time for range test %f\n", run_time);

    printf("Succesful Run! --------------------------------\n");
    return 0;
}