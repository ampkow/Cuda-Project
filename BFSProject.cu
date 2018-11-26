// nvgraph_assign.cu -- NVGraph  Programming Assignment
// Adam Piorkowski
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "nvgraph.h"

#include "KernelFunctions.h"
 
int main(int argc, char** argv)
{
	// read command line arguments
    int verticies  = 8;

    if (argc >= 2) 
    {
		verticies = atoi(argv[1]);
	}

    printf("Element test size is %d\n", verticies);

    // Using Advanced Library NVGraph for finding the shortest Path
    float run_time = 0.0; //run_nvgraph_search(verticies);

    RunBFSShortestDistance(8, 7, 0);

    printf("Total run time for range test %f\n", run_time);

    printf("Succesful Run! --------------------------------\n");
    return 0;
}