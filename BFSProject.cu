// nvgraph_assign.cu -- NVGraph  Programming Assignment
// Adam Piorkowski
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "nvgraph.h"

#include "KernelFunctions.h"

void RunBFSUsingCuda(std::vector<std::vector<int> > &graph,
                     int                             dest,
                     int                             source,
                     int                             totalEdges)
{
    RunBFSShortestDistance(graph, dest, source, totalEdges);
}
