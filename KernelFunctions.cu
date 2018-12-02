// Make Kernel Functions inside of here
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>


// STL
#include <vector> 
#include <list> 
#include <iostream>

// Internal Headers
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

__global__ void FindShortestPath(int *path,
                                 int *pred,
                                 int  dest)
{
    int pointer = dest;

    int pathSize = 0;

    while (pred[pointer] != -1) 
    { 
        pathSize++;
        path[pathSize] = pred[pointer]; 
        pointer = pred[pointer]; 
    } 

    // printing path from source to destination 
    printf("\n Shortest Path: \n"); 
    for (int iter = pathSize - 1; iter >= 0; iter--) 
    {
      printf("index %d ", path[iter]); 
    }

}
  

// Run BFS algorithm using thurst library
float RunBFSUsingThrust(std::vector<std::vector<int> > &graph,
                         int                            destination,
                         int                            source,
                         int                            totalEdges)
{
    std::list<int> queue; 

    int vertexSize = graph.size();

    thrust::device_vector<int> d_visited(vertexSize);
    thrust::device_vector<int> d_pred(vertexSize);
    thrust::device_vector<int> d_dist(vertexSize);
    thrust::device_vector<int> d_path(totalEdges);

    thrust::fill(d_visited.begin(), d_visited.end(), false);
    thrust::fill(d_dist.begin(), d_dist.end(), 0);
    thrust::fill(d_pred.begin(), d_pred.end(), -1);

    queue.push_back(source);
    d_visited[source] = true;

    bool foundDest = false;

    // BFS algorithm  
    while (!queue.empty()) 
    {  
        int queueIter = queue.front();  // Current Vertex
        queue.pop_front(); 

        int edgeCount = graph.at(queueIter).size();

        // need to populate with vector of edges for current vertex
        thrust::device_vector<int> d_edges(graph.at(queueIter));

        for (int iter0 = 0; iter0 < edgeCount; iter0++)
        {
            int nextVert = d_edges[iter0];
            if (d_visited[nextVert] == false) 
            { 
                printf("Element (%d, %d) = %d \n", queueIter, iter0, nextVert);
    
                d_visited[nextVert] = true; 
                d_dist[nextVert]    = d_dist[queueIter] + 1; 
                d_pred[nextVert]    = queueIter; 
    
                queue.push_back(nextVert);
    
                // Stop When finding destination
                if (nextVert == destination) 
                {
                    foundDest = true;
                    break; 
                }
            }
        }

        if (foundDest)
        {
            d_path.push_back(destination);

            int pointer = destination;

            int pathSize = 1;

            while (d_pred[pointer] != -1) 
            { 
               pathSize++;
               d_path.push_back(d_pred[pointer]); 
               pointer = d_pred[pointer];
            }       
              
            // printing path from source to destination 
            printf("\nShortest Path Length is %d\n", pathSize); 
            int count = 0;
            int iter = d_path.size() - 1;
            while (count < pathSize) 
            {
                std::cout << d_path[iter] << " "; 
                count++;
                iter--;
            }

            return true;
        }
    } 
    return false; 
}