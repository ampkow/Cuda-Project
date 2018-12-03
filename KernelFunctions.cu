// Contains Kernel functions for Cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

// STL
#include <vector> 
#include <list> 
#include <iostream>

// Internal Headers
#include "KernelFunctions.h"

// taken from global_memory.cu, Creates event and records time
__host__ cudaEvent_t get_time(void)
{
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time);
	return time;
}

__global__ void FindShortestPath(int *path,
                                 int *pred,
                                 int  fullSize,
                                 int  dest)
{
    int pointer = dest;

    int pathSize = 1;

    while (pred[pointer] != -1) 
    { 
       path[pathSize - 1] = pred[pointer]; 
       pointer = pred[pointer];
       pathSize++;
    }       
      
    // printing path from source to destination 
    printf("\nShortest Path Length is %d\n", pathSize); 
    int count = 0;
    int iter = fullSize - 1;
    while (count < pathSize) 
    {
        printf(" %d ", path[iter]); 
        count++;
        iter--;
    }

    // printing path from source to destination 
    printf("\n Shortest Path: \n"); 
    for (int iter = pathSize - 1; iter >= 0; iter--) 
    {
      printf("index %d ", path[iter]); 
    }

}

__global__ void BFSAlgorithm(int  *graph, 
                             int  *numEdges,     // arrray for number of edges for each vertex
                             int  *nextVertList, // List of edges to be searched
                             bool *visited,      // array to see if edges have been visited
                             int   source,
                             int   dest,
                             int   vertexSize,
                             int   maxEdges,
                             int  *pred,
                             int  *dist,
                             int   totalEdges,   // Total number of edges in the graph
                             volatile bool *found)    // Variable to signify destination was found
{

    volatile __shared__ bool foundDest;

    // initialize shared status
    if (threadIdx.x == 0) 
    {
        foundDest = *found;
    }
    __syncthreads();

    // TODO initalize these outside of this kernel
    for (int iter = 0; iter < vertexSize; iter++) 
    { 
        visited[iter] = false; 
        dist[iter]    = 0; 
        pred[iter]    = -1; 
    } 

    visited[source] = true; 
    dist[source] = 0; 
    nextVertList[0] = source; 

    // BFS algorithm  

    // Pointer for front of List
    int pointer     = 0;
    int backPointer = 1;

    while (pointer != totalEdges && !foundDest) 
    {  
        int currVertIter = nextVertList[pointer]; 
        int iter = maxEdges * pointer;
        pointer++; 

        int count = 0;
        
        while (count < maxEdges)
        { 
            
            int xVal = iter/maxEdges;
            int yVal = iter - (xVal * maxEdges);

            int index = xVal * maxEdges + yVal;
            if (graph[index] == -100)
            {
                // jump to next vertex
                count =  maxEdges;
            }
            else
            {
                if (visited[graph[index]] == false) 
                {    
                    visited[graph[index]] = true; 
                    dist[graph[index]]    = dist[currVertIter] + 1; 
                    pred[graph[index]]    = currVertIter; 

                    nextVertList[backPointer] = graph[index]; 

                    iter = index;
                    backPointer++;

                    // Stop When finding destination
                    if (graph[index] == dest) 
                    {
                        foundDest = true;
                    }
                } 
            }    
            count++;
            iter++;
        }
    }

    if (threadIdx.x == 0 && *found) 
    {
        foundDest = true;
    }
    __syncthreads(); 
}

/* Flattens vector of vectors into 1 D array
 * Graph    -- Vector of vectors containing graph 
 * maxEdges -- Max number of edges in the graph
 *
 */
int *ConvertVectorTo2D(std::vector<std::vector<int> >  &graph,
                       int                              maxEdges)
{
    int *h_graph;
    int graphSize = graph.size();
    h_graph = new int[graphSize * maxEdges];

    for (int outIter = 0; outIter < graphSize; outIter++)
    {
        for(int iter2 = 0; (iter2 < maxEdges); ++iter2)
        {
            if (iter2 >= graph[outIter].size())
            {
                h_graph[outIter * maxEdges + iter2] = -100;
            }
            else
            {
                h_graph[outIter * maxEdges + iter2] = graph[outIter][iter2];
            }
        } 
    }

    return h_graph;
}

// Sets up memory for graph and calls kernels
float RunBFSShortestDistance(std::vector<std::vector<int> > &graph,
                             int                             destination,
                             int                             source,
                             int                             totalEdges)
{
    bool foundDest = false;

    int numVerticies = graph.size();

    int arraySizeVertices = sizeof(int) * numVerticies;
    int arraySizeEdges    = sizeof(int) * totalEdges;

    int maxEdges = 0;

    int *h_numEdges = (int *) malloc(arraySizeVertices); 
    int *h_dist     = (int *) malloc(arraySizeVertices); 
    int *h_pred     = (int *) malloc(arraySizeVertices); 

    for (int vectIter = 0; vectIter < numVerticies; vectIter++)
    {
        int edgeCount = graph[vectIter].size();
        h_numEdges[vectIter] = edgeCount;

        if (edgeCount > maxEdges)
        {
            maxEdges = edgeCount;
        }
    }

    // Convert Vector of Vector to 2d array
    int* h_graph = ConvertVectorTo2D(graph, maxEdges);

    int graphSize = sizeof(int) * (graph.size() * maxEdges);

    bool *d_vertexVisited;   // array to see if edges have been visited
    bool *d_found;           // Found Destination
    int  *d_numEdges;        // arrray for number of edges for each vertex
    int  *d_edgeList;        // List of edges to be searched
    int  *d_predecessors;            // array to store predecssors
    int  *d_distances;            // array to store distances

    int  *d_graph;           // Pointer to 1 d array containing the graph

    // Allocate  Global memory
    cudaMalloc((void**) &d_graph, graphSize);
    cudaMalloc((void**) &d_vertexVisited,  sizeof(bool) * numVerticies);
    cudaMalloc((void**) &d_found,          sizeof(bool));                 
    cudaMalloc((void**) &d_numEdges,       arraySizeVertices);            
    cudaMalloc((void**) &d_edgeList,       arraySizeEdges);
    cudaMalloc((void**) &d_predecessors,           arraySizeVertices);
    cudaMalloc((void**) &d_distances,           arraySizeVertices); 

    // Copy Memory to the device for the kernel
    cudaMemcpy(d_numEdges, h_numEdges, arraySizeVertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph, h_graph, graphSize, cudaMemcpyHostToDevice);


    // BFS call
    BFSAlgorithm<<<numVerticies,1>>>(d_graph,
                   d_numEdges,
                   d_edgeList, 
                   d_vertexVisited, 
                   source,
                   destination,
                   numVerticies,
                   maxEdges,
                   d_predecessors,
                   d_distances,
                   totalEdges,
                   d_found);

    cudaDeviceSynchronize();

    cudaMemcpy((void**)foundDest, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_dist, d_distances, arraySizeVertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pred, d_predecessors, arraySizeVertices, cudaMemcpyDeviceToHost);

    if (foundDest)
    {
        thrust::device_vector<int> d_path(totalEdges);
        d_path.push_back(destination);

        int pointer = destination;

        int pathSize = 1;

        while (h_pred[pointer] != -1) 
        { 
           pathSize++;
           d_path.push_back(h_pred[pointer]); 
           pointer = h_pred[pointer];
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
    }
    

    cudaFree(d_vertexVisited);
    cudaFree(d_found);
    cudaFree(d_numEdges);
    cudaFree(d_edgeList);
    cudaFree(d_predecessors);
    cudaFree(d_distances);

    free(h_numEdges);
    free(h_dist);
    free(h_pred);

    return 0.0;
}

// Run BFS algorithm using thurst library
float RunBFSUsingThrust(std::vector<std::vector<int> > &graph,
                         int                            destination,
                         int                            source,
                         int                            totalEdges)
{
    std::list<int> nextVertList; 

    int vertexSize = graph.size();

    thrust::device_vector<int> d_visited(vertexSize);
    thrust::device_vector<int> d_predecessors(vertexSize);
    thrust::device_vector<int> d_distances(vertexSize);
    thrust::device_vector<int> d_path(totalEdges);

    thrust::fill(d_visited.begin(), d_visited.end(), false);
    thrust::fill(d_distances.begin(), d_distances.end(), 0);
    thrust::fill(d_predecessors.begin(), d_predecessors.end(), -1);

    nextVertList.push_back(source);
    d_visited[source] = true;

    bool foundDest = false;

    // BFS algorithm  
    while (!nextVertList.empty()) 
    {  
        int currVertIter = nextVertList.front();  // Current Vertex
        nextVertList.pop_front(); 

        int edgeCount = graph.at(currVertIter).size();

        // need to populate with vector of edges for current vertex
        thrust::device_vector<int> d_edges(graph.at(currVertIter));

        for (int iter0 = 0; iter0 < edgeCount; iter0++)
        {
            int nextVert = d_edges[iter0];
            if (d_visited[nextVert] == false) 
            { 
                printf("Element (%d, %d) = %d \n", currVertIter, iter0, nextVert);
    
                d_visited[nextVert] = true; 
                d_distances[nextVert]    = d_distances[currVertIter] + 1; 
                d_predecessors[nextVert]    = currVertIter; 
    
                nextVertList.push_back(nextVert);
    
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

            // FindShortestPath<<<1, 1>>>(thrust::raw_pointer_cast(d_path.data()),
            //                            thrust::raw_pointer_cast(d_predecessors.data()),
            //                            destination,
            //                            d_path.size());

            while (d_predecessors[pointer] != -1) 
            { 
               pathSize++;
               d_path.push_back(d_predecessors[pointer]); 
               pointer = d_predecessors[pointer];
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