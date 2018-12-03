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

// Option 1
__global__ void BFSAlgorithm(int  *graph, 
    int  *numEdges,     // arrray for number of edges for each vertex
    int  *queue,        // queue of edges to be searched
    bool *visited,      // array to see if edges have been visited
    int   source,
    int   dest,
    int   vertexSize,
    int   maxEdges,
    int  *pred,
    int  *dist,
    int   totalEdges,
    volatile bool *found)    // Total number of edges in the graph
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
    queue[0] = source; 

    // BFS algorithm  

    // Pointer for front of Queue
    int pointer     = 0;
    int backPointer = 1;

    while (pointer != totalEdges && !foundDest) 
    {  
        int queueIter = queue[pointer]; 
        int iter = maxEdges * pointer;
        pointer++; 

        int count = 0;
        
        while (count < maxEdges )
        { 
            
            int i = iter/maxEdges;
            int j = iter - (i * maxEdges);

            int index = i * maxEdges + j;
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
                    dist[graph[index]]    = dist[queueIter] + 1; 
                    pred[graph[index]]    = queueIter; 

                    queue[backPointer] = graph[index]; 

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
    int  *d_edgeQueue;       // queue of edges to be searched
    int  *d_pred;            // array to store predecssors
    int  *d_dist;            // array to store distances

    int  *d_graph;           // Pointer to 1 d array containing the graph

    // Allocate  Global memory
    cudaMalloc((void**) &d_graph, graphSize);
    cudaMalloc((void**) &d_vertexVisited,  sizeof(bool) * numVerticies); //doesnt need populating
    cudaMalloc((void**) &d_found,          sizeof(bool));                 
    cudaMalloc((void**) &d_numEdges,       arraySizeVertices);            // does
    cudaMalloc((void**) &d_edgeQueue,      arraySizeEdges);            // doesnt
    cudaMalloc((void**) &d_pred,           arraySizeVertices);            // doesnt
    cudaMalloc((void**) &d_dist,           arraySizeVertices);            // doesnt

    // Copy Memory to the device for the kernel
    cudaMemcpy(d_numEdges, h_numEdges, arraySizeVertices, cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph, h_graph, graphSize, cudaMemcpyHostToDevice);


    // BFS call
    BFSAlgorithm<<<numVerticies,1>>>(d_graph,
                   d_numEdges,
                   d_edgeQueue, 
                   d_vertexVisited, 
                   source,
                   destination,
                   numVerticies,
                   maxEdges,
                   d_pred,
                   d_dist,
                   totalEdges,
                   d_found);

    cudaDeviceSynchronize();

    cudaMemcpy((void**)foundDest, d_found, sizeof(bool), cudaMemcpyDeviceToHost);

    cudaMemcpy(h_dist, d_dist, arraySizeVertices, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pred, d_pred, arraySizeVertices, cudaMemcpyDeviceToHost);

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
    cudaFree(d_edgeQueue);
    cudaFree(d_pred);
    cudaFree(d_dist);

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

            // FindShortestPath<<<1, 1>>>(thrust::raw_pointer_cast(d_path.data()),
            //                            thrust::raw_pointer_cast(d_pred.data()),
            //                            destination,
            //                            d_path.size());

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