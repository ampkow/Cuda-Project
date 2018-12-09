// Contains GPU Cuda code that executes BFS algorithm
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

/**
 * Runs BFS using Thrust Vectors and Thrust Library
 * graph       - contains all vertices and their edges
 * destination - destination vertex
 * source      - source vertex
 * totalEdges  - total number of edges in graph
 * 
**/
float RunBFSUsingThrust(std::vector<std::vector<int> > &graph,
                         int                            destination,
                         int                            source,
                         int                            totalEdges)
{
    std::list<int> nextVertList; 

    int vertexSize = graph.size();

    cudaEvent_t start = get_time();

    // Device vectors, Allocate memory
    thrust::device_vector<int> d_visited(vertexSize);
    thrust::device_vector<int> d_predecessors(vertexSize);
    thrust::device_vector<int> d_distances(vertexSize);
    thrust::device_vector<int> d_path;

    // Initialize Values for vectors
    thrust::fill(d_visited.begin(), d_visited.end(), false);
    thrust::fill(d_distances.begin(), d_distances.end(), 0);
    thrust::fill(d_predecessors.begin(), d_predecessors.end(), -1);

    // Push source onto vertex list to iterate through
    nextVertList.push_back(source);
    d_visited[source] = true;

    cudaEvent_t memFinished = get_time();

    bool foundDest = false;

    // BFS algorithm  
    while (!nextVertList.empty()) 
    {  
        int currVertIter = nextVertList.front();  // Current Vertex
        nextVertList.pop_front(); 

        // Amount of edges for current Vertex
        int edgeCount = graph.at(currVertIter).size();

        // need to populate with vector of edges for current vertex
        thrust::device_vector<int> d_edges(graph.at(currVertIter));

        // Iterate over edges of current vertex
        for (int edgeIter = 0; edgeIter < edgeCount; ++edgeIter)
        {
            // grab end of edge
            int nextVert = d_edges[edgeIter];

            // Check to see if edge has been visited yet
            if (d_visited[nextVert] == false) 
            {  
                d_visited[nextVert]      = true; 
                d_distances[nextVert]    = d_distances[currVertIter] + 1; 
                d_predecessors[nextVert] = currVertIter; 
    
                nextVertList.push_back(nextVert);
    
                // Return after reaching destination
                if (nextVert == destination) 
                {
                    foundDest = true;
                    break; 
                }
            }
        }
    }

    cudaEvent_t bfsFinised = get_time();

    if (foundDest)
    {
        // Add destination to path
        d_path.push_back(destination);

        int pointer = destination;
        int pathSize = 1;

        // if predecessor was visited add to path 
        // for destination
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

        std::cout << std::endl;
    } 
    else
    {
        printf("Shortest Path not found to destination %d using Thrust library \n", destination); 
    }    

    cudaEventSynchronize(bfsFinised);

    float memAllocTime;
    cudaEventElapsedTime(&memAllocTime, start, memFinished);

    float kernelRunTime;
    cudaEventElapsedTime(&kernelRunTime, memFinished, bfsFinised);
    
    float totalTime;
    cudaEventElapsedTime(&totalTime, start, bfsFinised);

    // Clean Up Events
    cudaEventDestroy(start);
    cudaEventDestroy(memFinished);
    cudaEventDestroy(bfsFinised);

    printf("Thrust Memory Allocation time %f\n", memAllocTime);
    printf("Thrust Kernenl Runtime %f \n", kernelRunTime);

    printf("\n");
    
    return totalTime; 
}


/**
 * Kernel Algorithm
 * Runs BFS on GPU by searching level by level
 * vertices     - list of vertices for GPU
 * edges        - list of edge destinations for GPU
 * distances    - stores distance from source to current vertex
 * predecessors - stores previous vertex
 * vertIndices  - list of start points for each vertices edges in edge list
 * edgeSize     - list of how many edges each vertex has
 * levels       - stores if current level of this vertex has been visited
 * visitedVert  - stores if vertices have been visited or not
 * foundDest    - to signal if destination has been found
 * numVert      - number of vertices in graph
 * destination  - destination vertex
 **/
__global__ void BFSLevels(int  *vertices,
                          int  *edges,
                          int  *distances,
                          int  *predecessors,
                          int  *vertIndices,
                          int  *edgeSize,
                          bool *levels,
                          bool *visitedVertices,
                          bool *foundDest,
                          int   numVert,
                          int   destination)
{
    // Grab ThreadID
    int thrID = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ bool destFound;
    destFound = false;

    if (thrID < numVert && !destFound)
    {
        int curVert = vertices[thrID];

        // Iterate through level if true
        if (levels[curVert])
        {
            levels[curVert]          = false;
            visitedVertices[curVert] = true;

            // Grab indexes for curVert edges in edge array
            int edgesEnd  = edgeSize[thrID] + vertIndices[thrID];  

            // Iterate through all edges for current vertex
            for (int edgeIter = vertIndices[thrID]; edgeIter < edgesEnd; ++edgeIter)
            {
                // Grab next Vertex at end of edge
                int nextVert = edges[edgeIter];

                // If it hasn't been visited store info 
                // for distance and predecessors and set level
                // to true for next level of vertices
                if (!visitedVertices[nextVert])
                {       
                    distances[nextVert] = distances[curVert] + 1;
                    levels[nextVert] = true;
                    predecessors[nextVert]  = curVert; 
      
                    // Set found destination to true and sync threads
                    if (nextVert == destination) 
                    {
                        *foundDest = true;
                        destFound  = true;
                        __syncthreads();
                    }
                }
            }
        }
    }    
}

/**
 * Runs BFS on GPU by searching level by level
 * vertices    - list of vertices for GPU
 * edges       - list of edge destinations for GPU
 * vertIndices - list of start points for each vertices edges in edge list
 * edgeSize    - list of how many edges each vertex has
 * destination - destination vertex
 * source      - source vertex
 **/
float BFSByLevel(std::vector<int> &vertices,
                 std::vector<int> &edges,
                 std::vector<int> &vertIndices,
                 std::vector<int> &edgeSize,
                 int               destination,
                 int               source)
{
    int numVertices = vertices.size();

    int arraySizeInBytes     = sizeof(int) * numVertices;
    int arraySizeInBytesBool = sizeof(bool) * numVertices;

    // Create Host Arrays to pass into kernel call for BFS
    bool *h_visitedVertices;
    bool *h_levels;
    int  *h_distances;
    int  *h_predecessors;

    h_distances        = (int *)  malloc(arraySizeInBytes); 
    h_predecessors     = (int *)  malloc(arraySizeInBytes);
    h_visitedVertices  = (bool *) malloc(arraySizeInBytesBool); 
    h_levels           = (bool *) malloc(arraySizeInBytesBool); 

    // Initializes arrays data
    for (int vertexIter = 0; vertexIter < numVertices; ++vertexIter)
    {
        h_visitedVertices[vertexIter] = false;
        h_levels[vertexIter]          = false;
        h_predecessors[vertexIter]    = -1;
    }

    h_levels[source]    = true;
    h_distances[source] = 0;

    cudaEvent_t start = get_time();

    // Create device arrays
    bool *d_visitedVertices;
    bool *d_levels;
    int  *d_distances;
    int  *d_predecessors;

    // use thrust library for host vectors
    thrust::device_vector<int> d_vertices(vertices);
    thrust::device_vector<int> d_edges(edges);
    thrust::device_vector<int> d_vertIndices(vertIndices);
    thrust::device_vector<int> d_edgeSize(edgeSize);

    cudaError_t result;

    // allocate other device arrays
    result  = cudaMalloc((void**) &d_distances, arraySizeInBytes);
    if (result > 0)
    {
        printf("ERROR -- CUDAMALLOC failed to allocate memory\n");
        return 0.0;
    }

    result = cudaMalloc((void**) &d_predecessors, arraySizeInBytes);
    if (result > 0)
    {
        printf("ERROR -- CUDAMALLOC failed to allocate memory\n");
        cudaFree(d_distances);
        return 0.0;
    }

    result = cudaMalloc((void**) &d_levels, arraySizeInBytesBool);
    if (result > 0)
    {
        printf("ERROR -- CUDAMALLOC failed to allocate memory\n");
        cudaFree(d_distances);
        cudaFree(d_predecessors);
        return 0.0;
    }

    result = cudaMalloc((void**) &d_visitedVertices, arraySizeInBytesBool);
    if (result > 0)
    {
        printf("ERROR -- CUDAMALLOC failed to allocate memory\n");
        cudaFree(d_distances);
        cudaFree(d_predecessors);
        cudaFree(d_levels);
        return 0.0;
    } 

    // Copy memory to device from host
    cudaMemcpy(d_levels,           h_levels,           arraySizeInBytesBool, cudaMemcpyHostToDevice);
    cudaMemcpy(d_visitedVertices,  h_visitedVertices,  arraySizeInBytesBool, cudaMemcpyHostToDevice);
    cudaMemcpy(d_predecessors,     h_predecessors,     arraySizeInBytes,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances,        h_distances,        arraySizeInBytes,     cudaMemcpyHostToDevice);

    // Specify block count and threads, each vertex gets a thread
    int blockCount = 1;
    int numThreads = numVertices;

    // Use pinned memory to store whether destination was found or not
    bool *h_foundDest;

    result = cudaHostAlloc((void**)&h_foundDest, sizeof(bool), cudaHostAllocDefault);

    if (result > 0)
    {
        printf("ERROR -- CUDAHOSTALLOC failed to allocate memory\n");

        cudaFree(d_distances);
        cudaFree(d_predecessors);
        cudaFree(d_levels);
        cudaFree(d_visitedVertices);
        return 0.0;
    }

    h_foundDest = false;

    bool *d_foundDest;

    // Copy pinned memory to device
    result = cudaMalloc((void**) &d_foundDest,    sizeof(bool));

    if (result > 0)
    {
        printf("ERROR -- CUDAMALLOC failed to allocate memory for destination\n");


        cudaFree(d_distances);
        cudaFree(d_predecessors);
        cudaFree(d_levels);
        cudaFree(d_visitedVertices);
        return 0.0;
    }

    cudaMemcpy(d_foundDest,  &h_foundDest,  sizeof(bool), cudaMemcpyHostToDevice);

    cudaEvent_t memFinished = get_time();

    // Run BFS algorithm going through each level
    int runCount = 0;
    while (runCount < numVertices)
    {
        BFSLevels<<<blockCount, numThreads>>>(thrust::raw_pointer_cast(d_vertices.data()),
                                              thrust::raw_pointer_cast(d_edges.data()),
                                              d_distances,
                                              d_predecessors,
                                              thrust::raw_pointer_cast(d_vertIndices.data()),
                                              thrust::raw_pointer_cast(d_edgeSize.data()),
                                              d_levels,
                                              d_visitedVertices,
                                              d_foundDest,
                                              numVertices,
                                              destination);

        runCount++;
    }

    cudaEvent_t kernelFinished = get_time();

    // Copy Back Results
    cudaMemcpy(&h_foundDest, d_foundDest, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_distances, d_distances, arraySizeInBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_predecessors, d_predecessors, arraySizeInBytes, cudaMemcpyDeviceToHost);

    // Free Device Memory
    cudaFree(d_distances);
    cudaFree(d_levels);
    cudaFree(d_visitedVertices);
    cudaFree(d_predecessors);

    cudaFree(d_foundDest);

    cudaEvent_t end = get_time();

    cudaEventSynchronize(end);

    float totalTime;
    cudaEventElapsedTime(&totalTime, start, end);

    float memAllocTime;
    cudaEventElapsedTime(&memAllocTime, start, memFinished);

    float kernelRunTime;
    cudaEventElapsedTime(&kernelRunTime, memFinished, kernelFinished);

    float copyMemBack;
    cudaEventElapsedTime(&copyMemBack, kernelFinished, end);

    // Clean Up Events
    cudaEventDestroy(start);
    cudaEventDestroy(memFinished);
    cudaEventDestroy(kernelFinished);
    cudaEventDestroy(end);
    
    printf("GPU Memory Allocation time %f\n", memAllocTime);
    printf("GPU Kernenl Runtime %f \n", kernelRunTime);
    printf("GPU Copy Memory Back %f \n", copyMemBack);
  

    std::vector<int> path;

    if (h_foundDest)
    {
        // Push destination onto path
        path.push_back(destination);
        int pointer = destination;

        // Push each predecessor that was visited onto path
        while (h_predecessors[pointer] != -1) 
        { 
            path.push_back(h_predecessors[pointer]); 
            pointer = h_predecessors[pointer];
        }       
              
        // printing path from source to destination 
        printf("\nShortest Path Length is %zd\n", path.size()); 
        int count = 0;
        int iter = path.size() - 1;
        while (count < path.size()) 
        {
            std::cout << path[iter] << " "; 
            count++;
            iter--;
        }
        std::cout << std::endl;
    }
    else
    {
        printf("Shortest Path not found to destination %d using GPU BFS Search by levels  \n", destination); 
    }

    // Free Host Memory
    free(h_visitedVertices);
    free(h_levels);
    free(h_distances);
    free(h_predecessors);

    cudaFreeHost(h_foundDest);

    return totalTime;

}