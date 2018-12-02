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
__global__ void BFSThrust(int  *graph, 
    int  *numEdges,     // arrray for number of edges for each vertex
    int  *queue,        // queue of edges to be searched
    bool *visited,      // array to see if edges have been visited
    int   source,
    int   dest,
    int   vertexSize,
    int   maxEdges,
    int  *pred,
    int  *dist,
    bool  &found,
    int   totalEdges)    // Total number of edges in the graph
{

    //printf("We inside kernel %d, %d, %d\n", graph[0 + 0 + maxEdges], graph[2 + 2 + maxEdges], graph[1 + 1 + maxEdges]);
    for (int iter = 0; iter < vertexSize; iter++) 
    { 
        visited[iter] = false; 
        dist[iter]    = 0; 
        pred[iter]    = -1; 

        //printf("Edges Count %d\n", numEdges[iter]);
    } 

    for (int test = 0; test < vertexSize * maxEdges; ++test)
    {
        int i = test/maxEdges;
        int j = test - (i * maxEdges);
        int in = i * maxEdges + j;
        printf("Index %d, %d with Value %d ...\n", test, in,  graph[i * maxEdges + j]);
    }

    printf("\n");

    visited[source] = true; 
    dist[source] = 0; 
    queue[0] = source; 

    // BFS algorithm  

    // Pointer for front of Queue
    int pointer     = 0;
    int backPointer = 1;

    printf("Total Edges %d, maxEdges%d \n", totalEdges, maxEdges);
    while (pointer != totalEdges) 
    {  
        int queueIter = queue[pointer]; 
        int iter = maxEdges * pointer;
        pointer++; 

       // printf("Current Vertex %d: \n", queueIter);
      //  printf("PTR %d, backPTR: %d\n", pointer, backPointer);
        //printf("Edges %d\n", numEdges[queueIter]);
        int count = 0;
        
        while (count < maxEdges )//int iter = maxEdges *pointer; iter < maxEdges; iter++) 
        { 
            
            // int index = iter * maxEdges + queueIter;
            int i = iter/maxEdges;
            int j = iter - (i * maxEdges);

            int index = i * maxEdges + j;
            printf("Current PTR %d, \n", index);
            if (graph[index] == -100)
            {
                // jump to next vertex
                count =  maxEdges;
                printf("Next Pointer %d,  %d\n", pointer, queue[pointer] );
            }
            else
            {

                if (visited[graph[index]] == false) 
                {    
                    // printf("We inside visit\n");
                    printf("Element (%d, %d, %d) = %d\n", queueIter, iter, index, graph[index]);
                    visited[graph[index]] = true; 
                    dist[graph[index]]    = dist[queueIter] + 1; 
                    pred[graph[index]]    = queueIter; 

                    queue[backPointer] = graph[index]; 

                    iter = index;
                    backPointer++;

                    // Stop When finding destination
                    if (graph[index] == dest) 
                    {
                        found =  true; 
                        printf("WE WON!!!!\n");
                    }
                } 
            }    
            count++;
            iter++;
        }
    } 
}

// Option 1
int *ConvertVectorTo2D(std::vector<std::vector<int> >  &graph,
    int                              maxEdges)
{
    int *h_graph;
    int graphSize = graph.size();
    h_graph = new int[graphSize * maxEdges];

//h_graph[0] = new int[graphSize * maxEdges];

// for(int iter1 = 1; (iter1 < graph.size()); ++iter1)
// { 
//    h_graph[iter1] = h_graph[iter1 - 1] + maxEdges;
// }

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

// Option 1
float RunBFSShortestDistance(std::vector<std::vector<int> > &graph,
    int                             dest,
    int                             source,
    int                             totalEdges)
{
    bool foundDest = false;

    int numVerticies = graph.size();

    int arraySizeVertices = sizeof(int) * numVerticies;
    int arraySizeEdges    = sizeof(int) * totalEdges;

    int maxEdges = 0;

    int *h_numEdges = (int *) malloc(arraySizeVertices); 

    for (int vectIter = 0; vectIter < numVerticies; vectIter++)
    {
        int edgeCount = graph[vectIter].size();
        h_numEdges[vectIter] = edgeCount;

        if (edgeCount > maxEdges)
        {
            maxEdges = edgeCount;
        }
    }

    printf("Converting Vectors of Vectors...\n");

    // Convert Vector of Vector to 2d array
    int* h_graph = ConvertVectorTo2D(graph, maxEdges);


    int graphSize = sizeof(int) * (graph.size() * maxEdges);

    printf("Done... %d\n", graphSize);

    bool *d_vertexVisited;   // array to see if edges have been visited
    int  *d_numEdges;        // arrray for number of edges for each vertex
    int  *d_edgeQueue;       // queue of edges to be searched
    int  *d_pred;            // array to store predecssors
    int  *d_dist;            // array to store distances

    int  *d_graph;

    printf("Before allocating graph\n");

    cudaMalloc((void**) &d_graph, graphSize);

    printf("Afer allocating graph\n");

    // Allocate  Global memory
    cudaMalloc((void**) &d_vertexVisited,  sizeof(bool) * numVerticies); //doesnt need populating
    cudaMalloc((void**) &d_numEdges,       arraySizeVertices);            // does
    cudaMalloc((void**) &d_edgeQueue,      arraySizeEdges);            // doesnt
    cudaMalloc((void**) &d_pred,           arraySizeVertices);            // doesnt
    cudaMalloc((void**) &d_dist,           arraySizeVertices);            // doesnt

    // After Cuda Malloc

    printf("After cuda malloc \n");

    cudaMemcpy(d_numEdges, h_numEdges, arraySizeVertices, cudaMemcpyHostToDevice);


    printf("After Edge Malloc \n");

    // allocate correctly and pass to kernel by making verticies pointers so memory on memory
    cudaMemcpy(d_graph, h_graph, graphSize, cudaMemcpyHostToDevice);

    printf("After Copy Graph \n");

    //thrust::device_vector<int> d_graph(numVerticies * totalEdges);
    //thrust::device_vector<int> d_graph(vertices);

    // thrust::copy(&(vertices[0][0]), &(vertices[numVerticies - 1][numVerticies - 1]), d_graph.begin());
    // printf("Afer copy\n");
    // thrust::sequence(d_graph.begin(), d_graph.end());

    printf("Before Kernel Call %d, %d, %d\n", h_graph[0], h_graph[1], h_graph[1]);

    // BFS call
    BFSThrust<<<1,1>>>(d_graph /*thrust::raw_pointer_cast(d_graph.data())*/,
                   d_numEdges,
                   d_edgeQueue, 
                   d_vertexVisited, 
                   source,
                   dest,
                   numVerticies,
                   maxEdges,
                   d_pred,
                   d_dist,
                   foundDest,
                   totalEdges);


    printf("After Kernel Call\n");
    cudaDeviceSynchronize();

    cudaFree(d_vertexVisited);
    cudaFree(d_numEdges);
    cudaFree(d_edgeQueue);
    cudaFree(d_pred);
    cudaFree(d_dist);

    free(h_numEdges);
    //free(vertices);

    return 0.0;
}

// Option 2

__device__ void FillVector(int *vector, int current, int val)
{
    vector[current] = val;
}


// pass edge count as size to iterate through, pass current index
// return value next vect.atqueuIter pass in and pass back queueIter[iter]
// vec  = graph.at(queueIter)
// dist = just to set values
// pred = just to set values
// visited
// bool = if found
// retun val for push back 
// dest need to be found
__global__ void IterateEdges(int *edges, 
    int  *dist, 
    int  *pred, 
    bool *visited,
    int  *nextIndex, 
    bool &foundDest, 
    int   edgeCount, 
    int   currentVertex, 
    int   dest)
{
    int current = threadIdx.x + blockIdx.x * blockDim.x;

    FillVector(nextIndex, current, -1);

    while (current < edgeCount)
    {
        int nextVert = edges[current];
        if (visited[nextVert] == false) 
        { 
            printf("Element (%d, %d) = %d with edgeCount%d\n", currentVertex, current, nextVert, edgeCount);

            visited[nextVert] = true; 
            dist[nextVert]    = dist[currentVertex] + 1; 
            pred[nextVert]    = currentVertex; 

            nextIndex[current] = nextVert;

            // Stop When finding destination
            if (nextVert == dest) 
            {
                printf("We won!!\n");
                foundDest = true;
                return; 
            }
        }
    }
}

// Option 2
float RunBFSUsingStreams(std::vector<std::vector<int> > &graph,
    int                             dest,
    int                             source,
    int                             totalEdges)
{

    std::list<int> queue; 

    int vertexSize = graph.size();

    bool foundDest = false;

    int numVerticies = graph.size();

    int arraySizeVertices = sizeof(int) * numVerticies;

    int *h_nextIndex;

    h_nextIndex = (int *) malloc(arraySizeVertices);

    bool *d_vertexVisited;   // array to see if edges have been visited
    int  *d_pred;            // array to store predecssors
    int  *d_dist;            // array to store distances
    int  *d_nextIndex;       // array to store 


    // Allocate  Global memory
    cudaMalloc((void**) &d_vertexVisited,  sizeof(bool) * numVerticies); //doesnt need populating
    cudaMalloc((void**) &d_pred,           arraySizeVertices);            // doesnt
    cudaMalloc((void**) &d_dist,           arraySizeVertices);            // doesnt
    cudaMalloc((void**) &d_nextIndex,      arraySizeVertices);            // doesnt

    queue.push_back(source);

    // BFS algorithm  
    while (!queue.empty()) 
    {  
        int queueIter = queue.front();  // Current Vertex
        queue.pop_front(); 

        int edgeCount = graph.at(queueIter).size();

        // need to populate
        thrust::device_vector<int> d_edges(graph.at(queueIter));

        IterateEdges<<<1, edgeCount>>>(thrust::raw_pointer_cast(d_edges.data()), 
            d_dist, 
            d_pred, 
            d_vertexVisited,
            d_nextIndex, 
            foundDest, 
            edgeCount, 
            queueIter, 
            dest);

        cudaMemcpy(h_nextIndex, d_nextIndex, arraySizeVertices, cudaMemcpyDeviceToHost);

        for (int nextIter = 0; nextIter < numVerticies; ++nextIter)
        {
            if (h_nextIndex[nextIter] != -1)
            {
                queue.push_back(h_nextIndex[nextIter]);
                printf("Use nextIndex %d", h_nextIndex[nextIter]);
            }
        }

        if (foundDest)
        {
            // Free memory
            cudaFree(d_vertexVisited);
            cudaFree(d_pred);
            cudaFree(d_dist);
            cudaFree(d_nextIndex);
            return true;
        }
    } 

    // Free memory
    cudaFree(d_vertexVisited);
    cudaFree(d_pred);
    cudaFree(d_dist);
    cudaFree(d_nextIndex);

    return false; 
}

// Option 3

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