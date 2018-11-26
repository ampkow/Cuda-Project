// Make Kernel Functions inside of here
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#include "nvgraph.h"

// STL
#include <vector> 
#include <iostream>

// Internal Headers
#include "KernelFunctions.h"

// Run BFS algorithm using thurst library

// __host__ allows to run on device and host and void code duplication
// taken from global_memory.cu
__host__ cudaEvent_t get_time(void)
{
    cudaEvent_t time;
    cudaEventCreate(&time);
    cudaEventRecord(time);
	return time;
}

// utility function to form edge between two vertices 
// source and dest 
void add_edge(int *graph, int src, int dest) 
{ 
    ++graph[src]  = dest; 
    ++graph[dest] = src; 
} 
  

__global__ void BFSThrust(int  *graph, 
                          int  *numEdges,     // arrray for number of edges for each vertex
                          int  *queue,        // queue of edges to be searched
                          bool *visited,      // array to see if edges have been visited
                          int   source,
                          int   dest,
                          int   vertexSize,
                          int  *pred,
                          int  *dist,
                          bool  &found,
                          int   totalEdges)    // Total number of edges in the graph
{

   printf("We inside kernel\n");
   for (int iter = 0; iter < vertexSize; iter++) 
   { 
       visited[iter] = false; 
       dist[iter]    = 0; 
       pred[iter]    = -1; 

       printf("Edges Count %d\n", numEdges[iter]);
   } 
 
   visited[source] = true; 
   dist[source] = 0; 
   queue[0] = source; 
 
   // BFS algorithm  

   // Pointer for front of Queue
   int pointer     = 0;
   int backPointer = 1;
   while (pointer < totalEdges) 
   {  
       printf("We inside queue\n");
       int queueIter = queue[pointer]; 
       pointer++; 
       printf("Edgs %d\n", numEdges[queueIter]);
       for (int iter = 0; iter < numEdges[queueIter]; iter++) 
       { 
           printf("test\n");
           printf("graph %d\n", graph[queueIter]);
           if (visited[graph[queueIter][iter]] == false) 
           {    
               printf("We inside visit\n");

               visited[graph[queueIter][iter]] = true; 
               dist[graph[queueIter][iter]]    = dist[queueIter] + 1; 
               pred[graph[queueIter][iter]]    = queueIter; 

               queue[backPointer] = graph[queueIter][iter]; 
               backPointer++;
 
               // Stop When finding destination
               if (graph[queueIter][iter] == dest) 
               {
                  found =  true; 
                  printf("WE WON!!!!\n");
                  return;
               }
           } 
       } 
   } 
}

__global__ void FindShortestPath(int *path,
                                 int *pred,
                                 int  dest)
{
    int pointer = dest;

    int pathSize = 0;
    for ( ;pred[pointer] != -1; pathSize++) 
    { 
        path[pathSize] = pred[pointer]; 
        pointer = pred[pointer]; 
    } 

    // printing path from source to destination 
    // std::cout << "\nShortest Path: \n"; 
    // for (int iter = pathSize - 1; iter >= 0; iter--) 
    // {
    //     std::cout << path[iter] << " "; 
    // }

}

float RunBFSShortestDistance(int numVerticies,
                             int dest,
                             int source)
{

    // Double Vector for verticies
    // Vector of vectors to store graph
    int *vertices = (int*) malloc(sizeof(int) * numVerticies * numVerticies); 

    // Initialize vertex to correct size
   // verticies.resize(numVerticies);

    // need to read in verticies
  
    // Creating graph given in the above diagram. 
    // add_edge function takes adjacency list, source  
    // and destination vertex as argument and forms 
    // an edge between them. 

    int totalEdges = 0;

    bool foundDest = false;

    totalEdges++;
    add_edge(vertices, 0, 1); 
    totalEdges++;
    add_edge(vertices, 0, 3);
    totalEdges++;
    add_edge(vertices, 1, 2); 
    totalEdges++;
    add_edge(vertices, 3, 4); 
    totalEdges++;
    add_edge(vertices, 3, 7); 
    totalEdges++;
    add_edge(vertices, 4, 5); 
    totalEdges++;
    add_edge(vertices, 4, 6); 
    totalEdges++;
    add_edge(vertices, 4, 7); 
    totalEdges++;
    add_edge(vertices, 5, 6); 
    totalEdges++;
    add_edge(vertices, 6, 7); 
    // int source = 0;
    // int dest   = 7;

    int arraySizeInBytes = sizeof(int) * numVerticies;

    bool *d_vertexVisited;   // array to see if edges have been visited
    int  *d_numEdges;        // arrray for number of edges for each vertex
    int  *d_edgeQueue;       // queue of edges to be searched
    int  *d_pred;            // array to store predecssors
    int  *d_dist;            // array to store distances

    int  *d_graph;

    printf("Before allocating graph\n");

    cudaMalloc((void**) &d_graph, sizeof(int) * numVerticies * numVerticies);

    printf("Afer allocating graph\n");

    int *h_numEdges = (int *) malloc(arraySizeInBytes); 

    // Allocate  Global memory
    cudaMalloc((void**) &d_vertexVisited,  sizeof(bool) * numVerticies); //doesnt need populating
    cudaMalloc((void**) &d_numEdges,       arraySizeInBytes);            // does
    cudaMalloc((void**) &d_edgeQueue,      arraySizeInBytes);            // doesnt
    cudaMalloc((void**) &d_pred,           arraySizeInBytes);            // doesnt
    cudaMalloc((void**) &d_dist,           arraySizeInBytes);            // doesnt

    // After Cuda Malloc

    printf("Afer cuda malloc \n");

    for (int vectIter = 0; vectIter < numVerticies; vectIter++)
    {
       // h_numEdges[vectIter] = vertices[vectIter].size();
    }

    cudaMemcpy(d_numEdges, h_numEdges, arraySizeInBytes, cudaMemcpyHostToDevice);

    // allocate correctly and pass to kernel by making verticies pointers so memory on memory
    cudaMemcpy(d_graph,    vertices, sizeof(int) * numVerticies * numVerticies, cudaMemcpyHostToDevice);

    printf("Afer edgething \n");

    // thrust::device_vector<int> d_graph(numVerticies * numVerticies);
    // thrust::copy(&(verticies[0][0]), &(verticies[numVerticies - 1][numVerticies - 1]), d_graph.begin());
    // thrust::sequence(d_graph.begin(), d_graph.end());
    // BFS call
    BFSThrust<<<1,1>>>(d_graph, 
                       d_numEdges,
                       d_edgeQueue, 
                       d_vertexVisited, 
                       source,
                       dest,
                       numVerticies,
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
    free(vertices);

    return 0.0;
}

// Runs BFS on a generated graph and prints out
 // distances may not worry about this
 float run_nvgraph_search(int numVerticies)
 {
     // Graph is in CSR format
     cudaEvent_t start = get_time();
 
     int verticies = numVerticies;
     int num_edges = numVerticies;
 
     printf("Num_edges %d\n", num_edges);
     int vertex_numsets = 2; 
 
     // Create Verticies
     int offsets_size_in_bytes = sizeof(int) * (verticies + 1);
     int *host_offsets = (int *) malloc(offsets_size_in_bytes); 
 
     int dest_size_in_bytes = sizeof(int) * (num_edges);
     int *host_dest = (int *) malloc(dest_size_in_bytes); 
 
     for (int offsetIter = 0; offsetIter < verticies; offsetIter++)
     {
         host_offsets[offsetIter] = offsetIter;
     }
 
     // Set Up Destinations
     for (int destIter = 0; destIter < num_edges; destIter++)
     {
         host_dest[destIter] = host_offsets[destIter + 1];
     }

     //source_offsets 	Array of size nvertices+1, where i element equals to the number of the first edge for this vertex in the list of all outgoing edges in the destination_indices array. Last element stores total number of edges
     //destination_indices 	Array of size nedges, where each value designates destanation vertex for an edge. 
     
     // Last Value of Offsets equal number of edges in graph
     host_offsets[verticies] = num_edges;
 
     // holds results
     int *host_distances = (int *) malloc(offsets_size_in_bytes);
 
     // nvgraph values
     nvgraphHandle_t d_graph_handle;
     nvgraphGraphDescr_t d_desc;
     nvgraphCSRTopology32I_t d_input;
     cudaDataType_t* dimT;
     int distances_index = 0;
     int predecessors_index = 1;
     dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
     dimT[distances_index] = CUDA_R_32I;
     dimT[predecessors_index] = CUDA_R_32I;
 
     //Create Graph
     nvgraphStatus_t status = nvgraphCreate (&d_graph_handle);
     if ((int)status != 0)   
     {
         printf("ERROR nvgraphCreate: %d\n", status);
         exit(0);
     }
 
     status = nvgraphCreateGraphDescr (d_graph_handle, &d_desc);
     if ((int)status != 0)   
     {
         printf("ERROR nvgraphCreateGraphDescr: %d\n", status);
         exit(0);
     }
 
     // Set graph properties
     d_input = (nvgraphCSRTopology32I_t) malloc(sizeof(struct nvgraphCSRTopology32I_st));
     d_input->nvertices = verticies;
     d_input->nedges = num_edges;
     d_input->source_offsets = host_offsets;
     d_input->destination_indices = host_dest;
 
     status = nvgraphSetGraphStructure(d_graph_handle, d_desc, (void*)d_input, NVGRAPH_CSR_32);
     if ((int)status != 0)   
     {
         printf("ERROR nvgraphSetGraphStructure: %d\n", status);
         exit(0);
     }
     
     status = nvgraphAllocateVertexData(d_graph_handle, d_desc, vertex_numsets, dimT);
 
     if ((int)status != 0)   
     {
        printf("ERROR nvgraphAllocateVertexData: %d\n", status);
        exit(0);
     }
 
     cudaEvent_t create_graph = get_time();
 
     int starter_vert = 1;
     // Setting the traverse param
     nvgraphTraversalParameter_t traversal_param;
     status = nvgraphTraversalParameterInit(&traversal_param);
     if ((int)status != 0)   
     {
        printf("ERROR nvgraphTraversalParameterInit: %d\n", status);
        exit(0);
     }
 
     status = nvgraphTraversalSetDistancesIndex(&traversal_param, distances_index); 
     if ((int)status != 0)   
     {
        printf("ERROR nvgraphTraversalSetDistancesIndex: %d\n", status);
        exit(0);
     }
    
     status = nvgraphTraversalSetPredecessorsIndex(&traversal_param, predecessors_index);
     if ((int)status != 0)   
     {
        printf("ERROR nvgraphTraversalSetPredecessorsIndex: %d\n", status);
        exit(0);
     }
 
     status = nvgraphTraversalSetUndirectedFlag(&traversal_param, false);
     if ((int)status != 0)   
     {
        printf("ERROR nvgraphTraversalSetUndirectedFlag: %d\n", status);
        exit(0);
     }
 
     printf("Graph Traverse set \n");
     
     status = nvgraphTraversal(d_graph_handle, d_desc, NVGRAPH_TRAVERSAL_BFS, &starter_vert, traversal_param);
     if ((int)status != 0)   
     {
        printf("ERROR nvgraphTraversal: %d\n", status);
        exit(0);
     }
     
     cudaEvent_t traverse_time = get_time();
 
     // Get result
     status = nvgraphGetVertexData(d_graph_handle, d_desc, (void*)host_distances, distances_index);
     if ((int)status != 0)   
     {
        printf("ERROR nvgraphTraversal: %d\n",status);
        exit(0);
     }
 
     cudaEvent_t data_retr_time = get_time();
 
     // Print distances for every verticies
     for (int iter1 = 0; iter1 < verticies; iter1++)  
     {
         printf("Distance to vertex %d: %i\n", iter1, host_distances[iter1]);
 //        printf("Predecessor of vertex %d: %i\n",iter1, host_predecessors[iter1]);
     }
    
 
     free(dimT);
     free(d_input);
     nvgraphDestroyGraphDescr (d_graph_handle, d_desc);
     nvgraphDestroy (d_graph_handle);
 
     
     cudaEvent_t end = get_time();
     cudaEventSynchronize(end);
 
     free(host_offsets);
     free(host_dest);
     free(host_distances);
 
     float graph_alloc = 0.0;
     cudaEventElapsedTime(&graph_alloc, start, create_graph);
 
     float traverse_setup = 0.0;
     cudaEventElapsedTime(&traverse_setup, create_graph, traverse_time);
 
     float result_ret_time = 0.0;
     cudaEventElapsedTime(&result_ret_time, traverse_time, data_retr_time);
 
     float total_time;
     cudaEventElapsedTime(&total_time, start, end);
 
     cudaEventDestroy(start);
     cudaEventDestroy(create_graph);
     cudaEventDestroy(traverse_time);
     cudaEventDestroy(data_retr_time);
     cudaEventDestroy(end);
 
     printf("Time to create graph in memory  %f \n", graph_alloc);
     printf("Run time traverse setup and BFS %f \n", traverse_setup);
     printf("Time to copy memory back to host %f \n", result_ret_time);
 
     return total_time;
 }