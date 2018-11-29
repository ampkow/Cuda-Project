// Make Kernel Functions inside of here
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


#include "nvgraph.h"

// STL
#include <vector> 
#include <list> 
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
                            bool &foundDest, 
                            int   edgeCount, 
                            int  &nextIndex, 
                            int   currentVertex, 
                            int   dest)
{
    int current = threadIdx.x + blockIdx.x * blockDim.x ;

    while (current < edgeCount)
    {
        int nextVert = edges[current];
        if (visited[nextVert] == false) 
        { 
            printf("Element (%d, %d) = %d\n", currentVertex, current, nextVert);

            visited[nextVert] = true; 
            dist[nextVert]    = dist[currentVertex] + 1; 
            pred[nextVert]    = currentVertex; 

            nextIndex = nextVert;

            // Stop When finding destination
            if (nextVert == dest) 
            {
                foundDest = true;
                return; 
            }
        } 
    }
}

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

    bool *d_vertexVisited;   // array to see if edges have been visited
    int  *d_pred;            // array to store predecssors
    int  *d_dist;            // array to store distances


    // Allocate  Global memory
    cudaMalloc((void**) &d_vertexVisited,  sizeof(bool) * numVerticies); //doesnt need populating
    cudaMalloc((void**) &d_pred,           arraySizeVertices);            // doesnt
    cudaMalloc((void**) &d_dist,           arraySizeVertices);            // doesnt

    int nextIndex = 0;

    // BFS algorithm  
    while (!queue.empty()) 
    {  
        int queueIter = queue.front();  // Current Vertex
        queue.pop_front(); 

        int edgeCount = graph.at(queueIter).size();

        // need to populate
        thrust::device_vector<int> d_edges(edgeCount);
        
        IterateEdges<<<1,1>>>(thrust::raw_pointer_cast(d_edges.data()), 
                           d_dist, 
                           d_pred, 
                           d_vertexVisited,
                           foundDest, 
                           edgeCount, 
                           nextIndex, 
                           queueIter, 
                           dest);

        queue.push_back(nextIndex);

        if (foundDest)
        {
            // Free memory
            cudaFree(d_vertexVisited);
            cudaFree(d_pred);
            cudaFree(d_dist);
            return true;
        }
    } 

    // Free memory
    cudaFree(d_vertexVisited);
    cudaFree(d_pred);
    cudaFree(d_dist);
  
    return false; 
}

 // Runs BFS on a generated graph and prints out
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

     // source_offsets Array of size nvertices+1, where i element equals to the number of the first edge for this vertex in the list of all outgoing edges in the destination_indices array. Last element stores total number of edges
     //     int source_offsets_h[] = {0, 1, 3, 4, 6, 8, 10, 12};
     //     int destination_indices_h[] = {5, 0, 2, 0, 4, 5, 2, 3, 3, 4, 1, 5};
     // destination_indices 	Array of size nedges, where each value designates destanation vertex for an edge. 
     
     // Last Value of Offsets equal number of edges in graph
     host_offsets[verticies] = num_edges;
 
     // holds results
     int *host_distances    = (int *) malloc(offsets_size_in_bytes);
     int *host_predecessors = (int *) malloc(offsets_size_in_bytes);
 
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

     status = nvgraphGetVertexData(d_graph_handle, d_desc, (void*)host_predecessors, predecessors_index);
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
         printf("Predecessor of vertex %d: %i\n",iter1, host_predecessors[iter1]);
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