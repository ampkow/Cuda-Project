// Sources 
// Using basic BFS algorithm defined from websites below
// https://www.geeksforgeeks.org/shortest-path-unweighted-graph/
// https://en.wikipedia.org/wiki/Breadth-first_search
// Used to help in Design of CUDA Algorithm
// https://www.nvidia.co.uk/content/cudazone/CUDABrowser/downloads/Accelerate_Large_Graph_Algorithms/HiPC.pdf

// Author: Adam Piorkowski
// Runs BFS for shortest path using CPU and GPU algorithms
// Algorithm takes in graph using data points from an input file
// and populates different data structures for CPU BFS and GPU BFS.
// Timing of the runs is outputted at the end of the algorithm to 
// Compare CPU timing vs GPU

#include <algorithm>
#include <list> 
#include <vector> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

using namespace std; 
using namespace std::chrono;

// Has Kernel Functionality for Cuda and the GPU Calls
#include "KernelFunctions.h"
  
// Adds new edges to the graph
void createNewEdge(vector<vector<int> > &graph, int src, int dest) 
{ 
    graph[src].push_back(dest); 
    if (dest != -4)
    {
        graph[dest].push_back(src); 
    }
} 
  
/** Runs BFS and calculates distances on CPU
 *  Graph        - Contains all vertices and their edges in vector format
 *  Source       - Starting point of search
 *  Destination  - Ending spot of shortest path
 *  VertexSize   - Number of Vertices
 *  Distances    - Vector containing distances from source to each Vertex 
 *  Predecessors - Vector containing list of predecessor vertice
 * */
bool BFS(vector<vector<int>> &graph, 
         int                  source, 
         int                  destination, 
         int                  vertexSize, 
         vector<int>         &predecessors, 
         vector<int>         &distances) 
{ 
    
    // Acts as queue for next vertex to traverse through
    list<int> nextVertList; 
  
    // Indicates if a vertex has been visited yet or not
    vector<bool> visitedVertices;

    visitedVertices.resize(vertexSize);  

    // Initialize values
    for (int iter = 0; iter < vertexSize; iter++) 
    { 
        visitedVertices[iter] = false; 
        distances[iter] = 0; 
        predecessors[iter] = -1; 
    } 
  
    // Initialize Source
    visitedVertices[source] = true; 
    distances[source] = 0; 
    nextVertList.push_back(source); 
  
    // BFS 
    while (!nextVertList.empty()) 
    {  
        // Grab next vertex to search
        int currVertIter = nextVertList.front(); 

        // Remove vertex from list
        nextVertList.pop_front(); 

        for (int iter = 0; iter < graph.at(currVertIter).size(); iter++) 
        { 
            // New Vertex being explored
            int nextVert = graph.at(currVertIter)[iter];
            if (visitedVertices[nextVert] == false) 
            { 
                // Set necessary data and add to list to explore edges
                // of the nextVertex
                visitedVertices[nextVert] = true; 
                distances[nextVert]       = distances[currVertIter] + 1; 
                predecessors[nextVert]    = currVertIter; 

                nextVertList.push_back(nextVert); 
  
                // Found desired destination, so search can stop
                if (nextVert == destination) 
                {
                   return true; 
                }
            } 
        } 
    } 
  
    // Never found Desired Destination in the graph
    return false; 
} 

/* *
 *   Iterates through vertex predecessor list
 *   and finds the shortest path to dest
 * */
void FindShortestPath(vector<int> &path,
                      vector<int> &predecessors,
                      int          dest)
{
    int pointer = dest;
 
    while (predecessors[pointer] != -1) 
    { 
        path.push_back(predecessors[pointer]); 
        pointer = predecessors[pointer]; 
    } 
      
    // printing path from end of path to source
    printf("\nShortest Path: \n"); 
    for (int iter = path.size() - 1; iter >= 0; iter--) 
    {
        printf("%d ", path[iter]); 
    }
}
  
  /**
   *  Sets up parameters and calls BFS CPU algorithm
   *  looking for the destination values starting
   *  from the source value
   * 
   * */ 
double BFSComputeShortedDist(vector<vector<int>> graph, 
                            int source,  
                            int destination, 
                            int vertexSize) 
{ 
    std::vector<int> predecessors;
    std::vector<int> distances;

    predecessors.resize(vertexSize);
    distances.resize(vertexSize);


    high_resolution_clock::time_point start = high_resolution_clock::now();

    bool foundDest = BFS(graph, source, destination, vertexSize, predecessors, distances);
  
   high_resolution_clock::time_point stop = high_resolution_clock::now();
   auto totalTime = duration_cast<milliseconds>( stop - start ).count();
   if (foundDest == false) 
    { 
        printf("Shortest Path not found to destination %d \n", destination); 
        return 0.0; 
    } 

    std::vector<int> path;
    path.push_back(destination);

    printf("Shortest Path Length is %d \n", distances[destination]); 

    FindShortestPath(path, predecessors, destination);

    printf("\n");

    return totalTime;
} 

/**
 * Reads in vertices and adds them to vector of vectors
 * Also populates a vertex and edge vector for the GPU algorithms
 * 
 * fileName    - File to be read containing data points
 * graph       - vector of vectors to be populated for CPU
 * vertices    - list of vertices for GPU
 * edges       - list of edge destinations for GPU
 * vertIndices - list of start points for each vertices edges in edge list
 * edgeSize    - list of how many edges each vertex has
 * destination - destination vertex
 * source      - source vertex
 * totalEdges  - total number of edges in path
 **/ 
void ReadInFile(std::string          fileName,
               vector<vector<int> > &graph,
               vector<int>          &vertices,
               vector<int>          &edges,
               vector<int>          &vertIndices,
               vector<int>          &edgeSize,
               int                  &destination,
               int                  &source,
               int                  &totalEdges)
{

    printf("FileName: %s \n", fileName.c_str());
    std::ifstream inputFile(fileName);

    // Read in edges
    int val1;
    int val2;

    while (inputFile >> val1 >> val2)
    {
        printf("Val1 %d, Val2 %d\n", val1, val2);
        if (val2 == -1)           // Designates number of Vertices
        {
            graph.resize(val1);
        }
        else if (val2 == -2)      // Designates source
        {
            source = val1;
        }
        else if (val2 == -3)      // Designates destination vertex
        {
            destination = val1;
        }
        else
        {
            // Create Graph for CPU
            totalEdges++;
            createNewEdge(graph, val1, val2); 
        }
    }

    printf ("Finished Reading in Graph\n\n");

    // Populate data structures for GPU code
    int count = 0;
    for (int graphIter = 0; graphIter < graph.size(); ++graphIter)
    {
        vertices.push_back(graphIter);
        edgeSize.push_back(graph[graphIter].size());
        vertIndices.push_back(count);
        for (int graphIter2 = 0; graphIter2 < graph[graphIter].size(); ++graphIter2)
        {
            edges.push_back(graph[graphIter][graphIter2]);
            count++;
        }
    }
}

// Prints RunTime Totals
void PrintRunTimeTotal(float cpuRunTime, float thrustRunTime, float gpuRunTime)
{
    printf("Total Runtime Analysis_________________\n");
    printf("The CPU BFS runtime was %f\n", cpuRunTime);
    printf("The GPU Thrust BFS runtime was %f\n", thrustRunTime);
    printf("The GPU BFS runtime was %f\n", gpuRunTime);
    printf("________________________________________\n");
}
  
// Main Program 
int main(int argc, char const *argv[]) 
{ 

    std::string graphFile = "";
    if (argc >= 2)
    {
        graphFile = argv[1];
    }
    else
    {
        printf("Specify a file to define the graph!!! \n");

        return 0;
    }
  
    // Vector of vectors to store graph
    vector<vector<int> > vertices; 

    int destination = 0;
    int source      = 0;
    int totalEdges  = 0;

    // Vectors to store data in format for GPU algorithm
    vector<int> vertex; 
    vector<int> edges; 
    vector<int> vertIndices; 
    vector<int> edgeSize; 

    // need to read in vertices
    ReadInFile(graphFile,
               vertices,
               vertex,
               edges,
               vertIndices,
               edgeSize,
               destination,
               source,
               totalEdges);

    int vertexSize = vertices.size();

    if (vertexSize == 0)
    {
        printf("Graph Size of 0 \n");
        return 0;
    }


    // CPU
    printf("Running BFS on CPU\n");
    float cpuRunTime = BFSComputeShortedDist(vertices, source, destination, vertexSize); 
    printf("\n");

    // GPU - One using Thrust, another making kernel calls
    printf("Running BFS on GPU\n\n");
    printf("Running BFS with Thrust___________");
    float thrustRunTime = RunBFSUsingThrust(vertices, destination, source, totalEdges);

    printf("\n");

    printf("Running BFS iterating through Levels__________\n");
    float gpuRunTime = BFSByLevel(vertex,
                                  edges,
                                  vertIndices,
                                  edgeSize,
                                  destination,
                                  source);

    printf("\n");

    PrintRunTimeTotal(cpuRunTime, thrustRunTime, gpuRunTime);

    return 0; 
} 