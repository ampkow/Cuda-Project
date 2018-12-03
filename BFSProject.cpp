// Sources 
// Using basic BFS algorithm defined from websites below
// https://www.geeksforgeeks.org/shortest-path-unweighted-graph/
// https://en.wikipedia.org/wiki/Breadth-first_search

// Author: Adam Piorkowski
// Runs BFS for shortest path using CPU and GPU algorithms


#include <list> 
#include <vector> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
using namespace std; 

// Has Kernel Functionality
#include "KernelFunctions.h"
  
// Adds new edges to the graph
void createNewEdge(vector<vector<int> > &graph, int src, int dest) 
{ 
    graph[src].push_back(dest); 
    graph[dest].push_back(src); 
} 
  
// Runs BFS and calculates distances
bool BFS(vector<vector<int>> &graph, 
         int                  source, 
         int                  destination, 
         int                  vertexSize, 
         vector<int>         &predecessors, 
         vector<int>         &distances) 
{ 
    
    list<int> nextVertList; 
  
    vector<bool> visitedVertices;

    visitedVertices.resize(vertexSize);  

    for (int iter = 0; iter < vertexSize; iter++) 
    { 
        visitedVertices[iter] = false; 
        distances[iter] = 0; 
        predecessors[iter] = -1; 
    } 
  
    visitedVertices[source] = true; 
    distances[source] = 0; 
    nextVertList.push_back(source); 
  
    // BFS 
    while (!nextVertList.empty()) 
    {  
        int currVertIter = nextVertList.front(); 
        nextVertList.pop_front(); 

        for (int iter = 0; iter < graph.at(currVertIter).size(); iter++) 
        { 
            int currVert = graph.at(currVertIter)[iter];
            if (visitedVertices[currVert] == false) 
            { 
                printf("Element (%d, %d) = %d\n", currVertIter, iter, currVert);

                visitedVertices[currVert] = true; 
                distances[currVert]       = distances[currVertIter] + 1; 
                predecessors[currVert]    = currVertIter; 

                nextVertList.push_back(currVert); 
  
                // Stop When finding destination
                if (currVert == destination) 
                {
                   return true; 
                }
            } 
        } 
    } 
  
    return false; 
} 

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
  
void BFSComputeShortedDist(vector<vector<int>> graph, 
                           int source,  
                           int destination, 
                           int vertexSize) 
{ 
    std::vector<int> predecessors;
    std::vector<int> distances;

    predecessors.resize(vertexSize);
    distances.resize(vertexSize);
  
    if (BFS(graph, source, destination, vertexSize, predecessors, distances) == false) 
    { 
        printf("Shortest Path not found"); 
        return; 
    } 

    std::vector<int> path;
    path.push_back(destination);

    printf("Shortest Path Length is %d \n", distances[destination]); 

    FindShortestPath(path, predecessors, destination);

    printf("\n");
} 

  // Reads in vertices and adds them to vector of vectors
void ReadInFile(std::string          fileName,
               vector<vector<int> > &graph,
               int                   &destination,
               int                   &source,
               int                   &totalEdges)
{

    printf("FileName: %s \n", fileName.c_str());
    std::ifstream inputFile(fileName);

    // Read in edges
    int val1;
    int val2;


    while (inputFile >> val1 >> val2)
    {
        if (val2 == -1)
        {
            graph.resize(val1);
        }
        else if (val2 == -2)
        {
            source = val1;
        }
        else if (val2 == -3)
        {
            destination = val1;
        }
        else
        {
             totalEdges++;
             createNewEdge(graph, val1, val2); 
        }
    }
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

    // need to read in vertices
    ReadInFile(graphFile,
               vertices,
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
    BFSComputeShortedDist(vertices, source, destination, vertexSize); 
    printf("\n");

    // GPU - One using Thrust, another making kernel calls
    printf("Running BFS on GPU\n");
    RunBFSShortestDistance(vertices, destination, source, totalEdges);
    RunBFSUsingThrust(vertices, destination, source, totalEdges);

    return 0; 
} 
