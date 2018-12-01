// Sources 
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

#include "KernelFunctions.h"
  
// Adds new edges to the graph
void createNewEdge(vector<vector<int> > &graph, int src, int dest) 
{ 
    graph[src].push_back(dest); 
    graph[dest].push_back(src); 
} 
  
// Runs BFS and calculates distances from src to the dest
bool BFS(vector<vector<int>> &graph, 
         int                  src, 
         int                  dest, 
         int                  vertexSize, 
         vector<int>         &pred, 
         vector<int>         &dist) 
{ 
    // List of next vertices to be scanned
    list<int> queue; 
  
    vector<bool> visited;

    visited.resize(vertexSize); 
  

    for (int iter = 0; iter < vertexSize; iter++) 
    { 
        visited[iter] = false; 
        dist[iter] = 0; 
        pred[iter] = -1; 

    } 
  
    visited[src] = true; 
    dist[src] = 0; 
    queue.push_back(src); 
  
    // BFS algorithm  
    while (!queue.empty()) {  
        int queueIter = queue.front(); 
        queue.pop_front(); 
        for (int iter = 0; iter < graph.at(queueIter).size(); iter++) 
        { 
            if (visited[graph.at(queueIter)[iter]] == false) 
            { 
                printf("Element (%d, %d) = %d\n", queueIter, iter, graph.at(queueIter)[iter]);

                visited[graph.at(queueIter)[iter]] = true; 
                dist[graph.at(queueIter)[iter]]    = dist[queueIter] + 1; 
                pred[graph.at(queueIter)[iter]]    = queueIter; 


                queue.push_back(graph.at(queueIter)[iter]); 
  
                // Stop When finding destination
                if (graph[queueIter][iter] == dest) 
                {
                   return true; 
                }
            } 
        } 
    } 
  
    return false; 
} 

void FindShortestPath(vector<int> &path,
                      vector<int> &pred,
                      int          dest)
{
    int pointer = dest;
 

    while (pred[pointer] != -1) 
    { 
        path.push_back(pred[pointer]); 
        pointer = pred[pointer]; 
    } 
      
    // printing path from source to destination 
    cout << "\nShortest Path: \n"; 
    for (int iter = path.size() - 1; iter >= 0; iter--) 
    {
        cout << path[iter] << " "; 
    }
}
  
void BFSComputeShortedDist(vector<vector<int>> graph, 
                           int source,  
                           int dest, 
                           int vertexSize) 
{ 
    std::vector<int> pred;
    std::vector<int> dist;

    pred.resize(vertexSize);
    dist.resize(vertexSize);
  
    if (BFS(graph, source, dest, vertexSize, pred, dist) == false) 
    { 
        cout << "Shortest Path not found"; 
        return; 
    } 

    std::vector<int> path;
    path.push_back(dest);

    // distance from source is in distance array 
    cout << "Shortest Path Length is " << dist[dest] << " \n"; 

    FindShortestPath(path, pred, dest);

    cout << endl;
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
        cout << "Specify a file to define the graph!!!" << endl;

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
        cout << "Graph Size of 0" << endl;
        return 0;
    }


    // CPU
    printf("Running BFS on CPU\n");
    BFSComputeShortedDist(vertices, source, destination, vertexSize); 
    printf("\n");

    // GPU - these two
    //RunBFSShortestDistance(vertices, dest, source, totalEdges);
    printf("Running BFS on GPU\n");
    RunBFSUsingThrust(vertices, destination, source, totalEdges);

    // run_nvgraph_search(8);

    return 0; 
} 
