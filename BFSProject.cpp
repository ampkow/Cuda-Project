// Sources 
// https://www.geeksforgeeks.org/shortest-path-unweighted-graph/
// https://en.wikipedia.org/wiki/Breadth-first_search


#include <list> 
#include <vector> 
#include <iostream>
using namespace std; 

#include "KernelFunctions.h"
  
// utility function to form edge between two vertices 
// source and dest 
void add_edge(vector<vector<int> > &graph, int src, int dest) 
{ 
    graph[src].push_back(dest); 
    graph[dest].push_back(src); 
} 
  
// a modified version of BFS that stores predecessor 
// of each vertex in array p 
// and its distance from source in array d 
bool BFS(vector<vector<int>> &graph, 
         int                  src, 
         int                  dest, 
         int                  vertexSize, 
         vector<int>         &pred, 
         vector<int>         &dist) 
{ 
    // a queue to maintain queue of vertices whose 
    // adjacency list is to be scanned as per normal 
    // DFS algorithm 
    list<int> queue; 
  
    // boolean array visited[] which stores the 
    // information whether ith vertex is reached 
    // at least once in the Breadth first search 
    vector<bool> visited;

    visited.resize(vertexSize); 
  

    for (int iter = 0; iter < vertexSize; iter++) 
    { 
        visited[iter] = false; 
        dist[iter] = 0; 
        pred[iter] = -1; 

       //printf("Edges Count %d\n", graph.at(iter).size());
    } 
  
    // now source is first to be visited and 
    // distance from source to itself should be 0 
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
  
// utility function to print the shortest distance  
// between source vertex and destination vertex 
void BFSComputeShortedDist(vector<vector<int>> graph, 
                           int source,  
                           int dest, 
                           int vertexSize) 
{ 
    // predecessor[i] array stores predecessor of 
    // i and distance array stores distance of i
    // from s 
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
    cout << "Shortest Path Length is " << dist[dest]; 

    FindShortestPath(path, pred, dest);

    cout << endl;
} 
  
// Main Program 
int main(int argc, char const *argv[]) 
{ 
    // Default Vertex Size
    int vertexSize = 8;  
    int source     = 0;
    int dest       = 7;

    if (argc >= 2)
    {
        vertexSize = atoi(argv[1]);
    }

    if (argc >= 3)
    {
        source = atoi(argv[2]);
    }

    if (argc >= 4)
    {
        dest = atoi(argv[3]);
    }
  
    // Vector of vectors to store graph
    vector<vector<int> > vertices; 
// need to think about blasdf
// balsdlfskafasdf sddfa
// Think how to access the vector and so forth
// 
// can we use thrust for the cu side?

    // Initialize vertex to correct size
    vertices.resize(vertexSize);

    // need to read in vertices
  
    // Creating graph given in the above diagram. 
    // add_edge function takes adjacency list, source  
    // and destination vertex as argument and forms 
    // an edge between them. 

    int totalEdges = 0;
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

    // CPU
    BFSComputeShortedDist(vertices, source, dest, vertexSize); 

    // GPU - these two
    //RunBFSShortestDistance(vertices, dest, source, totalEdges);

    // run_nvgraph_search(8);

    return 0; 
} 
