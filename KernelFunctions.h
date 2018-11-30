// Kernel Functions.h

#include <vector>

float RunBFSShortestDistance(std::vector<std::vector<int> > &graph,
                             int                             dest,
                             int                             source,
                             int                             totalEdges);


float RunBFSUsingStreams(std::vector<std::vector<int> > &graph,
                        int                             dest,
                             int                             source,
                             int                             totalEdges);

// Runs BFS on a generated graph and prints out
 // distances
 float run_nvgraph_search(int numVerticies);