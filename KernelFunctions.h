// Kernel Functions.h

#include <vector>

float RunBFSUsingThrust(std::vector<std::vector<int> > &graph,
                        int                             dest,
                        int                             source,
                        int                             totalEdges);


float BFSByLevel(std::vector<int> &vertices,
                 std::vector<int> &edges,
                 std::vector<int> &vertIndices,
                 std::vector<int> &edgeLength,
                 int               destination,
                 int               source);
