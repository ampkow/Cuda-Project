// Kernel Functions.h

#include <vector>

float RunBFSShortestDistance(std::vector<std::vector<int> > &graph,
                             int                             dest,
                             int                             source,
                             int                             totalEdges);


float RunBFSUsingThrust(std::vector<std::vector<int> > &graph,
                        int                             dest,
                        int                             source,
                        int                             totalEdges);
