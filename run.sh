#!/bin/sh
echo "Starting Tests ----------------------------------"

make
# Run 3 different tests
BFS SampleGraph.txt
BFS MedSizeGraph.txt
BFS LargeSizeGraph.txt

echo "Ending Tests -------------------------------------"

