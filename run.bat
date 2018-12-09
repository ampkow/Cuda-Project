echo "Starting Tests ----------------------------------"

nvcc.exe -lcudart -lcuda -I ..\..\..\common\inc\ .\BFSProject.cpp .\KernelFunctions.cu -o .\BFS

BFS SampleGraph.txt
BFS MedSizeGraph.txt
BFS LargeSizeGraph.txt

echo "Ending Tests -------------------------------------"