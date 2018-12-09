Project:
Adam Piorkowski

Runs BFS Algorithms on three different graphs.

There are two algorithms running on GPU and one on CPU.

Timing results are printed at the end.

Program_name: BFSProject.cpp
GPU Code:     KernelFunctions.cu
Header:       KernelFunction.h
Executable:   BFS

On linux running make will compile the executable.

run.bat on Windows and run.sh on Linux will compile
and run the executable on the three graphs defined in
the text files:
    SampleGraph.txt
    MedSizeGraph.txt
    LargeSizeGraph.txt
Timing is outputted for the CPU and GPU algorithms at 
the end.


