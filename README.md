ASSIGNMENT # 9 NVgraph Program
Adam Piorkowski

Program_name: nvgraph_assign.cu

Notes: 
NOTE!!!! I had issues trying to make nvgraph_assign.cu compile on Linux, but
the program compiles on windows with 
nvcc.exe -lcudart -lcuda -lnvgraph -I ..\..\..\common\inc\ .\nvgraph_assign.cu -o .\nvgraph_assign

Running run_nvgraph.bat will compile the program and run it

The output of this is shown in output_nvgraph.txt

The program creates a simple graph with n verticies and n edges

Time outputs are found at the end of each run.

The different runs are:
nvgraph_assign 256
nvgraph_assign 100
nvgraph_assign 50
nvgraph_assign 10


output_nvgragh.txt shows the output of running "run_nvgraph.bat"
There is a time comparison for the four different runs.
There is total time output, traversal time output, time
it took to allocate the memory, and time it took to copy
back to device.

JPEG files 
nvprof_nvgraph.jpg shows output of nvprof



