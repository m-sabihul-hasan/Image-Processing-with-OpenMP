# Image-Processing-with-OpenMP-&-CUDA
Speeding up image processing with OpenMP API

To compile the serial/OpenMP code using the NVCC compiler enter the following command in the terminal:  
nvcc -Xcompiler "-std=c99 -O3 -fopenmp" image_processing.c -o image_processing

To compile the cuda code using the NVCC compiler enter the following command in the terminal:  
nvcc image_processing_cuda.cu -o cuda

To run the compiled file use the following command:  
./<name_of_compiled_file> <input_image>.ppm <output_image>.ppm <'1' for Blur or '2' for Edge Detection>


