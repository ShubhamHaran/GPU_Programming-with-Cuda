#include <iostream>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Kernel function to add the elements of two arrays
__global__
void add(int n, float* x, float* y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    float* x, * y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 2.0f;
        y[i] = 1.0f;
    }
    int b=8,t=1024;

    // Run kernel on 1M elements on the GPU
    add <<<b, t>>> (N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout<<"Time for the GPU kernel of blocksize "<<b<< " thread size "<<t<<" number of elements "<<N<<": "<<time<<" ms"<<std::endl;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < N; i++) {
        x[i] = 2.0f;
        y[i] = 1.0f;
    }
    b=1;t=1;

    // Run kernel on 1M elements on the GPU
    add <<<b, t>>> (N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    std::cout<<"Time for the CPU kernel: "<<time<<" ms"<<std::endl;
    

    

    // Free memory
    cudaFree(x);
    cudaFree(y);
    return 0;
}
