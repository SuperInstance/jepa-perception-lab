// GPU hello world — verify CUDA works on RTX 4050 (WSL2)
#include <stdio.h>

__global__ void hello() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        printf("Hello from GPU! Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
    }
}

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA devices found. Error: %s\n", cudaGetErrorString(err));
        printf("WSL2 GPU passthrough may need Windows-side NVIDIA driver update.\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== GPU Detected ===\n");
    printf("Name: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory: %.0f MB\n", prop.totalGlobalMem / 1e6);
    printf("Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf("Clock Rate: %.0f MHz\n", prop.clockRate / 1e3);
    printf("====================\n");
    
    hello<<<1, 64>>>();
    cudaDeviceSynchronize();
    
    printf("GPU is ready.\n");
    return 0;
}
