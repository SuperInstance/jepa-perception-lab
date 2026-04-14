#include <stdio.h>
#include <math.h>
#include <chrono>

// CT snap: find nearest Pythagorean coordinate (simplified 3-4-5 multiples)
__device__ void ct_snap(float x, float y, float *sx, float *sy) {
    // Snap to nearest Pythagorean triple ratio
    float r = sqrtf(x*x + y*y);
    if (r < 1e-6f) { *sx = 0; *sy = 0; return; }
    
    // Test common Pythagorean ratios
    float best_dist = 1e9f;
    int triples[][2] = {{3,4},{5,12},{8,15},{7,24},{20,21},{9,40},{12,35},{11,60}};
    
    for (int i = 0; i < 8; i++) {
        float a = triples[i][0], b = triples[i][1];
        float c = sqrtf(a*a + b*b);
        // Two orientations: (a,b) and (b,a)
        for (int flip = 0; flip < 2; flip++) {
            float pa = flip ? b : a, pb = flip ? a : b;
            float nx = pa/c, ny = pb/c;
            // Scale to match input magnitude
            float scale = r;
            float sx2 = nx * scale, sy2 = ny * scale;
            float dist = (x-sx2)*(x-sx2) + (y-sy2)*(y-sy2);
            if (dist < best_dist) { best_dist = dist; *sx = sx2; *sy = sy2; }
        }
    }
    // Also test axis-aligned and diagonals
    float axes[][2] = {{1,0},{0,1},{1,1},{-1,1}};
    for (int i = 0; i < 4; i++) {
        float nx = axes[i][0]/sqrtf(axes[i][0]*axes[i][0]+axes[i][1]*axes[i][1]);
        float ny = axes[i][1]/sqrtf(axes[i][0]*axes[i][0]+axes[i][1]*axes[i][1]);
        float sx2 = nx*r, sy2 = ny*r;
        float dist = (x-sx2)*(x-sx2) + (y-sy2)*(y-sy2);
        if (dist < best_dist) { best_dist = dist; *sx = sx2; *sy = sy2; }
    }
}

// Benchmark kernel: CT snap
__global__ void bench_ct_snap(float *input_x, float *input_y, float *output_x, float *output_y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) ct_snap(input_x[i], input_y[i], &output_x[i], &output_y[i]);
}

// Benchmark kernel: float multiply (baseline)
__global__ void bench_float(float *input_x, float *input_y, float *output_x, float *output_y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float r = sqrtf(input_x[i]*input_x[i] + input_y[i]*input_y[i]);
        float scale = 1.0001f; // arbitrary multiply
        output_x[i] = input_x[i] * scale;
        output_y[i] = input_y[i] * scale;
    }
}

int main() {
    int N = 10000000; // 10M vectors
    float *ix, *iy, *ox, *oy;
    cudaMalloc(&ix, N*sizeof(float)); cudaMalloc(&iy, N*sizeof(float));
    cudaMalloc(&ox, N*sizeof(float)); cudaMalloc(&oy, N*sizeof(float));
    
    // Init random input
    float *hix = (float*)malloc(N*sizeof(float));
    float *hiy = (float*)malloc(N*sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) {
        hix[i] = (float)rand()/RAND_MAX * 100.0f;
        hiy[i] = (float)rand()/RAND_MAX * 100.0f;
    }
    cudaMemcpy(ix, hix, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(iy, hiy, N*sizeof(float), cudaMemcpyHostToDevice);
    
    int blocks = (N + 255) / 256;
    
    // Warmup
    bench_ct_snap<<<blocks, 256>>>(ix, iy, ox, oy, N);
    cudaDeviceSynchronize();
    
    // CT snap benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int rep = 0; rep < 100; rep++)
        bench_ct_snap<<<blocks, 256>>>(ix, iy, ox, oy, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float ct_ms;
    cudaEventElapsedTime(&ct_ms, start, stop);
    
    // Float baseline
    bench_float<<<blocks, 256>>>(ix, iy, ox, oy, N);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int rep = 0; rep < 100; rep++)
        bench_float<<<blocks, 256>>>(ix, iy, ox, oy, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float fl_ms;
    cudaEventElapsedTime(&fl_ms, start, stop);
    
    printf("=== CT Snap vs Float Multiply Benchmark ===\n");
    printf("Vectors: %d x 100 reps = %d operations\n", N, N*100);
    printf("CT snap:    %.2f ms (%.0f Mvec/s)\n", ct_ms, N*100.0f/ct_ms/1000.0f);
    printf("Float mul:  %.2f ms (%.0f Mvec/s)\n", fl_ms, N*100.0f/fl_ms/1000.0f);
    printf("Ratio:      %.2fx (CT snap / Float mul)\n", ct_ms/fl_ms);
    
    // Verify snap quality
    float *hox = (float*)malloc(N*sizeof(float));
    float *hoy = (float*)malloc(N*sizeof(float));
    cudaMemcpy(hox, ox, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hoy, oy, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_drift = 0, avg_drift = 0;
    for (int i = 0; i < N; i++) {
        float dx = hix[i]-hox[i], dy = hiy[i]-hoy[i];
        float drift = sqrtf(dx*dx+dy*dy);
        if (drift > max_drift) max_drift = drift;
        avg_drift += drift;
    }
    printf("\nSnap quality: avg_drift=%.4f max_drift=%.4f\n", avg_drift/N, max_drift);
    
    cudaFree(ix); cudaFree(iy); cudaFree(ox); cudaFree(oy);
    free(hix); free(hiy); free(hox); free(hoy);
    return 0;
}
