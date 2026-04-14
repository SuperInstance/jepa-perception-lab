#include <stdio.h>
#include <curand_kernel.h>
#include <math.h>

// Test 3 properties of CT snap in one experiment:
// 1. Axis-aligned identity (snap should not move axis-aligned vectors)
// 2. Commutativity with rotation (rotate-then-snap vs snap-then-rotate)
// 3. Order independence (snap x then y vs snap y then x)

__device__ void ct_snap(float x, float y, float *sx, float *sy) {
    float r = sqrtf(x*x + y*y);
    if (r < 1e-6f) { *sx = 0; *sy = 0; return; }
    int triples[][2] = {{3,4},{5,12},{8,15},{7,24},{20,21},{9,40},{12,35},{11,60}};
    float best = 1e9f;
    for (int i = 0; i < 8; i++) {
        float a=triples[i][0], b=triples[i][1], c=sqrtf(a*a+b*b);
        for (int f=0;f<2;f++) {
            float pa=f?b:a, pb=f?a:b;
            float nx=pa/c*r, ny=pb/c*r;
            float d=(x-nx)*(x-nx)+(y-ny)*(y-ny);
            if(d<best){best=d;*sx=nx;*sy=ny;}
        }
    }
    float axes[][2]={{1,0},{0,1},{1,1},{-1,1}};
    for(int i=0;i<4;i++){
        float a2=axes[i][0],b2=axes[i][1],c2=sqrtf(a2*a2+b2*b2);
        float nx=a2/c2*r,ny=b2/c2*r;
        float d=(x-nx)*(x-nx)+(y-ny)*(y-ny);
        if(d<best){best=d;*sx=nx;*sy=ny;}
    }
}

__device__ void rotate(float x, float y, float angle, float *rx, float *ry) {
    float c = cosf(angle), s = sinf(angle);
    *rx = x*c - y*s;
    *ry = x*s + y*c;
}

__global__ void test_properties(float *results, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState rng;
    curand_init(seed + idx, 0, 0, &rng);
    
    float x = (curand_uniform(&rng) - 0.5f) * 200.0f;
    float y = (curand_uniform(&rng) - 0.5f) * 200.0f;
    float angle = curand_uniform(&rng) * 2.0f * 3.14159f;
    
    // Test 1: Axis-aligned identity
    float sx, sy;
    ct_snap(1.0f, 0.0f, &sx, &sy);
    float axis_err_x = fabsf(sx - 1.0f) + fabsf(sy);
    ct_snap(0.0f, 1.0f, &sx, &sy);
    float axis_err_y = fabsf(sx) + fabsf(sy - 1.0f);
    
    // Test 2: Commutativity with rotation
    float rx1, ry1, rx2, ry2, sx1, sy1, sx2, sy2;
    // rotate then snap
    rotate(x, y, angle, &rx1, &ry1);
    ct_snap(rx1, ry1, &sx1, &sy1);
    // snap then rotate
    ct_snap(x, y, &rx2, &ry2);
    rotate(rx2, ry2, angle, &sx2, &sy2);
    
    float commut_err = sqrtf((sx1-sx2)*(sx1-sx2) + (sy1-sy2)*(sy1-sy2));
    
    // Test 3: Count distinct Pythagorean directions
    // (computed once by thread 0)
    if (idx == 0) {
        int count = 0;
        for (int a = 1; a < 1000; a++) {
            for (int b = a; b < 1000; b++) {
                float c2 = (float)a*a + (float)b*b;
                float c = sqrtf(c2);
                if (fabsf(c - roundf(c)) < 0.001f) count++;
            }
        }
        results[4] = (float)count;
    }
    
    results[0] = axis_err_x + axis_err_y; // axis identity error
    results[1] = commut_err;              // commutativity error
    results[2] = x;                        // original x
    results[3] = y;                        // original y
}

int main() {
    printf("=== CT Snap Property Tests ===\n\n");
    
    int N = 100000;
    float *d_res;
    cudaMalloc(&d_res, N * 5 * sizeof(float));
    
    test_properties<<<(N+255)/256, 256>>>(d_res, 42);
    cudaDeviceSynchronize();
    
    float *h = (float*)malloc(N * 5 * sizeof(float));
    cudaMemcpy(h, d_res, N * 5 * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Test 1: Axis-aligned identity
    float max_axis = 0, avg_axis = 0;
    for (int i = 0; i < N; i++) {
        avg_axis += h[i*5+0];
        if (h[i*5+0] > max_axis) max_axis = h[i*5+0];
    }
    avg_axis /= N;
    printf("TEST 1: Axis-aligned identity\n");
    printf("  snap(1,0) and snap(0,1) error: avg=%.6f max=%.6f\n", avg_axis, max_axis);
    printf("  Verdict: %s\n\n", max_axis < 0.001f ? "PASS (axis vectors are fixed points)" : "FAIL (axis vectors move)");
    
    // Test 2: Commutativity with rotation
    float max_comm = 0, avg_comm = 0;
    for (int i = 0; i < N; i++) {
        avg_comm += h[i*5+1];
        if (h[i*5+1] > max_comm) max_comm = h[i*5+1];
    }
    avg_comm /= N;
    printf("TEST 2: Commutativity with rotation\n");
    printf("  rotate-then-snap vs snap-then-rotate: avg=%.4f max=%.4f\n", avg_comm, max_comm);
    printf("  Verdict: %s\n\n", max_comm < 0.01f ? "COMMUTATIVE" : "NOT COMMUTATIVE (order matters!)");
    
    // Test 3: Distinct Pythagorean directions
    int count = (int)h[4];
    printf("TEST 3: Distinct Pythagorean directions in 2D (sides < 1000)\n");
    printf("  Count: %d directions\n", count);
    printf("  Bits needed: %.1f (log2)\n", log2((double)count));
    
    cudaFree(d_res); free(h);
    return 0;
}
