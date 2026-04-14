#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

#define N 1000000
#define MAX_SNAPS 1000

__device__ void ct_snap_simple(float x, float y, float *sx, float *sy) {
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

__global__ void test_idempotency(float *max_drift, float *max_drift_1k, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    curandState rng;
    curand_init(seed + idx, 0, 0, &rng);
    
    // Random starting point
    float x = (curand_uniform(&rng) - 0.5f) * 200.0f;
    float y = (curand_uniform(&rng) - 0.5f) * 200.0f;
    
    // First snap
    float sx, sy;
    ct_snap_simple(x, y, &sx, &sy);
    
    // Record first snap result
    float first_sx = sx, first_sy = sy;
    
    // Snap 999 more times
    float worst_drift = 0;
    for (int i = 0; i < MAX_SNAPS - 1; i++) {
        float nsx, nsy;
        ct_snap_simple(sx, sy, &nsx, &nsy);
        float drift = sqrtf((nsx-sx)*(nsx-sx) + (nsy-sy)*(nsy-sy));
        if (drift > worst_drift) worst_drift = drift;
        sx = nsx; sy = nsy;
    }
    
    // Total drift from first snap to last
    float total_drift = sqrtf((sx-first_sx)*(sx-first_sx) + (sy-first_sy)*(sy-first_sy));
    
    max_drift[idx] = total_drift;
    max_drift_1k[idx] = worst_drift;
}

int main() {
    printf("=== CT Snap Idempotency Stress Test ===\n");
    printf("1,000,000 random vectors, each snapped 1000 times\n");
    printf("Question: does snap(snap(...(x))) == snap(x)?\n\n");
    
    float *d_drift, *d_drift_1k;
    cudaMalloc(&d_drift, N*sizeof(float));
    cudaMalloc(&d_drift_1k, N*sizeof(float));
    
    test_idempotency<<<(N+255)/256, 256>>>(d_drift, d_drift_1k, 42);
    cudaDeviceSynchronize();
    
    float *h_drift = (float*)malloc(N*sizeof(float));
    float *h_drift_1k = (float*)malloc(N*sizeof(float));
    cudaMemcpy(h_drift, d_drift, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_drift_1k, d_drift_1k, N*sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_total = 0, avg_total = 0;
    float max_step = 0, avg_step = 0;
    int perfect = 0;
    
    for (int i = 0; i < N; i++) {
        if (h_drift[i] > max_total) max_total = h_drift[i];
        avg_total += h_drift[i];
        if (h_drift[i] < 1e-6f) perfect++;
        
        if (h_drift_1k[i] > max_step) max_step = h_drift_1k[i];
        avg_step += h_drift_1k[i];
    }
    
    printf("After 1000 snaps:\n");
    printf("  Total drift from first snap: avg=%.8f max=%.8f\n", avg_total/N, max_total);
    printf("  Max single-step drift:       avg=%.8f max=%.8f\n", avg_step/N, max_step);
    printf("  Perfectly idempotent:        %d / %d (%.1f%%)\n", perfect, N, 100.0*perfect/N);
    
    if (max_total < 1e-6f) {
        printf("\n✅ SNAP IS IDEMPOTENT — zero drift after 1000 applications.\n");
        printf("This means CT snap is a FIXED POINT operation.\n");
    } else {
        printf("\n⚠️  DRIFT DETECTED — snap is NOT perfectly idempotent.\n");
        printf("Max accumulated drift after 1000 snaps: %.8f\n", max_total);
        printf("This means the snap attractor has finite precision.\n");
    }
    
    cudaFree(d_drift); cudaFree(d_drift_1k);
    free(h_drift); free(h_drift_1k);
    return 0;
}
