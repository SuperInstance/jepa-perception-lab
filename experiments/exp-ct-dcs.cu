#include <stdio.h>
#include <math.h>
#include <curand_kernel.h>

#define N_AGENTS 512
#define N_FOOD 200
#define WORLD 1024
#define STEPS 5000
#define EPISODES 10

// CT snap to nearest Pythagorean ratio (simplified)
__device__ void ct_snap2(float x, float y, float *sx, float *sy) {
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

__global__ void run_dcs(float *results, int use_ct_snap, int add_noise, unsigned long long seed) {
    int aid = blockIdx.x*blockDim.x+threadIdx.x;
    if(aid >= N_AGENTS) return;
    
    curandState rng;
    curand_init(seed+aid, 0, 0, &rng);
    
    // Agent state
    float ax = curand_uniform(&rng)*WORLD;
    float ay = curand_uniform(&rng)*WORLD;
    float collected = 0;
    int guild = aid % 4;
    
    // Shared food locations (simplified DCS - each agent knows K=1 nearest food per guild)
    // In real DCS this would be a shared buffer, here we simulate
    float known_food_x[4], known_food_y[4]; // 1 per guild
    for(int g=0;g<4;g++) { known_food_x[g]=-1; known_food_y[g]=-1; }
    
    for(int step=0; step<STEPS; step++) {
        // Move toward known food or wander
        float tx = -1, ty = -1;
        
        // Check own guild's known food first (DCS: prefer own guild)
        if(known_food_x[guild] >= 0) {
            tx = known_food_x[guild]; ty = known_food_y[guild];
        }
        
        if(tx < 0) {
            // Wander
            ax += (curand_uniform(&rng)-0.5f)*6.0f;
            ay += (curand_uniform(&rng)-0.5f)*6.0f;
        } else {
            // Move toward target
            float dx=tx-ax, dy=ty-ay;
            float d=sqrtf(dx*dx+dy*dy);
            if(d > 3.0f) { ax+=dx/d*3.0f; ay+=dy/d*3.0f; }
        }
        
        // Wrap
        if(ax<0) ax+=WORLD; if(ax>=WORLD) ax-=WORLD;
        if(ay<0) ay+=WORLD; if(ay>=WORLD) ay-=WORLD;
        
        // Try to collect food (food positions are deterministic per step)
        for(int f=0; f<N_FOOD/4; f++) { // check local food
            unsigned long long fseed = step*1000ULL + f*4 + guild;
            float fx = ((float)((fseed*1103515245+12345)%WORLD))/WORLD * WORLD;
            float fy = ((float)(((fseed>>16)*1103515245+12345)%WORLD))/WORLD * WORLD;
            
            float dx=fx-ax, dy=fy-ay;
            float d = sqrtf(dx*dx+dy*dy);
            if(d < 15.0f) { // grab range
                collected += 1.0f;
                break;
            }
            
            // DCS: store nearest food for guild broadcast
            if(d < 200.0f) {
                if(use_ct_snap) {
                    ct_snap2(fx, fy, &known_food_x[guild], &known_food_y[guild]);
                } else {
                    if(add_noise) {
                        known_food_x[guild] = fx + (curand_uniform(&rng)-0.5f)*0.1f*WORLD;
                        known_food_y[guild] = fy + (curand_uniform(&rng)-0.5f)*0.1f*WORLD;
                    } else {
                        known_food_x[guild] = fx;
                        known_food_y[guild] = fy;
                    }
                }
            }
        }
    }
    
    results[aid] = collected;
}

int main() {
    printf("=== CT-Snap DCS vs Noisy DCS vs Raw DCS ===\n");
    printf("Agents: %d | Food: %d | World: %d | Steps: %d\n", N_AGENTS, N_FOOD, WORLD, STEPS);
    printf("JC1 Law 42: 5%% noise = -52%% DCS performance\n");
    printf("Hypothesis: CT snap (zero noise) > raw DCS > noisy DCS\n\n");
    
    float *d_res;
    cudaMalloc(&d_res, N_AGENTS*sizeof(float));
    
    float h_res[N_AGENTS];
    
    const char* modes[] = {"Raw DCS (no noise)", "Noisy DCS (5% noise)", "CT-Snap DCS (zero noise)"};
    int configs[][2] = {{0,0},{0,1},{1,0}}; // {use_ct_snap, add_noise}
    
    for(int c=0; c<3; c++) {
        float total = 0;
        for(int ep=0; ep<EPISODES; ep++) {
            run_dcs<<<2,256>>>(d_res, configs[c][0], configs[c][1], ep*10000+c);
            cudaDeviceSynchronize();
            cudaMemcpy(h_res, d_res, N_AGENTS*sizeof(float), cudaMemcpyDeviceToHost);
            for(int i=0;i<N_AGENTS;i++) total+=h_res[i];
        }
        float avg = total / (N_AGENTS * EPISODES);
        printf("%-25s avg_collection=%.3f\n", modes[c], avg);
    }
    
    cudaFree(d_res);
    printf("\nIf CT-snap DCS beats raw DCS, CT snap is the zero-noise protocol DCS needs.\n");
    return 0;
}
