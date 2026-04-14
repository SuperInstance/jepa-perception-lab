// Experiment 1: JEPA Latent Dimension Sweep (v2 — host-initialized world)
// Simplified but working version for RTX 4050

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define NUM_AGENTS 256
#define NUM_ROOMS 16
#define ROOM_FEATURES 8
#define TICKS 1000
#define EPISODES 50
#define LR 0.01f
#define HIDDEN 16  // small hidden layer

// Room features: [resource, danger, exits, density, stigmergy, temp, light, special]
typedef struct {
    float f[ROOM_FEATURES];
} Room;

// World: 4x4 grid of rooms
typedef struct {
    Room rooms[NUM_ROOMS];
    int adj[NUM_ROOMS][4]; // up, right, down, left
} World;

// Per-agent state (position, score, alive)
typedef struct {
    int room;
    float resource;
    int alive;
    int steps;
} Agent;

// Model weights (flat arrays, one set per agent)
// Encoder: input(8) -> hidden(16) -> latent(dim)
// Predictor: latent(dim) -> output(8)
typedef struct {
    int latent_dim;
    float enc_w1[HIDDEN * ROOM_FEATURES];  // 16x8 = 128
    float enc_b1[HIDDEN];                   // 16
    float enc_w2[64 * HIDDEN];              // max latent=64, 64x16 = 1024
    float enc_b2[64];                       // max 64
    float pred_w[ROOM_FEATURES * 64];       // 8x64 max = 512
    float pred_b[ROOM_FEATURES];            // 8
    float latent[64];                       // max 64
    float hidden[HIDDEN];                   // intermediate
} Model;

// Host: init world
void init_world_host(World *w) {
    srand(42);
    for (int r = 0; r < NUM_ROOMS; r++) {
        w->rooms[r].f[0] = (float)rand()/RAND_MAX;          // resource
        w->rooms[r].f[1] = (float)rand()/RAND_MAX * 0.3f;  // danger (low)
        w->rooms[r].f[4] = (float)rand()/RAND_MAX;          // stigmergy
        w->rooms[r].f[5] = 0.5f + (float)rand()/RAND_MAX * 0.2f;
        w->rooms[r].f[6] = (float)rand()/RAND_MAX;
        
        int row = r/4, col = r%4;
        w->adj[r][0] = row > 0 ? r-4 : -1;
        w->adj[r][1] = col < 3 ? r+1 : -1;
        w->adj[r][2] = row < 3 ? r+4 : -1;
        w->adj[r][3] = col > 0 ? r-1 : -1;
        
        w->rooms[r].f[2] = 0;
        for (int d = 0; d < 4; d++)
            if (w->adj[r][d] >= 0) w->rooms[r].f[2] += 1.0f;
    }
}

// CUDA kernel: run one episode for all agents
__global__ void run_episode(Model *models, World *world, Agent *agents,
                            float *losses, int latent_dim, unsigned long long seed) {
    int aid = blockIdx.x * blockDim.x + threadIdx.x;
    if (aid >= NUM_AGENTS) return;
    
    curandState rng;
    curand_init(seed + aid, 0, 0, &rng);
    
    Model *m = &models[aid];
    Agent *a = &agents[aid];
    
    // Start in random room
    a->room = (int)(curand_uniform(&rng) * NUM_ROOMS);
    if (a->room >= NUM_ROOMS) a->room = NUM_ROOMS - 1;
    a->resource = 0;
    a->alive = 1;
    a->steps = 0;
    float total_loss = 0;
    
    for (int t = 0; t < TICKS && a->alive; t++) {
        Room cur = world->rooms[a->room];
        
        // Collect resource
        a->resource += cur.f[0] * 0.01f;
        
        // Danger check
        if (cur.f[1] > 0.7f && curand_uniform(&rng) < cur.f[1]) {
            a->alive = 0;
            break;
        }
        
        // Encode current room
        for (int h = 0; h < HIDDEN; h++) {
            float sum = m->enc_b1[h];
            for (int i = 0; i < ROOM_FEATURES; i++)
                sum += m->enc_w1[h * ROOM_FEATURES + i] * cur.f[i];
            m->hidden[h] = fmaxf(0.0f, sum);
        }
        for (int l = 0; l < latent_dim; l++) {
            float sum = m->enc_b2[l];
            for (int h = 0; h < HIDDEN; h++)
                sum += m->enc_w2[l * HIDDEN + h] * m->hidden[h];
            m->latent[l] = sum;
        }
        
        // Choose best adjacent room using predictor
        int best = a->room;
        float best_score = -1e9f;
        
        for (int d = 0; d < 4; d++) {
            int adj = world->adj[a->room][d];
            if (adj < 0) continue;
            
            Room next = world->rooms[adj];
            // Quick score: predict resource - 2*danger from latent
            float score = 0;
            for (int l = 0; l < latent_dim; l++)
                score += m->pred_w[0 * latent_dim + l] * m->latent[l]  // resource
                       - 2.0f * m->pred_w[1 * latent_dim + l] * m->latent[l]; // danger
            if (score > best_score) { best_score = score; best = adj; }
        }
        
        // Move (90% greedy, 10% explore)
        if (curand_uniform(&rng) < 0.9f) {
            a->room = best;
        } else {
            int valid[4], nv = 0;
            for (int d = 0; d < 4; d++)
                if (world->adj[a->room][d] >= 0) valid[nv++] = world->adj[a->room][d];
            if (nv > 0) a->room = valid[(int)(curand_uniform(&rng) * nv) % nv];
        }
        
        // Train: predict next room features from current latent
        Room actual = world->rooms[a->room];
        for (int i = 0; i < ROOM_FEATURES; i++) {
            float pred = m->pred_b[i];
            for (int l = 0; l < latent_dim; l++)
                pred += m->pred_w[i * latent_dim + l] * m->latent[l];
            float err = pred - actual.f[i];
            total_loss += err * err / ROOM_FEATURES;
            
            // Gradient update on predictor
            float grad = 2.0f * err / ROOM_FEATURES * LR;
            for (int l = 0; l < latent_dim; l++)
                m->pred_w[i * latent_dim + l] -= grad * m->latent[l];
            m->pred_b[i] -= grad;
        }
        a->steps++;
    }
    
    losses[aid] = a->steps > 0 ? total_loss / a->steps : 0;
}

void run_sweep(int latent_dim) {
    // Allocate
    Model *d_models;
    Agent *d_agents;
    World *d_world;
    float *d_losses;
    
    cudaMalloc(&d_models, NUM_AGENTS * sizeof(Model));
    cudaMalloc(&d_agents, NUM_AGENTS * sizeof(Agent));
    cudaMalloc(&d_world, sizeof(World));
    cudaMalloc(&d_losses, NUM_AGENTS * sizeof(float));
    
    // Init models on host (Law 150: init doesn't matter)
    Model *h_models = (Model*)calloc(NUM_AGENTS, sizeof(Model));
    srand(42);
    for (int a = 0; a < NUM_AGENTS; a++) {
        h_models[a].latent_dim = latent_dim;
        for (int i = 0; i < HIDDEN * ROOM_FEATURES; i++)
            h_models[a].enc_w1[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        for (int i = 0; i < HIDDEN; i++)
            h_models[a].enc_b1[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        for (int i = 0; i < 64 * HIDDEN; i++)
            h_models[a].enc_w2[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        for (int i = 0; i < 64; i++)
            h_models[a].enc_b2[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        for (int i = 0; i < ROOM_FEATURES * 64; i++)
            h_models[a].pred_w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.3f;
        for (int i = 0; i < ROOM_FEATURES; i++)
            h_models[a].pred_b[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    }
    cudaMemcpy(d_models, h_models, NUM_AGENTS * sizeof(Model), cudaMemcpyHostToDevice);
    free(h_models);
    
    // Init world
    World h_world;
    init_world_host(&h_world);
    cudaMemcpy(d_world, &h_world, sizeof(World), cudaMemcpyHostToDevice);
    
    // Run episodes
    float best_loss = 1e9, best_surv = 0, best_res = 0;
    float final_loss = 0, final_surv = 0, final_res = 0;
    
    for (int ep = 0; ep < EPISODES; ep++) {
        run_episode<<<1, NUM_AGENTS>>>(d_models, d_world, d_agents, d_losses, 
                                        latent_dim, ep * 1000ULL + 42);
        cudaDeviceSynchronize();
        
        float h_losses[NUM_AGENTS];
        Agent h_agents[NUM_AGENTS];
        cudaMemcpy(h_losses, d_losses, NUM_AGENTS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_agents, d_agents, NUM_AGENTS * sizeof(Agent), cudaMemcpyDeviceToHost);
        
        float ep_loss = 0, ep_surv = 0, ep_res = 0;
        for (int a = 0; a < NUM_AGENTS; a++) {
            ep_loss += h_losses[a];
            ep_surv += h_agents[a].alive;
            ep_res += h_agents[a].resource;
        }
        ep_loss /= NUM_AGENTS;
        ep_surv /= NUM_AGENTS;
        ep_res /= NUM_AGENTS;
        
        if (ep_loss < best_loss) {
            best_loss = ep_loss;
            best_surv = ep_surv;
            best_res = ep_res;
        }
        
        if (ep >= EPISODES - 10) {
            final_loss += ep_loss;
            final_surv += ep_surv;
            final_res += ep_res;
        }
    }
    final_loss /= 10; final_surv /= 10; final_res /= 10;
    
    printf("  dim=%2d | final_loss=%.4f | best_loss=%.4f | survival=%.1f%% | resource=%.3f\n",
           latent_dim, final_loss, best_loss, final_surv * 100, final_res);
    
    cudaFree(d_models); cudaFree(d_agents); cudaFree(d_world); cudaFree(d_losses);
}

int main() {
    printf("=== JEPA Latent Dimension Sweep (v2) ===\n");
    printf("GPU: RTX 4050 (6.4 GB) | Agents: %d | Rooms: %d | Ticks: %d | Episodes: %d\n",
           NUM_AGENTS, NUM_ROOMS, TICKS, EPISODES);
    printf("JC1 Baseline: 4-dim, 16 params, survival ~51%%\n");
    printf("==========================================\n\n");
    
    int dims[] = {4, 8, 16, 32, 64};
    for (int i = 0; i < 5; i++) {
        printf("Running dim=%d...\n", dims[i]);
        run_sweep(dims[i]);
    }
    
    printf("\nJC1 hypothesis: 8-16 dims sweet spot, 32+ overfits.\n");
    return 0;
}
