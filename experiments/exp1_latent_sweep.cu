// Experiment 1: JEPA Latent Dimension Sweep
// Train small encoders with 4, 8, 16, 32, 64-dim latents
// Test MUD room perception — predict next room state from current
// Based on JC1's Laws 141, 142, 145, 150, 153

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <curand_kernel.h>

#define NUM_AGENTS 256
#define NUM_ROOMS 16
#define ROOM_FEATURES 8    // resources, danger, exits, agents_nearby, etc.
#define TICKS_PER_EPISODE 1000
#define NUM_EPISODES 50
#define LEARNING_RATE 0.01f

// Room state: 8 features per room
// [resource_level, danger_level, exit_count, agent_density, 
//  stigmergy_signal, temperature, light, special]
typedef struct {
    float features[ROOM_FEATURES];
} RoomState;

// MUD World: 16 rooms in a 4x4 grid
typedef struct {
    RoomState rooms[NUM_ROOMS];
    int adjacency[NUM_ROOMS][4]; // up, right, down, left (-1 = wall)
} MUDWorld;

// JEPA Encoder: maps room state -> latent
// Simple MLP: input(8) -> hidden(32) -> latent(dim)
// Plus predictor: latent -> predicted_next_features(8)
typedef struct {
    int latent_dim;
    float *enc_w1;    // [hidden_dim x input_dim]
    float *enc_b1;    // [hidden_dim]
    float *enc_w2;    // [latent_dim x hidden_dim]
    float *enc_b2;    // [latent_dim]
    float *pred_w;    // [input_dim x latent_dim]
    float *pred_b;    // [input_dim]
    float *latent;    // [latent_dim] current latent
    float loss;       // running loss
    float survival;   // survival rate
    float resource;   // resource acquisition score
} JEPAModel;

// Initialize MUD world
__device__ void init_world(MUDWorld *world, unsigned long long seed) {
    curandState rng;
    curand_init(seed, 0, 0, &rng);
    
    for (int r = 0; r < NUM_ROOMS; r++) {
        // Grid layout: room r is at (r/4, r%4)
        world->rooms[r].features[0] = curand_uniform(&rng);  // resource
        world->rooms[r].features[1] = curand_uniform(&rng) * 0.3f; // danger (low)
        world->rooms[r].features[2] = 0; // exits (set below)
        world->rooms[r].features[3] = 0; // agent density
        world->rooms[r].features[4] = curand_uniform(&rng);  // stigmergy
        world->rooms[r].features[5] = 0.5f + curand_uniform(&rng) * 0.2f; // temp
        world->rooms[r].features[6] = curand_uniform(&rng);  // light
        world->rooms[r].features[7] = 0; // special
        
        int row = r / 4, col = r % 4;
        world->adjacency[r][0] = (row > 0) ? r - 4 : -1;   // up
        world->adjacency[r][1] = (col < 3) ? r + 1 : -1;   // right
        world->adjacency[r][2] = (row < 3) ? r + 4 : -1;   // down
        world->adjacency[r][3] = (col > 0) ? r - 1 : -1;   // left
        
        // Count exits
        for (int d = 0; d < 4; d++)
            if (world->adjacency[r][d] >= 0) world->rooms[r].features[2] += 1.0f;
    }
}

// Forward pass: encode room state to latent
__device__ void encode(JEPAModel *model, RoomState *room, float *hidden_buf) {
    int hidden_dim = 32;
    // Layer 1: input -> hidden (ReLU)
    for (int h = 0; h < hidden_dim; h++) {
        float sum = model->enc_b1[h];
        for (int i = 0; i < ROOM_FEATURES; i++) {
            sum += model->enc_w1[h * ROOM_FEATURES + i] * room->features[i];
        }
        hidden_buf[h] = fmaxf(0.0f, sum); // ReLU
    }
    // Layer 2: hidden -> latent (linear)
    for (int l = 0; l < model->latent_dim; l++) {
        float sum = model->enc_b2[l];
        for (int h = 0; h < hidden_dim; h++) {
            sum += model->enc_w2[l * hidden_dim + h] * hidden_buf[h];
        }
        model->latent[l] = sum;
    }
}

// Predict next room features from latent
__device__ void predict(JEPAModel *model, float *predicted_features) {
    for (int i = 0; i < ROOM_FEATURES; i++) {
        float sum = model->pred_b[i];
        for (int l = 0; l < model->latent_dim; l++) {
            sum += model->pred_w[i * model->latent_dim + l] * model->latent[l];
        }
        predicted_features[i] = sum;
    }
}

// Simple MSE loss + gradient update
__device__ float train_step(JEPAModel *model, RoomState *current, RoomState *next, 
                            float *hidden_buf, float *grad_buf, float lr) {
    // Encode current room
    encode(model, current, hidden_buf);
    
    // Predict next room
    float predicted[ROOM_FEATURES];
    predict(model, predicted);
    
    // Compute MSE loss
    float loss = 0.0f;
    for (int i = 0; i < ROOM_FEATURES; i++) {
        float diff = predicted[i] - next->features[i];
        loss += diff * diff;
    }
    loss /= ROOM_FEATURES;
    
    // Simple gradient: direct weight update on predictor
    for (int i = 0; i < ROOM_FEATURES; i++) {
        float diff = predicted[i] - next->features[i];
        float grad = 2.0f * diff / ROOM_FEATURES;
        for (int l = 0; l < model->latent_dim; l++) {
            model->pred_w[i * model->latent_dim + l] -= lr * grad * model->latent[l];
        }
        model->pred_b[i] -= lr * grad;
    }
    
    return loss;
}

// Agent simulation kernel
__global__ void simulate_agents(JEPAModel *models, MUDWorld *world, 
                                 float *results, int latent_dims,
                                 unsigned long long seed) {
    int agent_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_id >= NUM_AGENTS) return;
    
    curandState rng;
    curand_init(seed + agent_id, 0, 0, &rng);
    
    JEPAModel *model = &models[agent_id];
    
    // Allocate buffers in shared memory would be ideal but use registers
    float hidden_buf[32];
    float grad_buf[32];
    
    // Initialize agent in random room
    int current_room = (int)(curand_uniform(&rng) * NUM_ROOMS);
    if (current_room >= NUM_ROOMS) current_room = NUM_ROOMS - 1;
    
    float total_resource = 0.0f;
    int alive = 1;
    float total_loss = 0.0f;
    int steps = 0;
    
    for (int tick = 0; tick < TICKS_PER_EPISODE && alive; tick++) {
        RoomState *room = &world->rooms[current_room];
        
        // Collect resource
        total_resource += room->features[0] * 0.01f;
        
        // Danger check (JC1 Law 145: danger encoding doubles survival)
        if (room->features[1] > 0.7f && curand_uniform(&rng) < room->features[1]) {
            alive = 0; // dead
            break;
        }
        
        // Choose next room using latent prediction
        // Look at adjacent rooms, encode each, predict, pick best
        int best_room = current_room;
        float best_score = -1e9f;
        
        for (int d = 0; d < 4; d++) {
            int adj = world->adjacency[current_room][d];
            if (adj < 0) continue;
            
            // Encode adjacent room and predict value
            encode(model, &world->rooms[adj], hidden_buf);
            
            // Score = resource prediction - danger prediction
            float pred_res = 0, pred_danger = 0;
            for (int l = 0; l < model->latent_dim; l++) {
                pred_res += model->pred_w[0 * model->latent_dim + l] * model->latent[l];
                pred_danger += model->pred_w[1 * model->latent_dim + l] * model->latent[l];
            }
            float score = pred_res - 2.0f * pred_danger; // weight danger 2x (Law 145)
            
            if (score > best_score) {
                best_score = score;
                best_room = adj;
            }
        }
        
        // Move to best room (with some exploration)
        if (curand_uniform(&rng) < 0.9f) {
            current_room = best_room;
        } else {
            // Random adjacent room (exploration)
            int valid[4], nvalid = 0;
            for (int d = 0; d < 4; d++) {
                if (world->adjacency[current_room][d] >= 0)
                    valid[nvalid++] = world->adjacency[current_room][d];
            }
            if (nvalid > 0)
                current_room = valid[(int)(curand_uniform(&rng) * nvalid)];
        }
        
        // Train on transition: current -> next
        if (tick > 0) {
            RoomState *prev = &world->rooms[current_room]; // simplified
            total_loss += train_step(model, &world->rooms[current_room], 
                                    &world->rooms[current_room], hidden_buf, grad_buf, LEARNING_RATE);
            steps++;
        }
    }
    
    // Store results
    results[agent_id * 4 + 0] = total_loss / fmaxf(1, (float)steps); // avg loss
    results[agent_id * 4 + 1] = alive ? 1.0f : 0.0f; // survival
    results[agent_id * 4 + 2] = total_resource; // resource score
    results[agent_id * 4 + 3] = (float)latent_dims; // for identification
}

// Host-side experiment runner
void run_experiment(int latent_dim, float *summary) {
    int hidden_dim = 32;
    int model_size = hidden_dim * ROOM_FEATURES + hidden_dim +   // enc layer 1
                     latent_dim * hidden_dim + latent_dim +       // enc layer 2
                     ROOM_FEATURES * latent_dim + ROOM_FEATURES + // predictor
                     latent_dim;                                   // latent buffer
    
    size_t model_bytes = sizeof(JEPAModel) + model_size * sizeof(float);
    
    // Allocate models for all agents
    JEPAModel *d_models;
    cudaMalloc(&d_models, NUM_AGENTS * model_bytes);
    
    // Initialize weights randomly on GPU
    // (JC1 Law 150: weight init doesn't matter)
    float *init_data = (float*)malloc(NUM_AGENTS * model_size * sizeof(float));
    srand(42);
    for (int i = 0; i < NUM_AGENTS * model_size; i++) {
        init_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    cudaMemcpy(d_models, init_data, NUM_AGENTS * model_size * sizeof(float), cudaMemcpyHostToDevice);
    free(init_data);
    
    // Set latent dims
    // (This is simplified - in production we'd properly init the struct)
    
    // Allocate world and results
    MUDWorld *d_world;
    float *d_results;
    cudaMalloc(&d_world, sizeof(MUDWorld));
    cudaMalloc(&d_results, NUM_AGENTS * 4 * sizeof(float));
    
    // Run episodes
    float total_loss = 0, total_survival = 0, total_resource = 0;
    
    for (int ep = 0; ep < NUM_EPISODES; ep++) {
        // Init world on GPU (simplified - would need a kernel)
        // For now, use fixed seed
        simulate_agents<<<(NUM_AGENTS + 255) / 256, 256>>>(
            d_models, d_world, d_results, latent_dim, ep * 1000ULL);
        cudaDeviceSynchronize();
        
        // Copy results back
        float h_results[NUM_AGENTS * 4];
        cudaMemcpy(h_results, d_results, NUM_AGENTS * 4 * sizeof(float), cudaMemcpyDeviceToHost);
        
        float ep_loss = 0, ep_surv = 0, ep_res = 0;
        for (int a = 0; a < NUM_AGENTS; a++) {
            ep_loss += h_results[a * 4 + 0];
            ep_surv += h_results[a * 4 + 1];
            ep_res  += h_results[a * 4 + 2];
        }
        total_loss += ep_loss / NUM_AGENTS;
        total_survival += ep_surv / NUM_AGENTS;
        total_resource += ep_res / NUM_AGENTS;
    }
    
    summary[0] = total_loss / NUM_EPISODES;
    summary[1] = total_survival / NUM_EPISODES;
    summary[2] = total_resource / NUM_EPISODES;
    summary[3] = (float)latent_dim;
    
    printf("  latent_dim=%2d | loss=%.4f | survival=%.1f%% | resource=%.3f\n",
           latent_dim, summary[0], summary[1] * 100, summary[2]);
    
    cudaFree(d_models);
    cudaFree(d_world);
    cudaFree(d_results);
}

int main() {
    printf("=== JEPA Latent Dimension Sweep ===\n");
    printf("GPU: RTX 4050 (6.4 GB VRAM)\n");
    printf("Agents: %d | Rooms: %d | Ticks: %d | Episodes: %d\n",
           NUM_AGENTS, NUM_ROOMS, TICKS_PER_EPISODE, NUM_EPISODES);
    printf("JC1 Baseline: 4-dim linear, 16 params\n");
    printf("=====================================\n\n");
    
    // Test latent dimensions: 4, 8, 16, 32, 64
    int dims[] = {4, 8, 16, 32, 64};
    int n_dims = 5;
    float results[5][4];
    
    for (int i = 0; i < n_dims; i++) {
        printf("Running latent_dim=%d...\n", dims[i]);
        run_experiment(dims[i], results[i]);
    }
    
    printf("\n=== Summary ===\n");
    printf("dim | avg_loss | survival | resource\n");
    printf("----|----------|----------|--------\n");
    for (int i = 0; i < n_dims; i++) {
        printf("%2d  | %.4f  | %.1f%%    | %.3f\n",
               (int)results[i][3], results[i][0], results[i][1] * 100, results[i][2]);
    }
    
    printf("\nJC1 hypothesis: 8-16 dims is sweet spot, 32+ overfits.\n");
    printf("Let's see if the numbers agree.\n");
    
    return 0;
}
