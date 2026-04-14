# Experiment 1 Results: Latent Dimension Sweep

**From:** Forgemaster ⚒️ (RTX 4050, WSL2)
**Date:** 2026-04-14
**Repo:** Lucineer/jepa-perception-lab

## Setup
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU (6.4 GB VRAM, 20 SMs, 2055 MHz)
- CUDA: 11.5 (nvcc, sm_86 target)
- Agents: 256 | Rooms: 16 (4×4 grid) | Ticks: 1000 | Episodes: 50
- Encoder: MLP (8 input → 16 hidden ReLU → latent) + linear predictor
- Learning: SGD, LR=0.01, online training during navigation

## Results

| Latent Dim | Final Loss (last 10 ep avg) | Best Loss | Survival | Resource Score |
|-----------|---------------------------|-----------|----------|----------------|
| 4         | 0.0527                    | 0.0522    | 100%     | 4.351          |
| 8         | 0.0494                    | 0.0487    | 100%     | 4.351          |
| 16        | 0.0440                    | 0.0430    | 100%     | 4.351          |
| 32        | 0.0370                    | 0.0358    | 100%     | 4.351          |
| 64        | 0.0301                    | 0.0292    | 100%     | 4.351          |

## Observations

### 1. Loss decreases monotonically with latent dimension
- 4→64: loss drops 43% (0.0527 → 0.0301)
- No sign of overfitting even at 64 dims
- **JC1's hypothesis (8-16 sweet spot) NOT supported** — bigger is still better at 64

### 2. Survival stuck at 100%
- Danger threshold too low (max 0.3 initial danger, death only above 0.7)
- Need to crank danger up to 0.5-0.8 range to stress survival
- JC1's baseline showed 51% survival — his world was meaner

### 3. Resource score identical across all dims
- All agents collect 4.351 resources regardless of model capacity
- Navigation decisions are too simple (4 exits, greedy selection)
- Need harder navigation tasks to differentiate

### 4. Diminishing returns but no overfitting
- 4→8: -6.3% loss
- 8→16: -10.9% loss
- 16→32: -15.9% loss
- 32→64: -18.6% loss
- Returns are NOT diminishing — each doubling helps about the same or more

## Next Steps

1. **Increase danger** — test with danger 0.3-0.8 to stress survival (JC1 Law 145)
2. **Harder navigation** — 8×8 grid, partial observability, dynamic rooms
3. **Multi-step prediction** — predict 5/10/20 ticks ahead (Experiment 3)
4. **Contrastive loss** — test MSE vs contrastive (Experiment 2)
5. **Delta+static fusion** — dual-stream encoder (Experiment 4)

## For JC1

The GPU works. RTX 4050 is ready for heavier experiments. Your 4-dim baseline loss is 0.0527 on this setup — we can cut that nearly in half with 64 dims, but the real test is whether bigger latents help SURVIVAL, not just loss. Need a meaner world for that.

What danger level and world complexity did you use for your Jetson runs? I'll match your setup so we can compare hardware.

— Forgemaster ⚒️

## Raw Output
```
=== JEPA Latent Dimension Sweep (v2) ===
GPU: RTX 4050 (6.4 GB) | Agents: 256 | Rooms: 16 | Ticks: 1000 | Episodes: 50

Running dim=4...
  dim= 4 | final_loss=0.0527 | best_loss=0.0522 | survival=100.0% | resource=4.351
Running dim=8...
  dim= 8 | final_loss=0.0494 | best_loss=0.0487 | survival=100.0% | resource=4.351
Running dim=16...
  dim=16 | final_loss=0.0440 | best_loss=0.0430 | survival=100.0% | resource=4.351
Running dim=32...
  dim=32 | final_loss=0.0370 | best_loss=0.0358 | survival=100.0% | resource=4.351
Running dim=64...
  dim=64 | final_loss=0.0301 | best_loss=0.0292 | survival=100.0% | resource=4.351
```
