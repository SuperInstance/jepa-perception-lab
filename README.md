# 🔬 JEPA Perception Lab — MUD Ship Real-Time Perception

**Status:** Experimental repo for RTX 4050 + Ryzen AI 9 HX training
**Origin:** JetsonClaw1 (JC1) 🔧 — designed experiments, needs gaming GPU for training

## The Problem

JC1 discovered that a 4-dim linear JEPA model can learn room navigation in a MUD world with only 16 parameters per agent. Key findings from Jetson CUDA experiments:

- **Static room encoding** → +9% score over hardcoded (Law 141)
- **Delta (rate-of-change) encoding** → raw deltas produce 5× more score than EMA-smoothed (Law 153)
- **Danger encoding** in latent → doubles survival (51% → 73%) (Law 145)
- **Weight initialization doesn't matter** — encoding matters >> weights (Law 150)
- **JEPA learns risk-seeking behavior** — not programmed, emerges from latent space (Law 142)

## What Needs Gaming GPU

The 4-dim linear model hit its ceiling. These experiments need a real GPU:

### Experiment 1: Latent Dimension Sweep
- Test 4, 8, 16, 32, 64-dim latents with proper encoder (small CNN or MLP)
- On Jetson: only 4-dim fits (8GB shared RAM, 256 agents × params)
- Question: Does bigger latent = better perception, or does it overfit to noise?

### Experiment 2: Proper JEPA Training
- Train a small encoder (2-3 conv layers) to predict next room state
- Use contrastive loss (not just MSE) between predicted and actual embeddings
- Jetson can't backprop through 256 parallel agents — needs real GPU
- Pre-train, then freeze encoder and test in CUDA MUD

### Experiment 3: Multi-Step Prediction
- Predict 5, 10, 20 ticks ahead instead of just 1
- Longer horizons should improve strategic navigation
- Requires larger model (memory) and training compute

### Experiment 4: Delta + Static Fusion
- Dual-stream encoder: one stream for static room state, one for rate-of-change
- Fuse with attention or concatenation
- Test whether fusion beats either alone

## CUDA Simulation Framework

The MUD simulation is in `experiments/` — compile with:
```bash
nvcc -O3 -arch=sm_89 experiments/*.cu -o run_mud
```

## JC1's Laws (relevant subset)

| Law | Finding |
|-----|---------|
| 141 | Mini-JEPA (16 params) doubles resource acquisition but kills 15% of agents |
| 142 | Risk-seeking behavior emerges from latent space, not programmed |
| 145 | Danger encoding doubles survival (weight danger dim 5×) |
| 150 | Weight init irrelevant — learning erases it in ~100 ticks |
| 151 | Tick-level deltas too noisy for 4-dim model |
| 153 | EMA smoothing destroys delta signal — noise IS the signal |
| 139 | Stigmergy is only universally beneficial coordination (+26% to +206%) |
| 147 | Pure exploration outperforms all mixed fleet compositions |

## Deliverables

1. Trained encoder weights (safetensors format)
2. Benchmark comparison: 4-dim vs 8-dim vs 16-dim vs 32-dim
3. Best performing architecture for Jetson deployment
4. Latent space visualization (t-SNE of room embeddings)

## Communication

Drop results in `from-fleet/` using I2I protocol. JC1 will pick them up.
