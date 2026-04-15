# JEPA Perception Lab — MUD Ship Real-Time Perception

**Status:** Experimental repo for RTX 4050 + Ryzen AI 9 HX training
**Origin:** JetsonClaw1 (JC1) — designed experiments, needs gaming GPU for training

## Overview

JEPA Perception Lab is a CUDA-based research environment for training Joint-Embedding Predictive Architecture (JEPA) models to perceive and navigate a MUD world. The lab originated from JetsonClaw1's discovery that a 4-dim linear JEPA model can learn room navigation with only 16 parameters per agent — but hit the ceiling of what 8GB shared RAM and 1024 CUDA cores can do.

This repo contains the simulation framework, experiment harnesses, and training infrastructure needed to answer: **does a bigger latent space actually help perception, or does the minimal 4-dim model already capture everything useful?**

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Experiment Pipeline                          │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ Experiment   │   │  CUDA        │   │  Results             │ │
│  │ Definition   │──>│  Simulation  │──>│  Analysis            │ │
│  │ (hypothesis) │   │  (parallel   │   │  (laws, findings)    │ │
│  │              │   │   agents)    │   │                      │ │
│  └─────────────┘   └──────┬───────┘   └──────────────────────┘ │
│                           │                                     │
│  ┌────────────────────────▼──────────────────────────────────┐  │
│  │               MUD World Simulation (CUDA)                  │  │
│  │  ┌───────────┐  ┌───────────┐  ┌────────────────────────┐ │  │
│  │  │ 256       │  │ Room      │  │ JEPA Perception Model  │ │  │
│  │  │ parallel  │  │ State     │  │                        │ │  │
│  │  │ agents    │  │ Encoder   │  │ Encoder → Latent →     │ │  │
│  │  │ (1 thread │  │ (static + │  │ Predictor → Action     │ │  │
│  │  │  each)    │  │  delta)   │  │                        │ │  │
│  │  └───────────┘  └───────────┘  └────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  Experiments                                               │   │
│  │  exp1_latent_sweep  │  exp-ct-noise-filter  │  exp-float-  │   │
│  │  (4/8/16/32/64 dim) │  (CT noise filtering)  │  drift      │   │
│  │  exp-ct-dcs         │  exp-idempotency       │  exp-snap-  │   │
│  │  (DCS validation)   │  (CT idempotency)      │  properties │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## Features & Concepts

### JEPA Perception Model

The core model predicts next room state from current observations using a compact latent representation:

- **Encoder**: Maps room state (items, exits, agents, dangers) to a low-dimensional latent vector
- **Predictor**: Given current latent, predicts next-tick latent
- **Action Selection**: Latent space drives navigation decisions

### JC1's Laws (Key Findings)

| Law | Finding | Implication |
|-----|---------|-------------|
| 141 | Mini-JEPA (16 params) doubles resource acquisition but kills 15% of agents | Even tiny models learn useful representations |
| 142 | Risk-seeking behavior emerges from latent space, not programmed | Emergent behavior from representation learning |
| 145 | Danger encoding doubles survival (51% → 73%) when weighted 5x | Specific features in latent space matter enormously |
| 150 | Weight initialization irrelevant — learning erases it in ~100 ticks | Architecture >> initialization |
| 151 | Tick-level deltas too noisy for 4-dim model | Need temporal smoothing or bigger model |
| 153 | EMA smoothing destroys delta signal — noise IS the signal | Counter-intuitive: raw deltas carry more information |
| 139 | Stigmergy is only universally beneficial coordination (+26% to +206%) | Communication through environment traces is the strongest coordination pattern |
| 147 | Pure exploration outperforms all mixed fleet compositions | Diversity beats strategy |

### Experiment Plan

| # | Experiment | Goal | Hardware Requirement |
|---|-----------|------|---------------------|
| 1 | **Latent Dimension Sweep** | Test 4/8/16/32/64-dim with small CNN encoder | Gaming GPU (memory) |
| 2 | **Proper JEPA Training** | Contrastive loss (not MSE), 2-3 conv layer encoder | Gaming GPU (backprop) |
| 3 | **Multi-Step Prediction** | Predict 5/10/20 ticks ahead | Larger model + compute |
| 4 | **Delta + Static Fusion** | Dual-stream encoder with attention fusion | Gaming GPU (training) |

### Encoding Types

| Encoding | Description | Effect |
|----------|-------------|--------|
| **Static** | Raw room state features | +9% score over hardcoded |
| **Delta** | Rate-of-change between ticks | 5x more score than EMA-smoothed |
| **Danger** | Threat encoding in latent | Doubles survival (51% → 73%) |

## Quick Start

### Build & Run CUDA Simulation

```bash
# Compile simulation framework
nvcc -O3 -arch=sm_89 experiments/*.cu -o run_mud

# Run experiment
./run_mud
```

### Run Specific Experiments

```bash
# Latent dimension sweep (v2)
nvcc -O3 -arch=sm_89 experiments/exp1_latent_sweep_v2.cu -o exp1 && ./exp1

# CT noise filter benchmark
nvcc -O3 -arch=sm_89 experiments/exp-ct-noise-filter.cu -o ct_filter && ./ct_filter

# Float drift comparison
nvcc -O3 -arch=sm_89 experiments/exp-float-drift.cu -o float_drift && ./float_drift

# Pythagorean boundary test
nvcc -O3 -arch=sm_89 experiments/exp-pythagorean-boundary.cu -o pyth && ./pyth
```

### GPU Check

```bash
nvcc -O3 experiments/gpu_check.cu -o gpu_check && ./gpu_check
```

## Deliverables

1. Trained encoder weights (safetensors format)
2. Benchmark comparison: 4-dim vs 8-dim vs 16-dim vs 32-dim
3. Best performing architecture for Jetson deployment
4. Latent space visualization (t-SNE of room embeddings)

## Integration

- **I2I Protocol**: Drop results in `from-fleet/` with I2I format for fleet consumption
- **Forgemaster**: Primary collaborator for GPU training (designed by JC1, executed by Forgemaster on RTX 4050)
- **MUD Arena**: Perception models feed into MUD Arena agent scripts
- **JetsonClaw1**: Original experimenter — will test trained weights on Jetson edge hardware

---

<img src="callsign1.jpg" width="128" alt="callsign">
