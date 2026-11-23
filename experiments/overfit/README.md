<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Can Samudra overfit one sample? (1°, CPU)

Experiment record, not intended to merge. Figures and the full write-up:
**[Can Samudra Overfit? (Claude artifact)](https://claude.ai/artifact/LMyndZhvX7PNd979eprQPQ)**.

**Answer: yes.** With residual prediction, a zero-initialised decoder and a
dynamic loss with no weight limit, the paper's ConvNeXt U-Net beats persistence
on all 77 channels of one memorized 1° sample from step 100 on. At step 592 the
median per-channel RMSE is 0.17× persistence's and the worst is 0.47×
(salinity at 105 m), still falling when the run stopped.

## Why

m2lines/Samudra#980 (`onestep_figures_onedeg.md`) scored the paper checkpoint
one step ahead and found its temperature and salinity error stuck at ~2–4% of a
spread, losing to persistence below the surface. It proposed that the cause is
the training objective (full-state prediction, normalization by each field's
spread, dynamic loss weights capped at 20×), not the architecture. If the
network can't memorize one sample below persistence, the floor is about the
architecture or numerics; if it can, it's about how it's trained.

## Setup

- **Model:** the paper architecture (`samudra_om4/model.yaml`: ConvNeXt U-Net,
  widths 280–520, 84M parameters), 2 input and 2 output steps, bf16 autocast,
  activation checkpointing.
- **Data:** 40 steps of the public v2026-09 `om4_onedeg` store (1975-01-03 to
  1975-07-17); see the clone command in `configs/overfit_onedeg.yaml`. Forcing
  is `tau_hfds`, because `hfds_anomalies` is identically zero on a 40-step
  clone.
- **Training:** one sample, batch 1, Adam at constant learning rate, gradient
  clip 1.0. One CPU core per arm, four arms in parallel, 5–6 hours each
  (24–36 s per step).
- **Score:** per-channel masked RMSE in normalized units, divided by
  persistence's RMSE (persistence = the latest input state, for both outputs).
  Below 1 beats doing nothing. One held-out sample is also scored.

## The best recipe (arm I)

1. `pred_residuals: true`.
2. Decoder zero-initialised, so training starts at the residual base (1.2×
   persistence error) instead of ~14×.
3. Dynamic loss with no limit on the channel weights, so the deep temperature
   and salinity channels get their share of the loss.
4. Learning rate 1e-4.

`overfit.py` and `configs/overfit_onedeg.yaml` implement exactly this recipe.
To reproduce, run from the repository root:

```bash
PYTHONUNBUFFERED=1 uv run python experiments/overfit/overfit.py \
  --config experiments/overfit/configs/overfit_onedeg.yaml \
  --n-batches 1 --steps 5000 --eval-every 25 --threads 1 \
  --max-hours 5 --out .LOCAL/overfit/I
uv run python experiments/overfit/analyze.py .LOCAL/overfit/I --out .LOCAL/overfit/I_plots
```

A 21-step rerun of this command reproduced the original run's logged losses
exactly (steps 0, 10, 20).

`results/history_I_dynNone_zero_lr1e-4.json` is the per-eval history of the
original run (every 25 steps: per-channel ratios and RMSEs, train and
held-out).

## Every arm tried

Ratio = model RMSE ÷ persistence RMSE on the memorized sample, at the arm's
final step unless noted. Only arm I's code is kept in this branch. The full harness with the flags for
the other arms, and their per-eval histories, are at commit
[`39fdc30`](https://github.com/gideonite/Samudra/tree/39fdc30/experiments/overfit)
on branch `ccr-5cf4208f-2sl5if`.

| Arm | Change | lr | Steps | Median | Beat | Worst channel | Outcome |
|---|---|---:|---:|---:|---:|---|---|
| A | residual, plain MSE | 6e-4 | 889 | 0.63 | 53% | 11.2 (so_14) | beats persistence on velocities only |
| B | A in fp32 | 6e-4 | 283 | 1.34 | 49% | 23.5 (so_14) | tracks A: bf16 is not the issue |
| C | full state (paper), plain MSE | 6e-4 | 909 | 0.93 | 51% | 16.5 (so_18) | worse than A |
| D | residual, dynamic loss limit 20 (paper loss) | 6e-4 | 719 | 0.15 | 94% | 2.19 (so_18) | deep T/S still lose |
| E | D + zero-init decoder | 6e-4 | 100 | 0.66 | 77% | 3.67 | killed at step 100 to free a core |
| F | residual, dynamic no limit, zero-init | 6e-4 | 702 | 0.60 | 84% | 4.71 (so_16) | 100% at step 300; best at 650 (median 0.19, worst 0.52), then spiked |
| G | residual, MSE × 1/(persistence MSE) per channel, zero-init | 6e-4 | 881 | 0.74 | 84% | 4.19 (so_15) | 100% at step 250; best at 750 (0.20, 0.33), then spiked |
| H | residual, dynamic limit 20, zero-init | 1e-4 | 694 | 0.22 | 100% | 0.95 (so_17) | deep salinity hovers at 0.85–1.0 |
| **I** | **residual, dynamic no limit, zero-init** | **1e-4** | **592** | **0.17** | **100%** | **0.47 (so_5)** | **all 77 from step 100, still improving** |

## Findings

1. **The architecture and bf16 are not what holds the paper checkpoint's
   deep-ocean error up.** The same network memorizes one sample below
   persistence on every channel, and fp32 (B) tracked bf16 (A) to within a few
   percent.
2. **Loss weighting is the biggest lever.** Under plain MSE the model fits only
   the 38 velocity channels for hundreds of steps. With the paper's 20× limit,
   deep salinity stays at or near persistence (D, H). Removing the limit (I) or
   weighting by 1/persistence MSE (G, a single-sample stand-in for #980's σ_Δ
   normalization) fits the deep channels.
3. **Starting at the residual base matters.** Residual prediction with a random
   decoder starts at ~14× persistence error; zero-initialising the decoder
   starts at 1.2×.
4. **lr 6e-4 is unstable once deep channels carry weight.** F and G reached
   100% and then spiked repeatedly (worst channel up to 8.4×). At 1e-4 neither H
   nor I spiked.

## Limits

- One memorized sample. Held-out median ratios stay at 1.16–1.27 for arm I, so this says nothing
  about generalization; #980's ablation as a real training run is the next
  step.
- Not fully memorized: error was still falling at ~600 steps.
- G's weights came from the memorized sample's own persistence error, not
  dataset-wide σ_Δ, and 1/persistence weighting at lr 1e-4 was not run.
- With 2 input and 2 output steps, residual output t+1 is anchored on x[t-1],
  not x[t] (which is why the zero-correction start is 1.2×, not 1.0×). Anchoring
  both outputs on x[t] was not tested.

## Provenance

- Runs: 2026-11-22. A, C, D 05:48–11:50 UTC; B 06:57–11:48 (after an OOM kill);
  E 11:53–12:52; F, G, H 11:53–17:54; I 12:55–17:54.
- Hardware: 4 vCPU Xeon (AMX bf16), 16 GB RAM, one core per arm.
- `results/history_I_dynNone_zero_lr1e-4.json` was produced by the full harness
  at commit `cd36b78` with `--loss dynamic --dynamic-limit none
  --zero-init-decoder --lr 1e-4` and the other flags as above; it predates the
  `provenance` field that `overfit.py` now writes into `history.json`.
