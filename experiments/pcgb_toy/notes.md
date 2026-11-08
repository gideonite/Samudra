# PCGB toy study — lab notebook

## Summary (to be filled when stopped; ≤10 lines answering Q1–4)

_TBD_

## Setup decisions (frozen)

**Hardware profile: CPU.** 4× Intel Xeon @ 2.1 GHz (1 thread/core), 15 GB RAM, no CUDA, no MPS.
torch from PyPI (download.pytorch.org is blocked by the sandbox proxy), venv at `/home/user/pcgb-venv`.

- Every run: `torch.set_num_threads(1)`; **3 runs execute concurrently** (one per core, 4th core left for
  coordination). This holds for the whole study so wall-clock budgets are comparable across arms.
- Budget: **12 min wall clock per run** for training + PCGB scoring + intermediate evals. The final
  val/test eval is outside the budget and identical for every arm. No early stopping / checkpoint picking:
  the final weights are evaluated.
- Primary dataset KS1D (128 pts). Kolmo2D and GrayScott2D at 32×32. Control: denoise (MNIST, or a fallback
  if the sandbox blocks the download — see below).
- Model: same 4-skip ConvNeXt U-Net, ~200–500k params.
- n_train grid {128, 512, 2048}. Phase 1: 5 arms × {KS1D, Kolmo2D} × n ∈ {512, 2048} (2048 = CPU stand-in for 8192) × 3 seeds.
- PCGB scoring subset ≤ 256 samples.
- Phase 2 loop: KS1D, n=512, 3 seeds.
- Branch: the session is pinned to `ccr-3b23dd38-45kn1a` (not `autoresearch/pcgb-<tag>`); all work lives in
  `experiments/pcgb_toy/` so it does not touch the Samudra package.

## Interface contract between prepare.py and train.py

`prepare.py` (frozen after setup):
- `DATASETS = ["ks1d", "kolmo2d", "grayscott2d", "denoise"]`
- `python prepare.py --dataset <name>|--all` generates and caches to `data/<name>.pt`.
- `load(name) -> dict` with normalized float32 tensors (per-channel mean/std from the train split):
  `train`, `val`, `test`, `calib`, each shaped `[N_traj, T, C, *spatial]` (denoise: the T axis holds
  (noisy, clean) i.e. T=2), plus `meta = {periodic: bool, ndim: 1|2, channels: int, spatial: tuple,
  rollout_steps: int (10, or 1 for denoise), mean, std}`.
- `get_train_pairs(name, n_train, seed) -> (X, Y)`: a fixed subset of n_train (x_t, x_{t+1}) pairs drawn from
  the train trajectories, deterministic in (n_train, seed). Shapes `[n, C, *spatial]`.
- `get_calib_pairs(name, n=256, seed=0) -> (X, Y)` from the calibration split.
- `evaluate(model_fn, name, split="val") -> {"metric": float, "onestep_mse": float, "per_step_nrmse": list}`.
  `model_fn` maps a normalized `[B, C, *spatial]` tensor to the next normalized state. Physics: autoregressive
  10-step rollout from a fixed set of ICs; metric = per-step nRMSE (RMSE / per-channel std of the target in
  physical units, so =RMSE in normalized units), averaged over the 10 steps and channels. Denoise: metric =
  one-step nRMSE. Rollouts that go non-finite count as metric = inf (flagged).
- Splits by trajectory: train / val / test / calib are disjoint sets of trajectories (or images).

`train.py`:
- `python train.py --dataset ks1d --arm {baseline,stochdepth,mask_only,reweight_only,pcgb} --n_train 512
  --seed 0 [--budget_s 720] [--tag ...]` ; writes `runs/<tag>/log.jsonl` + appends a row to `results.tsv`.

## Data (prepare.py frozen at a8895b9)

KS1D L=22π N=128 ETDRK4, Δt_model=2.0 (lag-1 corr 0.88). Kolmo2D: ν=1/1500, drag 0.1, k_f=4, 64² sim → 32²,
Δt_model=0.15 (lag-1 0.87), Re≈1000 (estimate). GrayScott F=0.04 k=0.06 (stripes/spots), lag-1 0.98.
Denoise = real MNIST (download worked), σ=0.2. Persistence val metric: ks1d 1.14, kolmo2d 1.03, gs 0.79, denoise 0.65.

## Calibration (commit a8895b9, n=512, seed 0, 3 concurrent 1-thread jobs)

- **KS1D baseline**: metric 0.061, 11.7k steps (16 steps/s). Val curve 0.19→0.072→0.065→0.069→0.061: left the
  transient, still creeping down. Budget OK, model size kept.
- **Kolmo2D baseline**: metric 0.73, only 3.0k steps (4.2 steps/s). Val metric is best early (0.64 @1k steps) and
  rises while train loss keeps falling → **overfits at n=512** (val one-step MSE also rises). Not undertrained in the
  "shrink the model" sense; kept as is. Note this regime is overfitting-limited → regularisers have room here.
- **KS1D pcgb (v1 defaults)**: metric **0.69 vs 0.061** baseline — >10× worse. Two pathologies:
  1. **Reweighting is a no-op**: entropy = 1.000 and max/min ratio = 1.00 in every round, i.e. β = 1 always
     (L̄ = mean r/max r ≥ 0.5: per-sample residuals are too uniform). Lbar/beta now logged (f5d18a7) to confirm.
  2. **Mask oscillation**: argmax picks mask 0 (all skips dropped) → after a round on mask 0 the unmasked net
     (15) becomes the worst → picks 15/14 → back to 0 … Evals swing with it (0.14, 1.14, 0.16, 0.90, 0.31).
     The deployed net is evaluated at whatever phase the budget ends in. Not "collapse" by the 70% rule (mask 0
     ~45%), but the adversarial rule is degenerate: the worst mask is nearly always "drop everything".

## Phase 0 — premise checks

_Running: baseline grid ks1d/kolmo2d × n{128,512,2048} × 3 seeds + denoise baseline (jobs/phase0_baselines.txt);
lesion + gradient analysis delegated._

## Phase 1 — main table

_TBD_

## Phase 2 — loop log

_TBD_

## Open questions

_TBD_
