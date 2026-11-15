# PCGB toy study — lab notebook

## Summary (Q1–4; CPU profile, 12-min wall-clock runs, 3 seeds; test table below, details in sections further down)

1. **Premise half true**: deep blocks get ≤1% of gradient, but the baseline is a co-adapted long-path function, not a path ensemble (KS1D: any single skip/block lesion ×3–35); on Kolmo2D/denoise the deep paths are just unused.
2. **PCGB loses at matched compute**: ×9 worse than baseline on KS1D (test 0.615 vs 0.068, n=512), ×1.1–2 worse on Kolmo2D/denoise/Gray–Scott; random stochastic depth beats it everywhere.
3. **Neither half helps**: adversarial skip masking does all the damage (oscillates "drop all" ↔ "keep all", worse with more data); AdaBoost.R2 reweighting is a no-op as specified (L̄ 0.67–0.82 → β=1) and neutral when forced live (reweight_only ≡ baseline).
4. **No physics/small-n specificity**: PCGB's deficit shrinks as n falls (KS1D ×9 → ×2.4 at n=128) but never flips; its one win (Kolmo2D n=128, −0.12) is dwarfed by stochdepth (−0.45); denoise behaves the same.
5. ~25 Phase-2 variants (mask rules, drop modes, block masks, warm-up, live/pixel reweighting, round length, Adam reset) never came within ×6 of baseline on KS1D or beat baseline on Kolmo2D n=512.
6. Mask-trained nets *are* more ensemble-like (top-3 masked-subnet average beats their own unmasked net, e.g. 0.48→0.32), but remain far worse than plain training.
7. **Best config**: plain baseline for the co-adapted chaotic 1D system; random stochastic depth (skip p=0.2, block p≤0.1) where data limits (Kolmo2D, denoise, Gray–Scott). It also extends Kolmo2D's predictability horizon (15–17 vs 8–13 steps).
8. **Budget check (10×, 120-min runs, ks1d+kolmo2d n=512, 3 seeds):** conclusions unchanged. Baselines are data-limited (ks1d 0.056, kolmo 0.728); stochdepth keeps its Kolmo2D lead (0.513) and gets worse on KS1D (0.335); PCGB improves (ks1d 0.344, kolmo 0.803) but saturates far above the best arm. Val curves plateau within the runs.

### Main table — TEST split (10-step rollout nRMSE; denoise 1-step), mean ± std over 3 seeds

| arm | ks1d 128 | ks1d 512 | ks1d 2048 | kolmo 128 | kolmo 512 | kolmo 2048 | denoise 512 | grayscott 512 |
|---|---|---|---|---|---|---|---|---|
| baseline | **0.240±.022** | **0.068±.001** | **0.032±.002** | 1.144±.025 | 0.760±.029 | 0.543±.013 | 0.217±.005 | 0.664±.008 |
| stochdepth | 0.330±.008 | 0.243±.011 | 0.255±.028 | **0.705±.025** | **0.541±.007** | **0.509±.009** | **0.193±.003** | **0.642±.023** |
| mask_only | – | 0.503±.035 | 0.682±.058 | – | 0.905±.114 | 1.082±.193 | – | – |
| reweight_only | – | 0.069±.001 | 0.032±.002 | – | 0.765±.033 | 0.543±.011 | 0.216±.006 | 0.671±.007 |
| pcgb (v1) | 0.581±.041 | 0.615±.065 | 0.579±.037 | 1.096±.050 | 0.922±.095 | 1.018±.125 | 0.420±.031 | 0.713±.021 |
| pcgb+warm0.5 | 0.521±.042 | 0.523±.124* | – | 1.029±.040 | – | – | 0.487±.085 | 0.667±.021 |

*seeds 3–5 only (seed 0–2 models lost to the tag-collision bug). Persistence (test): ks1d 1.15, kolmo 1.04,
denoise 0.65, grayscott 0.66. The val→test best arm agrees in all 8 columns.

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

## Phase 0 — premise checks (phase0_analysis.py; figs/phase0_*.png; n=512 seed-0 baselines)

Val metric: KS1D 0.061 (still improving), Kolmo2D 0.73 (best 0.64 @1k steps, then overfits), Denoise 0.217
(best 0.18 @~500 steps, then overfits).

- **Lesions (rel. to unmasked)**: KS1D is a *cliff*: any single skip drop ×7.6–35, any single block ×3.3–21,
  remove-k curve flat at ~×16–18 from k=1 (no Veit-style graceful degradation). Kolmo2D/Denoise: mild; the
  bottleneck blocks (d3, mid, u5) and in Denoise skips 2–3 are free to drop (×0.94–1.0); damage sits in the
  high-res blocks (d0, u7, u8, final) and skip 0.
- **Gradient**: concentrated at the high-res ends (KS1D: d0+d1 ≈55% of grad energy; Kolmo/Denoise: u8+final
  70–80%). The three deepest blocks get ≤0.5% (KS1D) / 0.1–1% (2D); deepest skip 1–8% of skip grad energy.
  Stable over training.
- **Path length**: the function is carried by long, near-full-depth paths; masked-net loss only beats
  persistence with ≥9 of 14 units kept (KS1D). Not the "ensemble of short paths" picture.
- **PCGB v1 reweighting is inert by construction here**: L̄ measured 0.67 (KS1D), 0.75 (Kolmo2D), 0.82 (PCGB net);
  per-sample residuals are near-homogeneous (max/median r ≈ 1.2–1.5) → β = 1 → D' = D every round.
- **Conclusion**: Premise partly true for gradient *distribution* (deep paths get almost no gradient) but false
  for the *ensemble* claim: the baseline is a co-adapted long-path function, not a sum of redundant paths. In the
  2D/denoise sets the deep paths are simply unused (free to remove) and the binding problem is overfitting at
  n=512, not gradient starvation. Little room for PCGB as specified; Phase 1 run anyway (as v1, reweight no-op
  included — that is part of the answer).

### Baseline seed-noise bar (val metric, mean ± std over 3 seeds)

| dataset | n=128 | n=512 | n=2048 | persistence |
|---|---|---|---|---|
| ks1d | 0.205 ± 0.026 | 0.060 ± 0.004 | 0.025 ± 0.003 | 1.14 |
| kolmo2d | 1.140 ± 0.034 | 0.741 ± 0.028 | 0.528 ± 0.014 | 1.03 |

Phase 2 keep threshold on ks1d n=512: 2× std ≈ 0.008 (using the larger of incumbent/baseline std).
Kolmo2D at n=128 is *worse than persistence* (overfits badly in 12 min).

## Phase 1 — main table (v1 defaults, val metric = 10-step rollout nRMSE, mean ± std over 3 seeds)

| arm | ks1d n=512 | ks1d n=2048 | kolmo2d n=512 | kolmo2d n=2048 |
|---|---|---|---|---|
| baseline | **0.060 ± 0.004** | **0.025 ± 0.003** | 0.741 ± 0.028 | 0.528 ± 0.014 |
| stochdepth | 0.218 ± 0.015 | 0.234 ± 0.031 | **0.530 ± 0.012** | **0.496 ± 0.005** |
| mask_only | 0.506 ± 0.035 | 0.691 ± 0.057 | 0.905 ± 0.110 | 1.077 ± 0.196 |
| reweight_only | 0.061 ± 0.004 | 0.026 ± 0.003 | 0.743 ± 0.031 | 0.527 ± 0.007 |
| pcgb | 0.619 ± 0.065 | 0.587 ± 0.042 | 0.924 ± 0.093 | 1.008 ± 0.128 |

- No arm hit the 70% mask-collapse flag, but argmax picks mask 0 (all skips dropped) 34–47% of rounds and
  oscillates 0 ↔ 15/14 on ks1d (0 ↔ 8/12 on kolmo2d). Weights never moved (β=1) → reweight_only ≡ baseline and
  pcgb ≡ mask_only within noise (a useful internal consistency check).
- **Masking (v1) is catastrophic**: ×8–27 worse on ks1d, ×1.2–2 worse on kolmo2d, and gets *worse* with more data.
- **Random stochastic depth** splits by regime: −28% (n=512) / −6% (n=2048) on overfitting-limited Kolmo2D,
  ×3.6–9 worse on co-adapted KS1D. Its benefit on Kolmo shrinks with n → it acts as a regulariser.
- Phase 2 starts from incumbent = pcgb v1 on ks1d n=512 (0.619 ± 0.065); the bar to beat overall is the baseline.

## Phase 2 — protocol

All Phase 2 options were implemented once as flags (defaults = v1, verified bit-identical with a fake clock).
Each experiment = ONE flag change on top of the incumbent config; "keep" = adopt the flag into the incumbent;
"revert" = drop it. Independent single-flag experiments vs the same incumbent are screened as a batch, then
keepers are combined one at a time.

## Phase 2 — loop log

_TBD_

## Open questions

_TBD_

## Phase 2 — loop log (ks1d, n=512, 3 seeds; val metric mean ± std)

### Screen 1 (vs incumbent pcgb v1 = 0.619 ± 0.065; keep needs Δ > 0.13). All runs also did `--ensemble_topk 3`.

| change | metric | ens(top-3 masked) | verdict |
|---|---|---|---|
| mask_rule=random | 0.980 ± 0.095 | 0.38 | revert — random masks are worse than adversarial |
| mask_rule=argmin | 0.059 ± 0.004 | 0.55 | **degenerate**: collapses to mask 15 (all-keep) 29/29 rounds = baseline. Reject |
| mask_exclude_all_drop | 0.567 ± 0.182 | 0.33 | revert (noise) |
| drop_mode=rescale | 0.528 ± 0.062 | 0.34 | revert (Δ 0.09 < 0.13) |
| rw_sq (reweighting becomes live) | 0.518 ± 0.050 | 0.37 | revert (Δ 0.10) |
| mask_target=blocks | 0.486 ± 0.232 | 0.66 | revert (huge variance) |
| **warm_frac=0.5** | **0.478 ± 0.016** | 0.32 | **keep** (Δ 0.141 > 0.13) |
| mask_rule=softmax | 0.531 ± 0.221 | 0.32 | revert |

- Every masking variant stays 6–16× worse than baseline (0.060). The closer a variant gets to "no masking"
  (argmin → all-keep; warm-up → fewer masked rounds), the better it does → masking itself is the harm.
- **Ensemble test**: for mask-trained nets the top-3 masked-subnet ensemble beats the unmasked net (e.g. 0.98→0.38,
  0.48→0.32): masked training does create usable sub-networks. For the baseline-like net (argmin) the ensemble is
  ~10× worse than the unmasked net. So the "ensemble" view only has teeth for nets trained to be ensembles, and
  those nets are still far worse than plain training. The deployed unmasked net is the wrong readout for PCGB.
- Bug found: run tags omitted flags, so screen-1 run dirs overwrote each other (results.tsv rows are intact; only
  per-round logs lost). Fixed: tag now ends with a hash of non-default flags.

### Screen 2 (incumbent pcgb + warm_frac=0.5 = 0.478 ± 0.016). Keep rule: Δ > 2·max(std_inc, std_cand).

| change | metric | verdict |
|---|---|---|
| warm_frac 0.8 (instead of 0.5) | 0.628 ± 0.065 | revert — more warm-up is *worse* (masked rounds compressed into 20% at low LR) |
| + rw_sq | 0.490 ± 0.071 | revert |
| + drop_mode=rescale | 0.409 ± 0.097 | inconclusive (Δ 0.07 < 0.19); +3 seeds queued |
| + mask_target=blocks | 0.629 ± 0.050 | revert |
| + reset_adam | 0.599 ± 0.066 | revert |
| + n_rounds=10 (longer rounds) | 0.919 ± 0.266 | revert — long masked rounds wreck the net |
| reweight_only + rw_sq (live, ratio ≈1.4) | 0.059 ± 0.002 | = baseline (0.060 ± 0.004) |
| reweight_only + pixel weights + rw_sq | 0.059 ± 0.006 | = baseline; pixel weights stay inert (β=1, L̄ 0.64–0.78) |
| reweight_only + p95, β cap removed (ratio ≈3.1, anti-boosting since β>1) | 0.058 ± 0.004 | = baseline |

- **Reweighting, even when made live (either direction), is neutral on ks1d.** All masking variants remain ≥6× worse
  than baseline. Q3 answer so far: masking hurts, reweighting does nothing; neither half helps.

### Rollout horizon (horizon.py, figs/horizon_*.png; Phase-1 models, val, 50 steps, ICs t0∈{0,5,10})

Decorrelation horizon = first step with nRMSE > 1 (mean over seeds):

| | baseline | stochdepth | pcgb | mask_only | reweight_only |
|---|---|---|---|---|---|
| ks1d n512 | 24.7 | 17.3 | 8.7 | 10.0 | 25.3 |
| ks1d n2048 | 30.0 | 17.7 | 9.0 | 8.0 | 29.3 |
| kolmo2d n512 | 8.3 | **15.3** | 6.3 | 6.7 | 8.7 |
| kolmo2d n2048 | 13.3 | **16.7** | 5.7 | 5.0 | 12.7 |

- The step-10 ranking holds at long horizons. On ks1d all arms reach the climatology floor (~1.4) by step 50, so the
  gaps show up as a shorter predictability horizon, not instability.
- Kolmo2D: stochdepth has roughly the *same one-step error* as baseline, but its lead grows with horizon. It stays
  bounded (1.25 at step 50) while baseline/reweight_only drift to 1.5–2.2, worse than predicting zeros. It is
  mildly smoothed (std ratio 0.86–0.92), and that is why it stays bounded.
- Masked arms (pcgb/mask_only) on Kolmo2D grow energy (std ratio 1.6–2.6 at step 50; nRMSE 1.8–2.8). This is soft
  instability; hard blow-ups (non-finite or |x|>20) are ≤ 2% everywhere.

### Screen 3: reseed, regime (n=128) and generalisation (denoise, grayscott2d at n=512)

- Rescale vs incumbent with 6 seeds: 0.425 ± 0.088 vs 0.497 ± 0.090 → revert. With 6 seeds the incumbent's own std
  is 0.090, so the earlier warm_frac keep (0.619 → 0.497) is **marginal**: it does not clear 2σ with the larger sample.

| config (val, 3 seeds) | baseline | stochdepth | reweight_only | pcgb v1 | pcgb+warm0.5 |
|---|---|---|---|---|---|
| ks1d n=128 | **0.205 ± 0.026** | 0.311 ± 0.019 | – | 0.574 ± 0.055 | 0.520 ± 0.030 |
| kolmo2d n=128 | 1.140 ± 0.034 | **0.692 ± 0.025** | – | 1.096 ± 0.039 | 1.024 ± 0.050 |
| denoise n=512 | 0.217 ± 0.004 | **0.193 ± 0.003** | 0.217 ± 0.004 | 0.417 ± 0.030 | 0.483 ± 0.083 |
| grayscott2d n=512 | 0.682 ± 0.037 | 0.664 ± 0.034 | 0.686 ± 0.027 | 0.742 ± 0.028 | 0.695 ± 0.015 |

- **Regime (Q4)**: PCGB's deficit vs baseline shrinks as n falls on ks1d (×10 at 512 → ×2.8 at 128). Kolmo2D n=128
  is the only place a PCGB arm beats baseline (warm: −0.12, just over 2σ), but there stochdepth is −0.45 and PCGB
  stays at the persistence level. PCGB behaves like a weak, badly tuned regulariser: useful only where plain training
  overfits catastrophically, and dominated by random stochastic depth there.
- **Physics vs non-physics**: no physics-specific benefit. On denoise PCGB is ×2 worse (stochdepth −11%); on
  non-chaotic grayscott all arms are within noise except pcgb v1 (+9%).

### Screen 4 — Kolmo2D n=512 probe (can any PCGB variant approach stochdepth where regularisation matters?)

| config (pcgb + warm0.5 + …) | val | vs baseline 0.741 / stochdepth 0.530 |
|---|---|---|
| (nothing) | 1.143 ± 0.122 | worse — warm-up *hurts* here (masked rounds pushed into low-LR half) |
| mask_rule=random | 0.810 ± 0.046 | best PCGB variant, still worse than baseline |
| mask_target=blocks | 0.887 ± 0.074 | worse |
| mask_rule=softmax | 1.050 ± 0.100 | worse |
| drop_mode=rescale | 1.095 ± 0.050 (3 seeds; one MASK_COLLAPSE on mask 0) | worse |

→ No PCGB variant beats baseline on Kolmo2D n=512; per-step stochastic depth remains far ahead. Round-level masking
(even random) is a much worse regulariser than per-step random dropping. **Loop stopped here**: the questions are
answered cleanly, and further PCGB tuning has shown no trend toward baseline.

## Long-budget check (10× = 120 min/run; jobs/long10x.txt)

Narrowed to the open question "do the gaps close with more training?": ks1d and kolmo2d at n=512 × {baseline,
stochdepth, pcgb (n_rounds=300 to keep ~24 s rounds)} × 3 seeds, 12 intermediate evals. Same profile (3 concurrent
1-thread jobs). Dropped reweight_only (inert by construction), denoise/grayscott and the other PCGB variants.

### KS1D n=512 at 10× budget (val, 3 seeds)

| arm | 12 min | 120 min | Δ |
|---|---|---|---|
| baseline | 0.060 ± 0.004 | 0.056 ± 0.010 | ≈ same: data-limited; val plateaus from ~50% of the run while train loss → 2e-8 |
| stochdepth | 0.218 ± 0.015 | 0.335 ± 0.138 | worse, high variance (0.48 / 0.32 / 0.20); val drifts up late in training |
| pcgb (300 rounds) | 0.619 ± 0.065 | 0.344 ± 0.012 | better: mask oscillation damps; still ×6 worse than baseline |

→ The KS1D verdict is not a short-budget artefact. More training does not close the gap: PCGB's improvement
saturates far above baseline, and stochastic depth gets worse.

### Kolmo2D n=512 at 10× budget (val, 3 seeds)

| arm | 12 min | 120 min |
|---|---|---|
| baseline | 0.741 ± 0.028 | 0.728 ± 0.020 (val best 0.62 at ~25% of the run, then overfits to 0.75) |
| stochdepth | 0.530 ± 0.012 | **0.513 ± 0.014** (plateaus ~0.51 from ~25% of the run) |
| pcgb (300 rounds) | 0.924 ± 0.093 | 0.803 ± 0.019 (still oscillating 0.66–1.0 between evals) |

→ Same picture as KS1D. More training leaves the baseline unchanged (it is limited by overfitting), stochdepth keeps
its lead, and PCGB improves but stays worse than baseline. Even an oracle-early-stopped baseline (0.62) loses to
stochdepth (0.51).

### 10× conclusion
The Q1–4 answers survive a 10× budget. Every arm's val metric plateaued well inside the 120-min runs, so these runs
are effectively converged; PCGB's apparent gains with budget saturate far above the best arm.

## Write-up
Claude doc (paper tab + plain-language summary tab): https://claude.ai/artifact/UXKrHZ8hZzcNhP5fK4duWT
Figures from paper_figs.py and example_figs.py. Corrections found while writing it up:
- Gray–Scott reweighting is *not* inert (L̄ < 0.5 in 90% of rounds, min β 0.18), yet still neutral (test 0.671 vs
  baseline 0.664). KS1D has β < 1 in 3.8% of rounds; Kolmo2D and denoise never do.
- Phase 2 had 21 distinct traceable configurations, not ~25.

## Gradient boosting variants

Code: `--rw_kind {adaboost,gb}` (default adaboost = unchanged path) and new arm `gboost` (`--gb_stages K`, `--gb_width`, `--gb_nu`), commit `da88b5f`;
jobs in `jobs/gb_variants.txt`; figure `figs/gb_variants.png`. All runs: n_train=512, 12-min budget, val split, seeds 0-2, rollout nRMSE (lower is better).
Reference rows are the existing 12-min default-flag runs in results.tsv (summarize.py). "Win" rule: Δmean (ref − new) > 2 × the larger seed std of the pair.

**Implementation.** `gb` weighting: t_i = mean_pix(r_i²)/mean_j(...), D_i ← (1−lam)D_i + lam·t_i on the scored subset, then the existing median clamp (clamp_ratio=5) and mean-1 renorm; Lbar logged, beta = NaN; pixel mode raises.
`gboost`: K separate UNets (stage width auto-chosen so K·params ≈ baseline: K=4 → width 5, K=2 → width 7; totals 241k/233k on ks1d, 276k/263k on 2D vs baseline 235k/263k; ratios 0.99–1.05), stage k trains budget/K s (cosine LR per stage, same AdamW/clip/mse), stage k>1 fits y − F_{k−1}(x) with F_{k−1}(X) cached under no_grad.
Because the UNet adds x internally, stages k>1 output `model(x) − x` (head-only correction); F_K = h_1 + ν Σ h_k, ν=1. Intermediate val evals are charged to the budget as in other arms.

### Val nRMSE, mean ± std over 3 seeds (n=512, 12 min)

| arm | ks1d | kolmo2d | grayscott2d |
|---|---|---|---|
| baseline | **0.0600 ± .0038** | 0.7406 ± .0276 | 0.6824 ± .0366 |
| stochdepth | 0.2182 ± .0153 | **0.5300 ± .0119** | **0.6635 ± .0335** |
| reweight_only (adaboost) | 0.0606 ± .0035 | 0.7434 ± .0313 | 0.6861 ± .0269 |
| reweight_only (gb) | 0.0602 ± .0037 | 0.7446 ± .0218 | 0.6736 ± .0272 |
| pcgb (adaboost) | 0.6189 ± .0653 | 0.9243 ± .0925 | 0.7419 ± .0280 |
| pcgb (gb) | 0.5550 ± .0753 | 0.9628 ± .0869 | 0.7367 ± .0228 |
| gboost K=2 | 0.0823 ± .0070 | 0.7760 ± .0078 | 0.7002 ± .0270 |
| gboost K=4 | 0.1340 ± .0173 | 0.8728 ± .1705* | 0.7571 ± .0168 |

*one seed (2) blew up to 1.069 in rollout (0.79 / 0.76 for the other two); its one-step val was fine, so it is a rollout instability, not a fit failure.

Wins under the 2σ rule: none of the four new configs beats baseline on any dataset (gboost K=2/K=4 are significantly worse on ks1d, K=4 also on Gray-Scott; everything else is a tie), and none beats stochdepth on Kolmo2D/Gray-Scott. The only "wins" are over pcgb(adaboost) (reweight_only-gb on Gray-Scott: Δ=+0.068; gboost on ks1d), i.e. over an arm that is itself bad.
pcgb with gb weighting vs pcgb with adaboost weighting: ties on all three datasets (ks1d −0.064, kolmo2d +0.038, gray-scott −0.005, all < 2σ). The mask damage dominates regardless of the weighting.

### Weight-spread diagnostics (gb; D max/min over the 29 rounds, 3 seeds per cell)

| arm / dataset | ratio median over rounds | max over rounds | final D max / D min | Lbar median |
|---|---|---|---|---|
| reweight_only ks1d | 2.3 | 2.6–2.7 | 1.5–1.7 / 0.71–0.73 | 0.74–0.76 |
| reweight_only kolmo2d | 2.8 | 3.2–3.3 | 1.7–1.8 / 0.53 | 0.75–0.76 |
| reweight_only grayscott2d | 5.0 (clamp active from round 1+) | 5.0 | 2.1–2.2 / 0.42–0.44 | 0.43–0.53 |
| pcgb ks1d | 2.3–2.6 | 2.9–3.6 | 1.4–1.9 / 0.51–0.55 | 0.81 |
| pcgb kolmo2d | 1.8–2.1 | 2.1–2.3 | 1.6–1.7 / 0.70–0.78 | 0.82–0.83 |
| pcgb grayscott2d | 5.0 | 5.0 | 2.2–2.3 / 0.45–0.46 | 0.53–0.56 |

So D does move (the adaboost version is exactly 1.0 on ks1d/kolmo2d), by a factor 2–3 on ks1d/kolmo2d and up to the 5× clamp on Gray-Scott, yet the normalised entropy stays ≥0.98 (most samples stay near 1, a tail is up-weighted).
Being live changes nothing measurable: reweight_only gb ≡ reweight_only adaboost ≡ baseline within noise (Δ ≤ 0.012 everywhere). The weighting is not the lever; with n=512 and per-sample residuals that track a smooth per-sample difficulty, emphasising hard samples neither helps nor hurts at this strength (lam=0.3, clamp 5).

### gboost per-stage losses (final train mse of the stage's target; mean over 3 seeds)

| dataset, K | stage-k train loss (k=1..K) | val nRMSE of F_k (k=1..K) |
|---|---|---|
| ks1d K=2 | 2.0e-5 → 6.8e-7 | 0.073 → 0.082 |
| ks1d K=4 | 8.7e-5 → 7.4e-6 → 7.0e-6 → 6.5e-6 | 0.105 → 0.134 → 0.134 → 0.134 |
| kolmo2d K=2 | 0.039 → 0.021 | 0.747 → 0.776 |
| kolmo2d K=4 | 0.062 → 0.047 → 0.038 → 0.032 | 0.83 → 0.84 → 0.86 → 0.87 (medians 0.73 → 0.75 → 0.78 → 0.79) |
| grayscott2d K=2 | 5.3e-4 → 1.5e-4 | 0.698 → 0.700 |
| grayscott2d K=4 | 1.4e-3 → 6.2e-4 → 5.0e-4 → 4.5e-4 | 0.751 → 0.756 → 0.756 → 0.757 |

Stages k>1 do reduce the train residual (ks1d: ~10× from stage 1 → 2; Kolmo2D: 20–40 % per stage), but that reduction does not transfer: val nRMSE of F_k is flat (Gray-Scott) or worse (ks1d, Kolmo2D) after each added stage. I.e. the later stages mostly fit the 512-sample training residual (noise/overfit), not generalisable structure. The dominant cost is the budget split: a single stage 1 trained for 1/K of the time at width 5–7 is already much worse than the baseline on ks1d (0.073 for K=2 stage 1, 0.105 for K=4 stage 1 vs 0.060), and later stages cannot recover it.

### Timing / drift check
Baseline ks1d seed 0 re-run (scratch out_dir, not in results.tsv): steps_per_s 18.6 vs 16.3 / 17.6 / 17.4 in the old runs (seeds 0/1/2), metric 0.0600 vs 0.0606. The machine is not slower (if anything ~5–10 % faster); gb reweight_only runs showed 18.0 steps/s. gboost steps/s are higher (24–28 on ks1d, 6–8 on 2D) because the stages are narrower.

### Verdict
- GB-style reweighting is live (D spread 2–5×, Lbar 0.43–0.83) but, like AdaBoost.R2 on Gray-Scott, it is neutral: reweight_only(gb) ties baseline on all three datasets, and pcgb(gb) ties pcgb(adaboost), still 1.3–9× worse than baseline. The inertness of the AdaBoost gate was not the reason the weighting did nothing.
- Stage-wise functional boosting at matched params and matched wall-clock loses to the single network: K=2 is 1.4× worse on ks1d (significant) and ties on the 2D sets; K=4 is worse still (2.2× on ks1d, 1.1× on Gray-Scott, unstable on Kolmo2D). Later stages fit train residual but add no val skill.
- Nothing here beats stochdepth on Kolmo2D (0.530) or Gray-Scott (0.664). Best-config recommendation is unchanged: plain baseline for KS1D, stochastic depth where data-limited.
- Caveat: one config per method (lam=0.3, clamp 5; ν=1, equal time per stage); n=512 val only, no test split evaluated.

ResNet redo (isotropic ConvNeXt ResNet, same five arms + GB variants, own results.tsv): see resnet/notes.md
