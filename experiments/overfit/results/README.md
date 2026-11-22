<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Can Samudra overfit one sample? (1°, CPU)

Experiment record, not intended to merge. Follows up the one-step report in
m2lines/Samudra#980 (`onestep_figures_onedeg.md`), which found Samudra 2's
temperature and salinity error sits at a ~2–4%-of-a-spread "precision floor"
and loses to persistence below the surface.

**Question.** If the architecture is asked to memorize a single 1° sample, can
it beat persistence on every channel? If it can't even do that, the floor is
about the model or its numerics. If it can, the floor in the paper checkpoint
is about the training objective and task.

## Setup

- **Model:** the paper Samudra architecture (`samudra_om4/model.yaml`: ConvNeXt
  U-Net, widths 280–520, 84M params). 2 input steps, 2 output steps, activation
  checkpointing on.
- **Data:** public v2026-09 `om4_onedeg` store, 40 steps cloned (1975-01-03 to
  1975-07-17). Forcing is `tau_hfds`, because `hfds_anomalies` is identically
  zero on a 40-step clone.
- **Training:** one memorized training sample (batch 1). Adam at lr 6e-4,
  constant. Grad clip 1.0. One CPU core per arm, 4 arms in parallel, each with a
  6-hour wall-clock budget.
- **Score:** per-channel masked RMSE in normalized units, as
  model RMSE ÷ persistence RMSE, where persistence is x[t] for both outputs.
  Below 1 beats doing nothing. Scored on the memorized sample, with one
  held-out sample for reference.
- **Code:** `experiments/overfit/overfit.py` and `analyze.py`; config
  `experiments/overfit/configs/overfit_onedeg.yaml`.

| Arm | Change |
|---|---|
| A | `pred_residuals: true`, plain MSE, bf16 |
| B | A in fp32 (no bf16 autocast) |
| C | `pred_residuals: false` (paper setting), plain MSE, bf16 |
| D | `pred_residuals: true`, dynamic loss (paper loss: per-channel 1/MSE weights, limit 20), bf16 |

## Results

Final step of each arm (ratio = model RMSE ÷ persistence RMSE on the memorized
sample):

| Arm | Steps | Median | Channels beating persistence | Worst channel | Median thetao | Median so | Held-out median |
|---|---:|---:|---:|---|---:|---:|---:|
| A res, MSE, bf16 | 889 | 0.63 | 53% (41/77) | 11.2 (so_14) | 1.21 | 1.78 | 1.49 |
| B res, MSE, fp32 | 283 | 1.34 | 49% | 23.5 (so_14) | 2.42 | 3.62 | 1.66 |
| C full state, MSE, bf16 | 909 | 0.93 | 51% | 16.5 (so_18) | 1.82 | 2.88 | 2.01 |
| D res, dynamic, bf16 | 719 | **0.15** | **94%** (72/77) | 2.19 (so_18) | **0.30** | **0.42** | 1.23 |

At step 275, the last step all arms reached, A, B, C and D have medians of
1.35, 1.39, 1.93 and 0.32.

![Median ratio vs step](ratio_vs_step.png)
![Fraction of channels beating persistence](frac_beat_vs_step.png)
![Ratio by depth at final step](ratio_by_depth_final.png)

### Findings

1. **No arm memorized one sample within ~6 CPU-hours.** The best arm (D) still
   loses to persistence on 5 channels: `so_18`, `thetao_18`, `so_14`,
   `so_15` and `so_16`: salinity at 2400–4000 m and temperature and salinity at 6000 m. All arms are
   still improving at the end. D's worst channel went 5.3 → 4.2 → 3.0 at steps
   300, 500 and 700. So this run doesn't show a hard floor; it shows that
   fitting the deep ocean is *slow*.
2. **The loss weighting matters most.** Under plain MSE (A, C), the model fits
   velocities quickly but for hundreds of steps beats persistence on *only*
   the 38 velocity channels: exactly 49% = 38/77 from step ~75 to ~500. At its
   final step (889), A still loses on all 19 temperature levels and on every
   salinity level except the top two. With the paper's dynamic loss (D),
   temperature and salinity beat persistence down to ~550 m by step 300, and
   everywhere above 2400 m by step 719. The deep channels' true 5-day
   change is tiny in normalized units, so their MSE is negligible unless it is
   reweighted.
3. **Residual prediction helps, as the #980 report proposed.** A (residual)
   beats C (full state) at every matched step: median 0.63 vs 0.93 near
   step 900. Its error on temperature and salinity at 1400 m and deeper is
   ~1.8× lower (median over those channels, at step 875). The gain is
   smaller than the loss-weighting gain.
4. **bf16 is not the bottleneck.** B (fp32) tracks A (bf16) to within a few
   percent at every matched step through 275: medians 1.54 vs 1.54 at step 200
   and 1.39 vs 1.35 at step 275. That is despite bf16's ~0.4% relative
   precision being comparable to the deep 5-day change. This rules out bf16
   numerics as the cause of the precision floor, at least in this regime.
5. **Two setup quirks that work against residual prediction:**
   - With `input_steps=2, output_steps=2`, `pred_residuals` uses same-slot
     assembly. Output t+1 is anchored on x[t-1], not x[t], so the "zero
     correction" baseline is already 1.2× worse than persistence (median), and
     the model must learn a 10-day change for one of its outputs.
   - The decoder isn't zero-initialised, so residual arms start at ~14×
     persistence error instead of at the residual base.
6. **Held-out medians stay above 1 for every arm** (1.2–2.0). This is expected
   when memorizing one sample, and is not a generalization result.

### Implications

- The ablation proposed in #980 (`pred_residuals`, then σ_Δ normalization) is
  pointed at the right knob, but the loss weighting is the bigger lever. Under
  the paper's dynamic loss with a limit of 20, the deepest levels still lag.
  The per-channel σ_Δ ratios exceed 20× (as #980 notes), so the limit is a
  plausible reason those channels get too little weight.
- Worth trying next:
  - D with a larger or no limit, or with weights of 1/(persistence MSE);
  - residual base anchored on x[t] for both output steps;
  - a zero-initialised decoder;
  - a learning-rate decay to push the memorization further.

## Provenance

- Runs: 2026-11-22, 05:48–11:50 UTC. B ran 06:57–11:48 after an OOM kill
  (the first attempt is not reported).
- Hardware: 4 vCPU Xeon (AMX bf16), 16 GB RAM, one core per arm.
- Per-step histories: `history_<arm>.json`; tables are in `summary.md`.
- Regenerate figures and tables with
  `python experiments/overfit/analyze.py <run dirs> --out <dir>`.
