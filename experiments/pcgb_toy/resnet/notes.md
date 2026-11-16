# ResNet redo of the PCGB toy study — lab notebook

Same study as `../notes.md` (five arms, 12-min wall-clock budget, 3 seeds, val for selection, test only for the final table) with the
ConvNeXt U-Net replaced by an isotropic ConvNeXt ResNet (`train.py --arch resnet`). All results live in `resnet/` (`results.tsv`, `runs/`
git-ignored, `phase0/`, `figs/`, `final_table.md`); jobs in `jobs/resnet_*.txt`; tables/figures from `resnet_report.py`.
Code commit `24d0636` (all 183 runs were made with it; the later commits only add results/analysis).

## Summary (Q1–4 for the ResNet; 3 seeds; test table below)

1. **Premise: gradient half true, ensemble half false on KS1D, graceful on the rest.** Gradient is strongly front-loaded: group 0 (blocks 0–2, nearest the stem) gets 55–76% of the grad energy, group 3 gets 4–8%, last block ≈1–2% (all 4 datasets). KS1D is a cliff (any single block ×14–32, any group-mask ×24–44; remove-1-random already ×16): co-adapted, not an ensemble. Kolmo2D/denoise degrade gracefully (single block ×1.4–1.8 / ×1.1–1.6; random-k removal rises smoothly, denoise ×1.2→×3.7 for k=1→11, Veit-like); Gray-Scott lesions are ≈×1.0 because the net is barely better than persistence.
2. **PCGB loses at matched compute, harder than for the U-Net on KS1D**: test 1.23 vs baseline 0.045 (×27; U-Net ×9), Kolmo2D 1.02 vs 0.561, denoise 0.271 vs 0.150; only Gray-Scott ties (0.651 vs 0.647). Argmax picks mask 0 (the stem→head identity net) in 68–72% of KS1D rounds.
3. **Neither half helps**: mask_only ≈ pcgb (×30 worse than baseline on KS1D 512); AdaBoost reweight_only is neutral on KS1D/denoise/Gray-Scott and slightly worse on Kolmo2D (0.594 vs 0.561, >2σ). GB reweighting, gboost K=2/4, mask_rule/norepeat/exclude_all_drop: nothing beats baseline.
4. **No small-n or physics specificity**: PCGB is worse than baseline in every column at n=128 too (KS1D ×28, Kolmo2D ×2.1). The ResNet baseline does not even improve with n (KS1D test 0.050/0.045/0.044; Kolmo2D 0.576/0.561/0.562): at 0.09–0.31× the U-Net's steps/s it is **compute-limited, not data-limited**, so there is no overfitting regime for a regulariser to fix.
5. **Stochastic depth no longer helps** (12-min caveat: partly a compute artefact, see 7) : at p_L=0.2 worse than baseline on 7/8 columns (tie only on Gray-Scott), and p_L=0.5 is worse still (KS1D 0.186 vs 0.116). This is the main U-Net conclusion that does not transfer.
6. **Best config for the ResNet: plain baseline** everywhere (Gray-Scott: stochdepth/baseline tied within noise). The ResNet baseline beats the best U-Net arm on KS1D n=128/512 (0.050 vs 0.240; 0.045 vs 0.068) and Kolmo2D n=128 and denoise, loses on KS1D n=2048 and Kolmo2D n=512/2048 (U-Net stochdepth).

7. **10× budget (120 min, Kolmo2D/Gray-Scott n=512):** the 12-min 2D ResNet was under-trained (~280 steps). At 120 min the baseline drops to 0.358 (Kolmo2D) / 0.421 (Gray-Scott) and beats every U-Net result (U-Net best 0.513 / 0.664). Stochdepth closes most of its gap on Kolmo2D (0.370) but not on Gray-Scott (0.465), and loses on both, so baseline stays best; PCGB gets worse with budget (Kolmo2D 1.17, Gray-Scott diverges).

### Main table — TEST split (10-step rollout nRMSE; denoise 1-step), mean ± std over 3 seeds, ResNet

| arm | ks1d 128 | ks1d 512 | ks1d 2048 | kolmo 128 | kolmo 512 | kolmo 2048 | denoise 512 | grayscott 512 |
|---|---|---|---|---|---|---|---|---|
| baseline | **0.050±.000** | **0.045±.002** | 0.044±.001 | **0.576±.009** | **0.561±.007** | **0.562±.011** | **0.150±.001** | 0.647±.005 |
| stochdepth | 0.131±.002 | 0.132±.003 | 0.136±.004 | 0.638±.008 | 0.631±.011 | 0.626±.013 | 0.164±.003 | **0.643±.004** |
| mask_only | – | 1.364±.166 | 1.052±.326 | – | 1.200±.174 | 1.090±.100 | – | – |
| reweight_only | – | 0.045±.001 | **0.044±.002** | – | 0.594±.003 | 0.595±.005 | 0.153±.001 | 0.661±.009 |
| pcgb (v1) | 1.395±.102 | 1.228±.161 | 1.315±.283 | 1.219±.204 | 1.021±.071 | 1.127±.137 | 0.271±.017 | 0.651±.022 |
| pcgb+warm0.5 | 0.962±.427 | 1.325±.178 | – | 1.117±.054 | – | – | 0.349±.147 | 0.665±.021 |

Bold = lowest mean in the column (a tie within noise where the next arm is within 2σ, e.g. KS1D 2048 baseline/reweight_only, Gray-Scott baseline/stochdepth).
Persistence (test): ks1d 1.15, kolmo 1.04, denoise 0.65, grayscott 0.66. Full table with val means, val→test rank agreement and seeds in `final_table.md`
(best arm agrees val→test in 6/8 columns; the two disagreements are ties: KS1D 2048 baseline vs reweight_only, Gray-Scott baseline vs stochdepth).
pcgb+warm0.5 has no n=2048 column and no Kolmo2D 512 (same cells as the U-Net table). 12 of the 51 stage-b mask_only/pcgb runs are flagged MASK_COLLAPSE (one mask picked >70% of rounds, always mask 0).

## Setup (what differs from the U-Net study)

- **Body**: stem conv (same as U-Net) → 12 ConvNeXt `Block`s at one width C, full resolution, kernel 7 (1D) / 5 (2D), same periodic padding → head conv (init ×0.1) ; output `x + head(h)`. No down/upsampling, no long skips.
- **Width**: `--res_width 0` = auto, the C whose param count is closest to the U-Net(width 10) on that dataset: C=48 (ks1d, 230k vs 235k), C=50 (kolmo2d/denoise/grayscott2d, 260–261k vs 263k); ratios 0.98–0.99. Printed at start of each run and stored as `model_width` in args.json.
- **Paths**: 4 contiguous groups of 3 blocks. `skip_mask` bit g = 1 keeps group g (0 → all 3 residual branches dropped → identity); mask 15 = full net, mask 0 = stem→head. `block_mask` has 12 entries; branch scale = group × block. The 16-candidate adversarial masking and `--mask_target blocks/both` work unchanged (13 / 28 candidates). Only `drop_mode=zero` is allowed (clear error otherwise).
- **stochdepth**: p_l = `--sd_block_p_res`·(l+1)/12, p_L=0.2, 1/(1−p) rescale, no separate skip drop.
- **gboost**: stages are ResNets, width reduced to match total params (K=4: C=24, 239–261k; K=2: C=34–35, 235–264k).
- **Grad norms**: per block, and the "skip" norms are per-group branch norms (‖∂L/∂branch output‖ summed over the group's 3 blocks).
- Everything else (data, budget, evals every 2 min charged to the budget, AdamW, cosine LR, rounds, keep rule Δ>2× larger seed std) as in the U-Net study. U-Net path verified bit-identical (fake clock: baseline, stochdepth, pcgb log + model.pt equal).
- **Cost (important)**: the isotropic full-resolution body is far more expensive per step than the U-Net at equal params (see below), and under a fixed 12-min wall clock gets 3–10× fewer optimisation steps (≈3.9k steps on KS1D, ≈280–320 steps on the 2D sets, i.e. ~20 epochs at n=512). A val eval costs ≈10 s on 2D (U-Net 0.8 s) and PCGB scoring (16 candidates × 128 samples) takes longer than a 24-s round, so 2D pcgb completes only 20–23 of 29 rounds (on Kolmo2D n=512 about 12 train steps per round).

## Calibration + Phase 0 (baselines n=512, seeds 0–2; val)

| dataset | n=512 baseline val | steps/s ResNet | steps/s U-Net | ratio | U-Net val | persistence |
|---|---|---|---|---|---|---|
| ks1d | 0.039 ± 0.002 | 5.33 | 17.1 | 0.31 | 0.060 ± 0.004 | 1.14 |
| kolmo2d | 0.564 ± 0.011 | 0.40 | 4.2 | 0.10 | 0.741 ± 0.028 | 1.03 |
| denoise | 0.151 ± 0.001 | 0.40 | 4.5 | 0.09 | 0.217 ± 0.004 | 0.65 |
| grayscott2d | 0.692 ± 0.056 | 0.40 | 4.0 | 0.10 | 0.682 ± 0.037 | 0.79 |

Seed-noise bar (val std over 3 seeds, n=512): ks1d 0.002, kolmo2d 0.011, denoise 0.001, grayscott2d 0.056 (Gray-Scott is noisy: 0.645–0.754) → keep threshold 2× larger std. At n=128/2048 (stage b): ks1d 0.045±0.004 / 0.039±0.002, kolmo2d 0.576±0.014 / 0.563±0.012.
**Steps/s: ResNet 5.3 (ks1d) and 0.40 (2D) vs U-Net 17.1 and 4.0–4.5**, i.e. 0.31× and 0.09–0.10×. Despite 3–10× fewer steps the ResNet baseline is *better* than the U-Net baseline on val for ks1d (0.039 vs 0.060), kolmo2d (0.564 vs 0.741) and denoise (0.151 vs 0.217), and equal on Gray-Scott (0.692±0.056 vs 0.682±0.037). The U-Net's Kolmo2D/denoise val curves peaked early then overfit (so the best arm there was a regulariser). The ResNet's val curves (seed 1) are still falling at the end on ks1d (0.049→0.046→0.042) and kolmo2d (0.593→0.575) and Gray-Scott (0.72→0.67); denoise has plateaued (0.152→0.151).

### Phase 0 premise checks (seed-0 baselines, `phase0/*.json`, `figs/phase0_grad.png`, `figs/phase0_lesion.png`)

Gradient (per-param energy share, late phase; grad norms logged every 50 steps → only ~6 points on 2D sets):

| | block 0 | block 1 | block 2 | … block 11 | group 0 / 1 / 2 / 3 (energy share) |
|---|---|---|---|---|---|
| ks1d | 0.32 | 0.20 | 0.14 | 0.019 | 0.67 / 0.18 / 0.08 / 0.07 |
| kolmo2d | 0.24 | 0.21 | 0.16 | 0.013 | 0.61 / 0.24 / 0.11 / 0.04 |
| denoise | 0.24 | 0.17 | 0.13 | 0.025 | 0.55 / 0.24 / 0.13 / 0.08 |
| grayscott2d | 0.38 | 0.23 | 0.15 | 0.014 | 0.76 / 0.14 / 0.06 / 0.04 |

Decay is monotone over blocks and stable through training. The residual-branch (per-group) norms say the same (ks1d 0.63 / 0.22 / 0.11 / 0.04). In an isotropic resnet "deep" means
"far from the stem": those blocks get ≤2% of per-param gradient even though the identity path gives them a direct route from the head.

Lesions (val metric relative to the unmasked net):

| | any single block removed | group masks (15 non-full) | random-k blocks removed (median, k=1…11) |
|---|---|---|---|
| ks1d | ×14–32 | ×24–44 (mask 0 ×35) | ×16 at k=1, ×24–35 for k≥2 (flat) |
| kolmo2d | ×1.4–1.8 | ×1.7–2.5 (mask 0 ×1.7 = 0.95, near persistence 1.03) | ×1.5→×2.2, then flat |
| denoise | ×1.07–1.6 | ×1.4–4.0 (mask 0 ×4.0) | ×1.2, 1.3, 1.5, … ×3.7 (smooth) |
| grayscott2d | ≈×1.0 (block 1 ×1.3) | ≈×1.0–1.1 | ≈×1.0–1.1 |

**Q1 (ResNet)**: yes for gradient (front-loaded, last group 4–8%); no graceful-degradation story on KS1D (cliff from the first removed block, like the U-Net), but a Veit-like graceful curve on Kolmo2D/denoise. The Gray-Scott net is essentially persistence-level, so its lesions are uninformative. Path-length curve for ks1d (k of 12 blocks kept, median loss): 0.241 (k=0) → 0.171 (k=4) → 0.076 (k=7) → 0.024 (k=10) → 0.008 (k=11) → 0 (k=12): loss falls smoothly with kept blocks but only the full net is accurate.

## Phase 1 — main table (val, 12 min, mean ± std over 3 seeds)

| arm | ks1d 128 | ks1d 512 | ks1d 2048 | kolmo2d 128 | kolmo2d 512 | kolmo2d 2048 | denoise 512 | grayscott2d 512 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.045 ± 0.004 | 0.039 ± 0.002 | 0.039 ± 0.002 | 0.576 ± 0.014 | 0.564 ± 0.011 | 0.563 ± 0.012 | 0.151 ± 0.001 | 0.692 ± 0.056 |
| stochdepth | 0.115 ± 0.006 | 0.116 ± 0.007 | 0.118 ± 0.007 | 0.633 ± 0.009 | 0.628 ± 0.009 | 0.624 ± 0.013 | 0.165 ± 0.003 | 0.724 ± 0.025 |
| mask_only | – | 1.401 ± 0.176 | 1.062 ± 0.327 | – | 1.192 ± 0.167 | 1.084 ± 0.103 | – | – |
| reweight_only | – | 0.040 ± 0.002 | 0.039 ± 0.003 | – | 0.599 ± 0.008 | 0.599 ± 0.009 | 0.153 ± 0.001 | 0.729 ± 0.029 |
| pcgb | 1.417 ± 0.106 | 1.243 ± 0.153 | 1.343 ± 0.281 | 1.214 ± 0.202 | 1.004 ± 0.076 | 1.118 ± 0.135 | 0.271 ± 0.018 | 0.783 ± 0.020 |
| pcgb+warm0.5 | 0.968 ± 0.420 | 1.344 ± 0.172 | – | 1.113 ± 0.070 | – | – | 0.350 ± 0.147 | 0.781 ± 0.009 |

Keep rule (Δ mean > 2x larger seed std) vs baseline, per column (+ = arm beats baseline, - = significantly worse, 0 = tie):
- stochdepth: ks1d128:- ks1d512:- ks1d2048:- kolmo2d128:- kolmo2d512:- kolmo2d2048:- denoise512:- grayscott2d512:0
- mask_only: ks1d128:. ks1d512:- ks1d2048:- kolmo2d128:. kolmo2d512:- kolmo2d2048:- denoise512:. grayscott2d512:.
- reweight_only: ks1d128:. ks1d512:0 ks1d2048:0 kolmo2d128:. kolmo2d512:- kolmo2d2048:- denoise512:- grayscott2d512:0
- pcgb: ks1d128:- ks1d512:- ks1d2048:- kolmo2d128:- kolmo2d512:- kolmo2d2048:- denoise512:- grayscott2d512:0
- pcgb+warm0.5: ks1d128:- ks1d512:- ks1d2048:. kolmo2d128:- kolmo2d512:. kolmo2d2048:. denoise512:0 grayscott2d512:0

- The n=512 baselines are the stage-a runs (same config, seeds 0–2; seed 0 additionally logs grad norms, no effect on training).
- Only a few cells differ in kind from the U-Net: **stochdepth is significantly worse than baseline in 7/8 columns**, PCGB is worse in 7/8 (tie on Gray-Scott), reweight_only is worse on Kolmo2D 512/2048 (Δ≈0.035, >2σ).
- Mask behaviour: argmax picks mask 0 (identity) 68–72% of rounds on KS1D (m15 only 6%), 41–59% on Kolmo2D, 68% denoise, 11% Gray-Scott. Because mask-0 steps are ~40× cheaper, masked runs do 150k steps on KS1D versus 3.9k for the baseline — still far worse.
- AdaBoost.R2 reweighting: L̄ median 0.75–0.85 (ks1d), 0.71–0.79 (kolmo2d), 0.85–0.87 (denoise) → β=1 (inert); Gray-Scott L̄ 0.37–0.38, β down to 0.45 (live) and still neutral, as for the U-Net.

## Gradient boosting variants (n=512, val, 3 seeds; jobs/resnet_c_gb.txt)

| arm | ks1d | kolmo2d | grayscott2d |
|---|---|---|---|
| baseline | 0.039 ± 0.002 | 0.564 ± 0.011 | 0.692 ± 0.056 |
| stochdepth | 0.116 ± 0.007 | 0.628 ± 0.009 | 0.724 ± 0.025 |
| reweight_only (adaboost) | 0.040 ± 0.002 | 0.599 ± 0.008 | 0.729 ± 0.029 |
| reweight_only (gb) | 0.043 ± 0.002 | 0.612 ± 0.018 | 0.741 ± 0.045 |
| pcgb (adaboost) | 1.243 ± 0.153 | 1.004 ± 0.076 | 0.783 ± 0.020 |
| pcgb (gb) | 1.299 ± 0.283 | 1.036 ± 0.086 | 0.759 ± 0.016 |
| gboost K=2 | 0.063 ± 0.007 | 0.638 ± 0.002 | 0.732 ± 0.027 |
| gboost K=4 | 0.061 ± 0.014 | 0.707 ± 0.006 | 0.755 ± 0.001 |

Wins (Δ>2σ) vs baseline / stochdepth, new configs:
- reweight_only (gb) ks1d: vs baseline tie; vs stochdepth WIN
- reweight_only (gb) kolmo2d: vs baseline worse; vs stochdepth tie
- reweight_only (gb) grayscott2d: vs baseline tie; vs stochdepth tie
- pcgb (gb) ks1d: vs baseline worse; vs stochdepth worse
- pcgb (gb) kolmo2d: vs baseline worse; vs stochdepth worse
- pcgb (gb) grayscott2d: vs baseline tie; vs stochdepth tie
- gboost K=2 ks1d: vs baseline worse; vs stochdepth WIN
- gboost K=2 kolmo2d: vs baseline worse; vs stochdepth tie
- gboost K=2 grayscott2d: vs baseline tie; vs stochdepth tie
- gboost K=4 ks1d: vs baseline tie; vs stochdepth WIN
- gboost K=4 kolmo2d: vs baseline worse; vs stochdepth worse
- gboost K=4 grayscott2d: vs baseline tie; vs stochdepth tie

(In the "wins" list a WIN over stochdepth on ks1d only reflects that stochdepth is itself bad there; vs baseline nothing wins.) Weighted/GB reweighting is neutral-to-worse
(reweight_only gb worse than baseline on Kolmo2D 0.612 vs 0.564). pcgb with gb weights ties adaboost-weighted pcgb everywhere. gboost: K=2 is ×1.6 worse on ks1d and ×1.13 on Kolmo2D (significant), K=4 ties ks1d (noisy, 0.061±0.014; later stages do help here: stage-1 val 0.10 → 0.046–0.067 in 2 of 3 seeds, unlike the U-Net) but is ×1.25 worse on Kolmo2D; Gray-Scott ties. Per-stage train losses barely drop on 2D (0.074→0.072 Kolmo K=4) because each stage gets only 40–60 steps.

## Phase 2 screen (pcgb v1 + one flag, ks1d and kolmo2d n=512, val, 3 seeds; jobs/resnet_d_screen.txt)

Variants: from the U-Net Phase-2 log, random masks, blocks target and exclude-all-drop (the three that moved something there), plus `--norepeat 2` (not run in the U-Net study; directly targets the mask-0 oscillation). warm_frac=0.5 is in stage b (ks1d 512 only here; Kolmo2D 512 not run, as in the U-Net grid). drop_mode=rescale is undefined for resnet.

| config | ks1d 512 | kolmo2d 512 | verdict vs pcgb v1 (ks1d / kolmo2d) | verdict vs baseline |
|---|---|---|---|---|
| pcgb v1 (incumbent) | 1.243 ± 0.153 | 1.004 ± 0.076 | tie / tie | worse / worse |
| mask_rule=random | 1.304 ± 0.406 | 1.225 ± 0.156 | tie / tie | worse / worse |
| norepeat=2 | 0.773 ± 0.359 | 1.172 ± 0.012 | tie / worse | worse / worse |
| mask_target=blocks | 0.459 ± 0.224 | 0.931 ± 0.024 | keep / tie | tie / worse |
| mask_exclude_all_drop | 1.112 ± 0.410 | 1.228 ± 0.268 | tie / tie | worse / worse |
| pcgb + warm_frac=0.5 (stage b) | 1.344 ± 0.172 | – | tie / . | worse / . |
| stochdepth p_L=0.2 (default) | 0.116 ± 0.007 | 0.628 ± 0.009 | keep / keep | worse / worse |
| stochdepth p_L=0.5 | 0.186 ± 0.011 | 0.652 ± 0.008 | keep / keep | worse / worse |
| baseline | 0.039 ± 0.002 | 0.564 ± 0.011 | keep / keep | tie / tie |

- Only `mask_target=blocks` clears the keep rule against pcgb v1 on KS1D (0.459±0.224 vs 1.243±0.153), but it is still ×12 worse than baseline; the "tie vs baseline" in that row is an artefact of its huge seed std (0.22). On Kolmo2D nothing beats pcgb v1 (norepeat is worse), and nothing approaches baseline (0.564).
- **stochdepth sensitivity**: p_L=0.5 is worse than the default p_L=0.2 on both (ks1d 0.186 vs 0.116, kolmo2d 0.652 vs 0.628) and both are worse than baseline. Drop rate is monotone harmful here; smaller p_L (not run) would converge to baseline. The failure of stochdepth is therefore not a mis-set default.

## Horizon (50-step rollouts, val, n=512 models; `figs/horizon_*_n512.png`, `horizon.log`)

First step with nRMSE>1 (mean over seeds):

| | baseline | stochdepth | pcgb | mask_only | reweight_only |
|---|---|---|---|---|---|
| ks1d n512 | **28.0** | 20.3 | 4.3 | 3.7 | 27.7 |
| kolmo2d n512 | **12.3** | 11.0 | 5.3 | 4.0 | 11.3 |

The step-10 ranking holds at long horizons. ResNet baseline has a longer horizon than the U-Net baseline (ks1d 28 vs 24.7; kolmo2d 12.3 vs 8.3), and **stochdepth's horizon lead on Kolmo2D (U-Net 15.3 vs 8.3) vanishes** (11.0 vs 12.3). On ks1d pcgb/mask_only/stochdepth rollouts explode (33% of ICs flagged |x|>20 or non-finite at step 50; baseline/reweight_only 0%); on Kolmo2D only mask_only grows energy (std ratio 1.9 at step 50; pcgb 1.0).

## Long-budget check (10× = 120 min/run; jobs/resnet_long10x.txt, `resnet_long10x.py`, `figs/budget10x_curves.png`)

Why: in 12 min the 2D ResNet gets only ~280 optimiser steps (0.4 steps/s), ~1/10 of the U-Net, so the 2D conclusions above may be
early-training artefacts. 120 min gives ~3k steps (≈ the U-Net's 12-min step count). Kolmo2D and Gray-Scott, n=512, 3 seeds,
12 intermediate evals; pcgb with `--n_rounds 100` (ResNet scoring is slow; 99/100 rounds completed). Gray-Scott seed 2 was re-run
after a container restart killed the first attempt minutes before the end (rows from commit d0078ef; same code).

### Kolmo2D n=512 (val, mean ± std over 3 seeds)

| arm | 12 min | 120 min | steps (120 min) |
|---|---|---|---|
| baseline | 0.564 ± 0.011 | **0.358 ± 0.003** | ~3.0k |
| stochdepth (p_L=0.2) | 0.628 ± 0.009 | 0.370 ± 0.001 | ~3.4k |
| pcgb (100 rounds) | 1.004 ± 0.076 | 1.166 ± 0.112 | ~420k (mask-0 steps are cheap) |

### Gray-Scott n=512 (val, mean ± std over 3 seeds)

| arm | 12 min | 120 min | steps (120 min) |
|---|---|---|---|
| baseline | 0.692 ± 0.056 | **0.421 ± 0.000** | ~3.0k |
| stochdepth (p_L=0.2) | 0.724 ± 0.025 | 0.465 ± 0.008 | ~3.3k |
| pcgb (100 rounds) | 0.783 ± 0.020 | 10.8 ± 11.9 (27.6 / 3.50 / 1.29: diverges) | ~220k |

**Curves.** Baseline and stochdepth improve monotonically and are nearly flat over the last ~20–30 min (baseline Kolmo2D
0.353–0.360, Gray-Scott 0.421; best eval = final eval ±0.003); no overfitting. PCGB's best eval is early (Kolmo2D 0.82–0.94 at
20–60 min, Gray-Scott 0.68–0.77) and it ends worse. On Kolmo2D it collapses onto mask 0 (identity net) in 84–85% of rounds; on
Gray-Scott it keeps switching (42–51% mask 0, ~70 switches, 15 distinct masks) and the rollout diverges late in training.

**Conclusions.**
- The 12-min 2D ResNet numbers were badly under-trained: baseline improves by 37–39% from 12 to 120 min. The 12-min rows above
  describe early training on the 2D sets, not converged behaviour.
- Stochdepth's failure on the ResNet was partly a compute artefact: the gap to baseline shrinks from 0.064 to 0.012 on Kolmo2D,
  but on Gray-Scott it grows from 0.032 to 0.044. Baseline still wins on both by > 2× the larger seed std, so
  the sign does not flip and the ResNet best config stays **plain baseline**.
- PCGB gets *worse* with more training (Kolmo2D 1.00 → 1.17; Gray-Scott diverges). The more-budget-helps-PCGB effect seen on the
  U-Net (0.92 → 0.80 Kolmo2D) does not hold here.
- **ResNet vs U-Net:** the 120-min ResNet baseline beats every U-Net result on the 2D sets: Kolmo2D 0.358 vs U-Net stochdepth 0.530
  (12 min) / 0.513 (120 min); Gray-Scott 0.421 vs U-Net stochdepth 0.664 (12 min; no U-Net 120-min Gray-Scott run). At roughly
  matched optimiser steps (ResNet 120 min ≈ U-Net 12 min ≈ 3k steps) the ResNet is ~30% better, so the gain is architectural,
  not just more steps. It costs ~10× more wall-clock per step on this CPU.

## Test split
`final_test.py --out_dir resnet`: for every Phase-1 cell, saved models re-evaluated on val (matches the logged metric to 1e-6 for all runs; none skipped) then on test. Table at the top of this file; details in `final_table.md`. No selection used test.

## U-Net vs ResNet

Test, mean±std over 3 seeds (U-Net numbers from `../final_table.md`); ratio = ResNet/U-Net seed means.

| arm | ks1d 128 | ks1d 512 | ks1d 2048 | kolmo2d 128 | kolmo2d 512 | kolmo2d 2048 | denoise 512 | grayscott2d 512 |
|---|---|---|---|---|---|---|---|---|
| baseline (U-Net) | 0.240±0.022 | 0.068±0.001 | 0.032±0.002 | 1.144±0.025 | 0.760±0.029 | 0.543±0.013 | 0.217±0.005 | 0.664±0.008 |
| baseline (ResNet) | 0.050±0.000 | 0.045±0.002 | 0.044±0.001 | 0.576±0.009 | 0.561±0.007 | 0.562±0.011 | 0.150±0.001 | 0.647±0.005 |
| stochdepth (U-Net) | 0.330±0.008 | 0.243±0.011 | 0.255±0.028 | 0.705±0.025 | 0.541±0.007 | 0.509±0.009 | 0.193±0.003 | 0.642±0.023 |
| stochdepth (ResNet) | 0.131±0.002 | 0.132±0.003 | 0.136±0.004 | 0.638±0.008 | 0.631±0.011 | 0.626±0.013 | 0.164±0.003 | 0.643±0.004 |
| mask_only (U-Net) | – | 0.503±0.035 | 0.682±0.058 | – | 0.905±0.114 | 1.082±0.193 | – | – |
| mask_only (ResNet) | – | 1.364±0.166 | 1.052±0.326 | – | 1.200±0.174 | 1.090±0.100 | – | – |
| reweight_only (U-Net) | – | 0.069±0.001 | 0.032±0.002 | – | 0.765±0.033 | 0.543±0.011 | 0.216±0.006 | 0.671±0.007 |
| reweight_only (ResNet) | – | 0.045±0.001 | 0.044±0.002 | – | 0.594±0.003 | 0.595±0.005 | 0.153±0.001 | 0.661±0.009 |
| pcgb (U-Net) | 0.581±0.041 | 0.615±0.065 | 0.579±0.037 | 1.096±0.050 | 0.922±0.095 | 1.018±0.125 | 0.420±0.031 | 0.713±0.021 |
| pcgb (ResNet) | 1.395±0.102 | 1.228±0.161 | 1.315±0.283 | 1.219±0.204 | 1.021±0.071 | 1.127±0.137 | 0.271±0.017 | 0.651±0.022 |
| pcgb_warm (U-Net) | 0.521±0.042 | 0.523±0.124 | – | 1.029±0.040 | – | – | 0.487±0.085 | 0.667±0.021 |
| pcgb_warm (ResNet) | 0.962±0.427 | 1.325±0.178 | – | 1.117±0.054 | – | – | 0.349±0.147 | 0.665±0.021 |

ResNet/U-Net test ratio, seed means:
- baseline: ks1d128:0.21 ks1d512:0.65 ks1d2048:1.39 kolmo2d128:0.50 kolmo2d512:0.74 kolmo2d2048:1.03 denoise512:0.69 grayscott2d512:0.97
- stochdepth: ks1d128:0.40 ks1d512:0.54 ks1d2048:0.54 kolmo2d128:0.90 kolmo2d512:1.17 kolmo2d2048:1.23 denoise512:0.85 grayscott2d512:1.00
- pcgb: ks1d128:2.40 ks1d512:2.00 ks1d2048:2.27 kolmo2d128:1.11 kolmo2d512:1.11 kolmo2d2048:1.11 denoise512:0.65 grayscott2d512:0.91

Best arm per column (test mean): ks1d128: U=baseline R=baseline; ks1d512: U=baseline R=baseline; ks1d2048: U=baseline R=reweight_only; kolmo2d128: U=stochdepth R=baseline; kolmo2d512: U=stochdepth R=baseline; kolmo2d2048: U=stochdepth R=baseline; denoise512: U=stochdepth R=baseline; grayscott2d512: U=stochdepth R=stochdepth

**Transfers**
- PCGB loses everywhere at matched compute, and the damage comes from adversarial masking; reweighting (AdaBoost.R2 or GB) is neutral. mask_only ≈ pcgb, reweight_only ≈ baseline. Gray-Scott is the only column where pcgb ties.
- Masking is *worse* on the ResNet on KS1D (×27 vs ×9 at n=512): the worst candidate is almost always the identity net (mask 0), a degenerate adversary on both architectures.
- KS1D baseline is a co-adapted long-path function (lesion cliff on both); deep/late units get little gradient on both.
- GB variants (reweight, gboost) do not beat baseline on either; no n=128 / physics specificity for PCGB.
- Val→test ranking agreement; ordering baseline ≻ pcgb ≻ everything on the horizon plot.

**Does not transfer**
- **Stochastic depth** is the U-Net's best regulariser on Kolmo2D/denoise/Gray-Scott (−28% at Kolmo n=512) but is worse than plain training on the ResNet in 7/8 columns and ties on Gray-Scott. Reason: the ResNet baseline does not overfit at n=512 in 12 min; it is compute-starved (280–320 steps in 2D, 3.9k on KS1D), so a regulariser only slows it. The best config therefore changes from "baseline for KS1D, stochdepth elsewhere" to "baseline everywhere".
- **Data-limited vs compute-limited regime**: U-Net metrics fall with n (KS1D 0.240→0.068→0.032); ResNet's are flat in n (0.050/0.045/0.044), so the ResNet is better at small n and worse at n=2048 (KS1D baseline ×1.4 vs U-Net baseline; Kolmo2D 2048 ResNet 0.562 vs U-Net best 0.509 = ×1.10).
- **Kolmo2D predictability horizon**: U-Net stochdepth extended it (15.3 vs 8.3 steps); on the ResNet the baseline already has 12.3 and stochdepth does not extend it.
- **Lesion structure**: Kolmo2D/denoise were "deep paths unused / mild" on the U-Net; on the ResNet they show a smooth Veit-like degradation with every block carrying some signal, while the gradient decays monotonically with depth. KS1D is a cliff on both.
- **PCGB on denoise** is less bad on the ResNet (×1.8 vs ×1.9 baseline) and ties on Gray-Scott; on KS1D/Kolmo2D it is worse (×2 on KS1D). gboost later stages help a little on KS1D (not for the U-Net).
- Caveat on all of the above: the ResNet arm is at ~0.1–0.3× the U-Net's optimiser steps under equal wall-clock. **Tested at 10× budget (see Long-budget check):** with ~3k steps the 2D ResNet baseline improves by ~38% and beats every U-Net result; stochdepth mostly catches up on Kolmo2D (0.370 vs 0.358) but still loses, so "stochdepth does not transfer" holds in sign but its 12-min size on Kolmo2D was a compute artefact.
