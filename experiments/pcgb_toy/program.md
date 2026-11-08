# program.md — PCGB on toy systems (autoresearch-style)

(Human-owned spec, copied verbatim from the task request at session start. See the session transcript for the full
original; key sections reproduced below.)

## The question

PCGB ("Path-Cyclic Gradient Boosting", proposed for the Samudra ocean emulator) claims that a residual U-Net is an implicit ensemble of paths, that most paths get too little gradient, and that you can fix this by training in rounds that (a) adversarially pick which skip connections to drop and (b) reweight samples AdaBoost.R2-style. The deployed model is the plain unmasked network.

1. Is the premise true? (Are paths actually under-trained in a baseline?)
2. Does PCGB beat plain training and random stochastic depth at matched compute?
3. Which half matters: the masking, the reweighting, or neither?
4. Is any gain specific to the small-sample, physics-emulation regime (the paper's motivation), or generic?

Negative results are fine. Clean answers matter more than a win.

## PCGB v1 reference implementation

```
θ random; D(i)=1 for all train samples; M = all-keep
for round t = 1..T:
  train K steps with mask M; loss = mean(D[idx] * ||ŷ - y||²)
  on scoring subset (from calibration split):
    r(i) = ||y_i - F(x_i)||          # unmasked net
    S(M') = Σ D(i)·||y_i - F(x_i; M')||²   for all 16 skip masks
  reweight (AdaBoost.R2 variant):
    L_i = r_i / max_j r_j;  L̄ = Σ D_i L_i (D normalized to sum 1 here)
    β = min(L̄/(1-L̄), 1);  D'_i ∝ D_i · β^(1-L_i)
    D ← 0.7·D + 0.3·D';  clamp max/min ≤ 5;  renormalize to mean 1
  M ← argmax_M' S(M')
```
Open choices (default = first): T,K so ~20–40 rounds; keep Adam state (also try reset); D computed on the
training samples themselves via a strided subset each round with stale values for the rest (also try held-out
only); dropped skip = zero the tensor (also try alternatives).

## Phases
- Phase 0: premise checks on a trained baseline (per-skip gradient norms, lesion study over 16 skip masks and each
  residual block removed).
- Phase 1: all 5 arms × datasets × n_train × 3 seeds; mean ± std; log mask sequence; flag mask collapse (>70%).
- Phase 2: loop of one focused change to train.py, 3 seeds on primary dataset at n=512, keep only if mean beats best
  by > 2× seed std, else revert; generalization re-check every ~10 kept changes.

## Rules
- Never edit prepare.py or the eval after setup. Never tune on test.
- One idea per commit. Keep train.py readable.
- Wall-clock budget includes everything (scoring passes count).
- Watch for degenerate wins: weight entropy < 0.8, mask collapse, NaNs, blown-up rollouts.
- Prefer simplicity.
- When stopped, notes.md opens with a ≤10-line summary answering Q1–4, main table and best config.
