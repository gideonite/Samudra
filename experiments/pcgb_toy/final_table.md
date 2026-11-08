| arm | ks1d n=128 | ks1d n=512 | ks1d n=2048 | kolmo2d n=128 | kolmo2d n=512 | kolmo2d n=2048 | denoise n=512 | grayscott2d n=512 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.240 ± 0.022 (0.205) | 0.068 ± 0.001 (0.060) | 0.032 ± 0.002 (0.025) | 1.144 ± 0.025 (1.140) | 0.760 ± 0.029 (0.741) | 0.543 ± 0.013 (0.528) | 0.217 ± 0.005 (0.217) | 0.664 ± 0.008 (0.682) |
| stochdepth | 0.330 ± 0.008 (0.311) | 0.243 ± 0.011 (0.218) | 0.255 ± 0.028 (0.233) | 0.705 ± 0.025 (0.692) | 0.541 ± 0.007 (0.530) | 0.509 ± 0.009 (0.496) | 0.193 ± 0.003 (0.193) | 0.642 ± 0.023 (0.664) |
| mask_only | – | 0.503 ± 0.035 (0.506) | 0.682 ± 0.058 (0.691) | – | 0.905 ± 0.114 (0.905) | 1.082 ± 0.193 (1.077) | – | – |
| reweight_only | – | 0.069 ± 0.001 (0.061) | 0.032 ± 0.002 (0.026) | – | 0.765 ± 0.033 (0.743) | 0.543 ± 0.011 (0.527) | 0.216 ± 0.006 (0.217) | 0.671 ± 0.007 (0.686) |
| pcgb | 0.581 ± 0.041 (0.574) | 0.615 ± 0.065 (0.619) | 0.579 ± 0.037 (0.587) | 1.096 ± 0.050 (1.096) | 0.922 ± 0.095 (0.924) | 1.018 ± 0.125 (1.008) | 0.420 ± 0.031 (0.417) | 0.713 ± 0.021 (0.742) |
| pcgb_warm | 0.521 ± 0.042 (0.520) | 0.523 ± 0.124 (0.516) | – | 1.029 ± 0.040 (1.024) | – | – | 0.487 ± 0.085 (0.482) | 0.667 ± 0.021 (0.695) |

Val→test rank agreement per column (over arms, using seed-mean metrics; lower = better):

- ks1d n=128: Spearman 1.00, Kendall 1.00, same best arm: True (val baseline, test baseline); val order baseline < stochdepth < pcgb_warm < pcgb; test order baseline < stochdepth < pcgb_warm < pcgb
- ks1d n=512: Spearman 1.00, Kendall 1.00, same best arm: True (val baseline, test baseline); val order baseline < reweight_only < stochdepth < mask_only < pcgb_warm < pcgb; test order baseline < reweight_only < stochdepth < mask_only < pcgb_warm < pcgb
- ks1d n=2048: Spearman 1.00, Kendall 1.00, same best arm: True (val baseline, test baseline); val order baseline < reweight_only < stochdepth < pcgb < mask_only; test order baseline < reweight_only < stochdepth < pcgb < mask_only
- kolmo2d n=128: Spearman 1.00, Kendall 1.00, same best arm: True (val stochdepth, test stochdepth); val order stochdepth < pcgb_warm < pcgb < baseline; test order stochdepth < pcgb_warm < pcgb < baseline
- kolmo2d n=512: Spearman 1.00, Kendall 1.00, same best arm: True (val stochdepth, test stochdepth); val order stochdepth < baseline < reweight_only < mask_only < pcgb; test order stochdepth < baseline < reweight_only < mask_only < pcgb
- kolmo2d n=2048: Spearman 0.90, Kendall 0.80, same best arm: True (val stochdepth, test stochdepth); val order stochdepth < reweight_only < baseline < pcgb < mask_only; test order stochdepth < baseline < reweight_only < pcgb < mask_only
- denoise n=512: Spearman 0.90, Kendall 0.80, same best arm: True (val stochdepth, test stochdepth); val order stochdepth < baseline < reweight_only < pcgb < pcgb_warm; test order stochdepth < reweight_only < baseline < pcgb < pcgb_warm
- grayscott2d n=512: Spearman 0.90, Kendall 0.80, same best arm: True (val stochdepth, test stochdepth); val order stochdepth < baseline < reweight_only < pcgb_warm < pcgb; test order stochdepth < baseline < pcgb_warm < reweight_only < pcgb

Cells: test mean ± sample std over seeds (val mean). k = number of seeds when not 3. Metric = 10-step rollout nRMSE (denoise: one-step nRMSE).

Non-finite test rollouts: none

Skipped/excluded:
- kolmo2d_pcgb_n512_s0_83413c5_aa4499: no model.pt / no final row (unfinished)
- kolmo2d_pcgb_n512_s1_83413c5_aa4499: no model.pt / no final row (unfinished)
- kolmo2d_pcgb_n512_s2_83413c5_aa4499: no model.pt / no final row (unfinished)
