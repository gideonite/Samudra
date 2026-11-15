| arm | ks1d n=128 | ks1d n=512 | ks1d n=2048 | kolmo2d n=128 | kolmo2d n=512 | kolmo2d n=2048 | denoise n=512 | grayscott2d n=512 |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.050 ± 0.000 (0.045) | 0.045 ± 0.002 (0.039) | 0.044 ± 0.001 (0.039) | 0.576 ± 0.009 (0.576) | 0.561 ± 0.007 (0.564) | 0.562 ± 0.011 (0.563) | 0.150 ± 0.001 (0.151) | 0.647 ± 0.005 (0.692) |
| stochdepth | 0.131 ± 0.002 (0.115) | 0.132 ± 0.003 (0.116) | 0.136 ± 0.004 (0.118) | 0.638 ± 0.008 (0.633) | 0.631 ± 0.011 (0.628) | 0.626 ± 0.013 (0.624) | 0.164 ± 0.003 (0.165) | 0.643 ± 0.004 (0.724) |
| mask_only | – | 1.364 ± 0.166 (1.401) | 1.052 ± 0.326 (1.062) | – | 1.200 ± 0.174 (1.192) | 1.090 ± 0.100 (1.084) | – | – |
| reweight_only | – | 0.045 ± 0.001 (0.040) | 0.044 ± 0.002 (0.039) | – | 0.594 ± 0.003 (0.599) | 0.595 ± 0.005 (0.599) | 0.153 ± 0.001 (0.153) | 0.661 ± 0.009 (0.729) |
| pcgb | 1.395 ± 0.102 (1.417) | 1.228 ± 0.161 (1.243) | 1.315 ± 0.283 (1.343) | 1.219 ± 0.204 (1.214) | 1.021 ± 0.071 (1.004) | 1.127 ± 0.137 (1.118) | 0.271 ± 0.017 (0.271) | 0.651 ± 0.022 (0.783) |
| pcgb_warm | 0.962 ± 0.427 (0.968) | 1.325 ± 0.178 (1.344) | – | 1.117 ± 0.054 (1.113) | – | – | 0.349 ± 0.147 (0.350) | 0.665 ± 0.021 (0.781) |

Val→test rank agreement per column (over arms, using seed-mean metrics; lower = better):

- ks1d n=128: Spearman 1.00, Kendall 1.00, same best arm: True (val baseline, test baseline); val order baseline < stochdepth < pcgb_warm < pcgb; test order baseline < stochdepth < pcgb_warm < pcgb
- ks1d n=512: Spearman 1.00, Kendall 1.00, same best arm: True (val baseline, test baseline); val order baseline < reweight_only < stochdepth < pcgb < pcgb_warm < mask_only; test order baseline < reweight_only < stochdepth < pcgb < pcgb_warm < mask_only
- ks1d n=2048: Spearman 0.90, Kendall 0.80, same best arm: False (val baseline, test reweight_only); val order baseline < reweight_only < stochdepth < mask_only < pcgb; test order reweight_only < baseline < stochdepth < mask_only < pcgb
- kolmo2d n=128: Spearman 1.00, Kendall 1.00, same best arm: True (val baseline, test baseline); val order baseline < stochdepth < pcgb_warm < pcgb; test order baseline < stochdepth < pcgb_warm < pcgb
- kolmo2d n=512: Spearman 1.00, Kendall 1.00, same best arm: True (val baseline, test baseline); val order baseline < reweight_only < stochdepth < pcgb < mask_only; test order baseline < reweight_only < stochdepth < pcgb < mask_only
- kolmo2d n=2048: Spearman 1.00, Kendall 1.00, same best arm: True (val baseline, test baseline); val order baseline < reweight_only < stochdepth < mask_only < pcgb; test order baseline < reweight_only < stochdepth < mask_only < pcgb
- denoise n=512: Spearman 1.00, Kendall 1.00, same best arm: True (val baseline, test baseline); val order baseline < reweight_only < stochdepth < pcgb < pcgb_warm; test order baseline < reweight_only < stochdepth < pcgb < pcgb_warm
- grayscott2d n=512: Spearman 0.60, Kendall 0.40, same best arm: False (val baseline, test stochdepth); val order baseline < stochdepth < reweight_only < pcgb_warm < pcgb; test order stochdepth < baseline < pcgb < reweight_only < pcgb_warm

Cells: test mean ± sample std over seeds (val mean). k = number of seeds when not 3. Metric = 10-step rollout nRMSE (denoise: one-step nRMSE).

Non-finite test rollouts: none

Skipped/excluded:
