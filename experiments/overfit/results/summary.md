<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Overfit run summary

Ratios are model RMSE / persistence RMSE on the memorized train sample (heldout column: heldout median ratio).

## Final recorded step per run

| run | step | train_s/step | median | max (channel) | frac<1 | med uo | med vo | med thetao | med so | med zos | heldout median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_res_mse_bf16 | 889 | 23.7 | 0.626 | 11.2 (so_14) | 0.53 | 0.131 | 0.1 | 1.21 | 1.78 | 0.626 | 1.49 |
| B_res_mse_fp32 | 283 | 60.6 | 1.34 | 23.5 (so_14) | 0.49 | 0.373 | 0.298 | 2.42 | 3.62 | 1.34 | 1.66 |
| C_full_mse_bf16 | 909 | 23.2 | 0.927 | 16.5 (so_18) | 0.51 | 0.171 | 0.132 | 1.82 | 2.88 | 0.927 | 2.01 |
| D_res_dynamic_bf16 | 719 | 29.5 | 0.146 | 2.19 (so_18) | 0.94 | 0.0393 | 0.0419 | 0.3 | 0.422 | 0.146 | 1.23 |

## Matched step (275)

| run | step | train_s/step | median | max (channel) | frac<1 | med uo | med vo | med thetao | med so | med zos | heldout median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A_res_mse_bf16 | 275 | 22.9 | 1.35 | 23.5 (so_14) | 0.49 | 0.384 | 0.319 | 2.44 | 3.58 | 1.35 | 1.65 |
| B_res_mse_fp32 | 275 | 60.5 | 1.39 | 25 (so_14) | 0.49 | 0.406 | 0.333 | 2.6 | 3.61 | 1.39 | 1.63 |
| C_full_mse_bf16 | 275 | 22.4 | 1.93 | 34.4 (so_18) | 0.49 | 0.436 | 0.363 | 3.82 | 6.02 | 1.93 | 2.39 |
| D_res_dynamic_bf16 | 275 | 28.7 | 0.318 | 5.22 (so_18) | 0.81 | 0.11 | 0.115 | 0.604 | 0.884 | 0.318 | 1.26 |
