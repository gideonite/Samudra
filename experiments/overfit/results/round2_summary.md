<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Overfit run summary

Ratios are model RMSE / persistence RMSE on the memorized train sample (heldout column: heldout median ratio).

## Final recorded step per run

| run | step | train_s/step | median | max (channel) | frac<1 | med uo | med vo | med thetao | med so | med zos | heldout median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F_dynNone_zero | 702 | 30.2 | 0.596 | 4.71 (so_16) | 0.84 | 0.423 | 0.381 | 0.805 | 0.976 | 0.611 | 1.17 |
| G_delta_zero | 881 | 23.9 | 0.742 | 4.19 (so_15) | 0.84 | 0.626 | 0.599 | 0.914 | 1 | 0.939 | 1.13 |
| H_dyn20_zero_lr1e-4 | 694 | 30.5 | 0.22 | 0.949 (so_17) | 1.00 | 0.0478 | 0.051 | 0.466 | 0.659 | 0.22 | 1.24 |
| I_dynNone_zero_lr1e-4 | 592 | 29.8 | 0.17 | 0.474 (so_5) | 1.00 | 0.132 | 0.145 | 0.205 | 0.299 | 0.113 | 1.25 |

## Matched step (575)

| run | step | train_s/step | median | max (channel) | frac<1 | med uo | med vo | med thetao | med so | med zos | heldout median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F_dynNone_zero | 575 | 30.7 | 0.236 | 0.57 (so_18) | 1.00 | 0.163 | 0.153 | 0.287 | 0.412 | 0.167 | 1.19 |
| G_delta_zero | 575 | 24.8 | 0.368 | 1.06 (so_18) | 0.99 | 0.325 | 0.309 | 0.421 | 0.504 | 0.371 | 1.16 |
| H_dyn20_zero_lr1e-4 | 575 | 31.0 | 0.232 | 1.33 (so_18) | 0.97 | 0.0541 | 0.056 | 0.491 | 0.685 | 0.232 | 1.24 |
| I_dynNone_zero_lr1e-4 | 575 | 29.9 | 0.183 | 0.482 (so_5) | 1.00 | 0.132 | 0.152 | 0.213 | 0.318 | 0.129 | 1.25 |
