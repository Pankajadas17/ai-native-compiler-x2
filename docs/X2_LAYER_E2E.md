# Real layers end to end on the X2 (`run_x2_layer_e2e.py`)

For each matrix layer of the real tiny MobileNetV2 (shapes from your IR), it runs IR layer → pack → two-block schedule →
engine → NumPy:
- **4-bit (project setting):** packed and run in Python with the ASSUMED 4-bit cell rule (`count = Σ pwm × level`), then
  compared exactly with NumPy. That rule is the `cell_compute_4bit` entry in `chip/x2_facts.py`, still `missing`,
  because the chip's X2 model has no 4-bit cells.
- **1-bit (what the chip's X2 model computes):** simulated in Icarus on `x2_engine2.v`, all tiles on one block and then
  alternating over both blocks. Every output is compared with NumPy, and measured cycles with the schedule model.
- **Weights:** random INT8, seeded by layer name, because the compiler artifact gives shapes, not trained weights.
  `--weights file.npz` swaps in real ones later.
- **Inputs:** random UINT8 patches for `--positions` output positions (a slice of the layer, default 2).
- **Size limit:** layers needing more than `--max-tiles` 1-bit tiles (default 16) are listed as skipped, never silently left out.

## Sandbox results (shapes like tiny MobileNetV2 layers)

| shape K,C,R,S,g | 4-bit tiles | 4-bit exact | 1-bit tiles | one block | two blocks | cut | model error |
|---|---|---|---|---|---|---|---|
| 16,3,3,3,1 | 2 | yes | 8 | 791,563 | 375,909 | 52.5 % | 0.013 % |
| 16,16,3,3,16 | 6 | yes | 8 | 272,101 | 135,156 | 50.3 % | 0.038 % |
| 24,16,1,1,1 | 3 | yes | 12 | 720,257 | 350,108 | 51.4 % | 0.023 % |
| 10,32,1,1,1 | 2 | yes | 5 | 579,061 | 317,516 | 45.2 % | 0.010 % |

All 1-bit outputs are exact in both layouts. New files: `run_x2_layer_e2e.py`, `tests/test_x2_layer_e2e.py`. No existing file changes.

## Real tiny MobileNetV2: all 17 layers (your machine, 2026-11-16, `python3 run_x2_layer_e2e.py --sim`)

| layer | K,C,R,S,g | 4-bit tiles | 4-bit exact | 1-bit tiles | one block | two blocks | cut | model error | sim exact |
|---|---|---|---|---|---|---|---|---|---|
| features_0_0 | 8,1,3,3,1 | 1 | yes | 4 | 127,265 | 60,176 | 52.7 % | 0.035 % | yes |
| features_1_conv_0_0 | 8,8,3,3,8 | 3 | yes | 4 | 124,561 | 63,204 | 49.3 % | 0.035 % | yes |
| features_1_conv_1 | 8,8,1,1,1 | 1 | yes | 4 | 115,926 | 55,416 | 52.2 % | 0.038 % | yes |
| features_2_conv_0_0 | 16,8,1,1,1 | 2 | yes | 8 | 250,086 | 123,594 | 50.6 % | 0.042 % | yes |
| features_2_conv_1_0 | 16,16,3,3,16 | 6 | yes | 8 | 261,881 | 135,157 | 48.4 % | 0.040 % | yes |
| features_2_conv_2 | 12,16,1,1,1 | 2 | yes | 6 | 343,164 | 179,992 | 47.5 % | 0.022 % | yes |
| features_3_conv_0_0 | 24,12,1,1,1 | 3 | yes | 12 | 559,369 | 279,875 | 50.0 % | 0.029 % | yes |
| features_3_conv_1_0 | 24,24,3,3,24 | 8 | yes | 12 | 421,466 | 209,239 | 50.4 % | 0.119 % | yes |
| features_3_conv_2 | 12,24,1,1,1 | 2 | yes | 6 | 529,237 | 251,675 | 52.4 % | 0.014 % | yes |
| features_4_conv_0_0 | 24,12,1,1,1 | 3 | yes | 12 | 537,834 | 274,400 | 49.0 % | 0.030 % | yes |
| features_4_conv_1_0 | 24,24,3,3,24 | 8 | yes | 12 | 420,006 | 203,099 | 51.6 % | 0.039 % | yes |
| features_4_conv_2 | 16,24,1,1,1 | 2 | yes | 8 | 698,709 | 353,937 | 49.3 % | 0.015 % | yes |
| features_5_conv_0_0 | 32,16,1,1,1 | 4 | yes | 16 | 983,454 | 498,683 | 49.3 % | 0.023 % | yes |
| features_5_conv_1_0 | 32,32,3,3,32 | 11 | yes | 16 | 574,481 | 285,265 | 50.3 % | 0.039 % | yes |
| features_5_conv_2 | 16,32,1,1,1 | 2 | yes | 8 | 952,038 | 470,920 | 50.5 % | 0.011 % | yes |
| features_6_0 | 32,16,1,1,1 | 4 | yes | 16 | 937,099 | 498,831 | 46.8 % | 0.024 % | yes |
| classifier | 10,32,1,1,1 | 2 | yes | 5 | 589,646 | 318,125 | 46.0 % | 0.069 % | yes |

- 2 output positions per layer.
- Weights are random INT8 (seeded); 4-bit uses the ASSUMED cell rule.
- The 1-bit runs use the BM behavioral X2. None of this is silicon.
