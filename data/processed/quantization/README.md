# Quantization release data

- `storage_accuracy.csv`: all 21 points used by the paper figure.
- `baseline_{15,25}_accuracy.csv`: seven tasks plus macro for both baselines
  at BF16, W8A16, and W4A16.
- `baseline_{15,25}_paired_bootstrap.csv`: 10,000-sample paired BF16-versus-
  quantized intervals for every task and macro.
- `baseline_{15,25}_bytes.csv`: exact packed artifact sizes and SHA-256
  digests; artifact paths are intentionally replaced by non-redistributed
  basenames.

Accuracy is a fraction in `[0,1]`. GiB uses `2^30` bytes. The raw W8A16 and
W4A16 labels are displayed as INT8 and INT4 in the manuscript.
