# Ours artifacts

These compact CSV files are the public, path-free release of the observed
results used by the paper and figures.

- `main_summary.csv` and `per_task.csv`: Pure, CE, and CE+KD results for six
  operating points and seeds 42/43/44.
- `directed_costs.csv`, `pair_analysis.csv`, and `group_analysis.csv`: directed
  replacement measurements and selected sharing groups.
- `joint_analysis.csv` and `structural_ablation.csv`: simultaneous-deployment
  distortion and the 3B--20% structure ablation.
- `quantization.csv`: Ours BF16/W8A16/W4A16 storage and accuracy at 8B--15/25%.
- `dense_references.csv`: dense pretrained and task-trained reference scores.

Private cluster paths, checkpoints, full datasets, and model weights are not
redistributed. See `REPRODUCIBILITY.md` for the boundary.
