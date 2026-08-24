<div align="center">

# RGCLS · Replacement-Guided Cross-Layer Sharing
### 以有方向性的全模型替換成本，建立可稽核、可恢復的跨層 FFN 分享結構

[English](README.md) · [論文 PDF](paper/main.pdf) · [論文原始檔](paper/README.md) · [方法](docs/METHOD.md) · [結果](docs/RESULTS.md) · [重現](docs/REPRODUCTION.md) · [外部 baseline 匯入](docs/BASELINES.md)

<img src="assets/figures/framework.png" alt="Replacement-guided cross-layer sharing 方法架構" width="100%">

</div>

---

這個方法不是用權重距離直接決定哪些 layer 可以共用 FFN，而是把 donor
FFN 實際放到 target layer，量測凍結模型的 NLL 與 KL 變化。接著在固定
FFN 儲存數量下，用 complete-link 控制群內最壞替換成本，再依真正的
donor-to-target 方向挑代表 FFN。

這份倉庫以「可稽核研究產物」的方式整理：Ours 的程式、Slurm 工作、
精簡觀測資料、圖片、論文原始檔與驗證器都已納入；Basis Sharing 與
SVD-LLM 因為在另一台 server 執行，已預留完整的 source、config、Slurm、
prediction、byte manifest、quantization manifest 與 provenance 位置。

## 主要結果

| Backbone | 壓縮目標 | Ours CE+KD | 最強 baseline | 差距 |
|---|---:|---:|---:|---:|
| Llama-3.2-3B | 15% | **84.83 ± 0.47** | 71.28 | +13.55 pp |
| Llama-3.2-3B | 20% | **85.06 ± 0.08** | 68.15 | +16.91 pp |
| Llama-3.2-3B | 25% | **84.09 ± 0.18** | 65.06 | +19.03 pp |
| Llama-3.1-8B | 15% | **86.14 ± 0.33** | 76.43 | +9.71 pp |
| Llama-3.1-8B | 20% | **85.94 ± 0.34** | 73.89 | +12.05 pp |
| Llama-3.1-8B | 25% | **85.92 ± 0.36** | 69.96 | +15.96 pp |

Ours 的 8B--25% W8A16 checkpoint 為 6.50 GiB、86.34% macro accuracy；
相較 dense BF16 儲存量減少 56.53%，與 dense teacher 相差 0.51 個百分點。

## 目前完整度

| 項目 | 狀態 |
|---|---|
| Ours 方法、訓練、評估與分析程式 | **已整理** |
| Ours Pure / CE / CE+KD 資料 | **已整理並由驗證器檢查** |
| 結構分析、joint distortion、ablation | **已整理並由驗證器檢查** |
| Basis Sharing 完整重現檔 | **等待另一台 server 匯入** |
| SVD-LLM 完整重現檔 | **等待另一台 server 匯入** |
| 三方法正式量化圖 | **等待外部 manifest 後自動產生** |

論文在外部量化資料尚未齊全時仍可編譯，但會明確顯示 pending，不會產生
無法追溯的假圖。

## 驗證

一般電腦可直接執行標準函式庫驗證：

```bash
python scripts/verify_repository.py
```

在 NCHC 請一律透過 Slurm：

```bash
sbatch slurm/verify_repository.sbatch
```

驗證器會從 per-task rows 重算 Ours macro mean/SD、檢查
`delta <= Delta`、paired-example 對齊、論文引用、README 連結，以及公開
檔案是否殘留私人路徑或憑證。

## 外部 server 補檔

每個 baseline 需要四個正式 payload：`predictions.csv`、
`byte_manifest.csv`、`quantization.csv`、`run_manifest.json`。格式與精確
目錄請看 [data/external/README.md](data/external/README.md)。全部放好後：

```bash
sbatch slurm/import_external_baselines.sbatch
```

完整重現流程與環境變數請看 [docs/REPRODUCTION.md](docs/REPRODUCTION.md)，
檔案導覽請看 [docs/FILE_GUIDE.md](docs/FILE_GUIDE.md)。
