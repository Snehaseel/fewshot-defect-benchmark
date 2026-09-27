# Few-shot defect detection benchmark (Manuscript 4)

Code for "How many normal images are needed? Label efficiency of contrastive self-supervised,
pretrained-feature and vision-language methods for visual defect detection".

## What it runs

| Method | Code name | Notes |
|---|---|---|
| SimCLR from scratch | `simclr_scratch_standard`, `simclr_scratch_mild` | ResNet-18, random init |
| SimCLR fine-tuned | `simclr_finetune_standard`, `simclr_finetune_mild` | ResNet-50, ImageNet init |
| PaDiM | `padim` | ResNet-18, 100 random dims, eps = 0.01 |
| PatchCore | `patchcore` | WideResNet-50, layers 2 + 3 |
| PatchCore with DINOv2 | `patchcore_dinov2` | ViT-B/14, blocks 8 and 12, 448 px |
| WinCLIP / WinCLIP+ | `winclip` | Anomalib 2.1.0, ViT-B-16-plus-240 (LAION-400M) |

Sample sizes k = 0 (WinCLIP only), 1, 2, 4, 8, 16 (five random subsets each) and the full
training set (one run; not run for WinCLIP). Every method gets exactly the same subsets.

## Setup (Google Colab or any machine with an NVIDIA GPU)

```bash
pip install -r requirements.txt
```

Download the data (both require accepting a licence):

* MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad — extract so that
  `mvtec_ad/bottle/train/good/...` exists.
* VisA: https://github.com/amazon-science/spot-diff — extract `VisA_20220922.tar` so that
  `VisA_20220922/split_csv/1cls.csv` exists.

## Run order

**1. Smoke test (a few minutes).** Checks the pipeline end to end on a tiny synthetic dataset.
Run the second line on the GPU machine before the real experiments; it checks every method.

```bash
python run.py --smoke
python run.py --smoke --out smoke_nn --methods patchcore,patchcore_dinov2,padim,simclr_scratch_standard,simclr_finetune_mild,winclip
```

**2. Fast methods first (a few GPU hours for both datasets).**

```bash
python run.py --dataset mvtec --root /data/mvtec_ad        --methods patchcore,patchcore_dinov2,padim,winclip
python run.py --dataset visa  --root /data/VisA_20220922   --methods patchcore,patchcore_dinov2,padim,winclip
```

**3. SimCLR (the slow part).** Each run trains a network. Rough times per run on a T4:
about 1 minute for ResNet-18, 3 minutes for ResNet-50; an A100 or RTX 4090 is 3–4 times
faster. With 27 categories × 26 runs per variant, the four SimCLR variants take roughly
4 days on a T4 or about 1 day on an A100. Run the standard-augmentation variants first:

```bash
python run.py --dataset mvtec --root /data/mvtec_ad      --methods simclr_scratch_standard,simclr_finetune_standard
python run.py --dataset visa  --root /data/VisA_20220922 --methods simclr_scratch_standard,simclr_finetune_standard
# then the same with simclr_scratch_mild,simclr_finetune_mild
```

All commands can be interrupted and re-run: finished runs are skipped. On Colab, set
`--out` to a Google Drive folder so results survive disconnects.

**4. Analysis.**

```bash
python analyze.py --out outputs
```

## Output and where it goes in the manuscript

| File | Manuscript |
|---|---|
| `analysis/table2_mvtec.md` | Table 2 |
| `analysis/table2_visa.md`, `table_sd_*.csv`, `table_pixel_*.csv`, `s1_table_full.csv` | S1 Table |
| `analysis/fig1_auroc_vs_k.tif` | Fig 1 (TIFF, 300 dpi, as PLOS requires) |
| `analysis/stats_wilcoxon_holm.csv` | "Contrastive training compared with frozen features" [R4.8] |
| `analysis/table_texture_object.csv` | "Textures and objects" [R4.9] |
| `analysis/failures_k4.csv` | "Failure cases" [R4.10]; choose Fig 2 examples from it |
| `analysis/table_deploy.csv` | Recall and actual false-alarm rate with a threshold set from training images only |
| `subsets.csv` | S1 File (training images in every subset) |
| `results.csv`, `scores/` | Raw results; deposit on Zenodo with the code |

## Design choices to state in Methods

* Images are resized whole (no centre crop) so that all methods are scored on the same
  256 × 256 ground-truth mask grid. The original PatchCore paper used a centre crop.
* PatchCore keeps every patch for k ≤ 16 and uses a 10% greedy coreset for the full set;
  the optional re-weighting step of the original paper is not used.
* PaDiM at k = 1 has zero sample covariance, so its score reduces to scaled Euclidean
  distance to the mean.
* SimCLR: 500 steps, batch 64 pairs, temperature 0.2, AdamW (lr 1e-3 scratch, 1e-4
  fine-tuned). Score = mean cosine distance to the 5 nearest memory embeddings; the memory
  holds each training image plus 16 mildly augmented views when k ≤ 16.
* Two triage thresholds are reported: an oracle (95th percentile of the test set's good
  scores; uses test labels, so it is an upper bound) and a deployable one (maximum
  leave-one-out score on the k training images, k = 2–16; no test information). For SimCLR
  the leave-one-out step does not retrain the encoder, which biases that threshold low; the
  realised false-alarm rate is reported so readers can see this.
* Statistics: Wilcoxon signed-rank tests over the 27 categories (both datasets pooled) for
  seven comparisons fixed in advance (`COMPARISONS` in analyze.py), Holm-corrected together.

Hyperparameters are fixed in the code and must not be changed after looking at test results.
