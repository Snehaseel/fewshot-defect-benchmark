"""Turns results.csv into the manuscript's tables, figure and statistics.

    python analyze.py --out outputs

Writes to outputs/analysis/:
    table2_<dataset>.csv / .md    mean image AUROC (%) by method and k, plus recall at k = 4
    table_sd_<dataset>.csv        standard deviation across subsets (for S1 Table)
    table_pixel_<dataset>.csv     pixel AUROC (%)
    table_deploy.csv              recall and realised FPR at the training-only threshold
    table_texture_object.csv      texture vs object categories (MVTec AD)
    stats_wilcoxon_holm.csv       pre-specified paired comparisons across categories
    failures_k4.csv               per-defect-type miss rate of the best method at k = 4
    fig1_auroc_vs_k.png / .tif    Fig 1
    s1_table_full.csv             every run (S1 Table)
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

K_ORDER = ["0", "1", "2", "4", "8", "16", "full"]
LABELS = {
    "simclr_scratch_standard": "SimCLR from scratch",
    "simclr_scratch_mild": "SimCLR from scratch, mild aug.",
    "simclr_finetune_standard": "SimCLR fine-tuned",
    "simclr_finetune_mild": "SimCLR fine-tuned, mild aug.",
    "padim": "PaDiM",
    "patchcore": "PatchCore",
    "patchcore_dinov2": "PatchCore with DINOv2",
    "winclip": "WinCLIP / WinCLIP+",
    "pixel_knn": "Pixel kNN (sanity check)",
}
# Pre-specified comparisons (reference, comparator). Written down before looking at results.
COMPARISONS = [
    ("patchcore", "simclr_scratch_standard"),
    ("patchcore", "simclr_finetune_standard"),
    ("patchcore", "simclr_scratch_mild"),
    ("patchcore", "simclr_finetune_mild"),
    ("patchcore", "padim"),
    ("patchcore", "patchcore_dinov2"),
    ("patchcore", "winclip"),
]


def load(out: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(out, "results.csv"), dtype={"k": str})
    df["k"] = pd.Categorical(df["k"], K_ORDER, ordered=True)
    return df


def category_means(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean over subsets within each category (the unit for statistics)."""
    return df.groupby(["dataset", "method", "k", "category"], observed=True)[col].mean().reset_index()


def summary_table(df, col, scale=100.0):
    cm = category_means(df, col)
    mean = cm.groupby(["dataset", "method", "k"], observed=True)[col].mean().mul(scale)
    # SD across subsets: average each subset over categories, then SD across the subsets.
    sm = df.groupby(["dataset", "method", "k", "subset"], observed=True)[col].mean().mul(scale)
    sd = sm.groupby(["dataset", "method", "k"], observed=True).std()
    return mean.unstack("k"), sd.unstack("k")


def order_methods(index):
    order = [m for m in LABELS if m in index]
    return order + [m for m in index if m not in order]


def write_tables(df, adir: Path):
    for ds in df["dataset"].unique():
        d = df[df["dataset"] == ds]
        mean, sd = summary_table(d, "image_auroc")
        mean, sd = mean.loc[ds], sd.loc[ds]
        mean = mean.reindex(order_methods(mean.index))
        rec, _ = summary_table(d[d["k"] == "4"], "recall_oracle_5fpr")
        if "4" in rec.columns:
            mean["Recall at 5% FPR, k = 4"] = rec.loc[ds]["4"].reindex(mean.index)
        mean.index = [LABELS.get(m, m) for m in mean.index]
        mean.round(1).to_csv(adir / f"table2_{ds}.csv")
        (adir / f"table2_{ds}.md").write_text(mean.round(1).fillna("—").to_markdown())
        sd.reindex(order_methods(sd.index)).round(1).to_csv(adir / f"table_sd_{ds}.csv")
        pix, _ = summary_table(d, "pixel_auroc")
        pix.loc[ds].reindex(order_methods(pix.loc[ds].index)).round(1).to_csv(adir / f"table_pixel_{ds}.csv")

    dep = df[df["recall_deploy"].notna()]
    if len(dep):
        cm = dep.groupby(["dataset", "method", "k", "category"], observed=True)[
            ["recall_deploy", "fpr_deploy"]].mean()
        t = cm.groupby(["dataset", "method", "k"], observed=True).mean().mul(100).round(1)
        t.to_csv(adir / "table_deploy.csv")

    mv = df[df["dataset"] == "mvtec"]
    if len(mv):
        cm = category_means(mv, "image_auroc")
        cm = cm.merge(mv[["category", "is_texture"]].drop_duplicates(), on="category")
        t = cm.groupby(["method", "k", "is_texture"], observed=True)["image_auroc"].mean().mul(100)
        t = t.unstack("is_texture").rename(columns={0: "objects", 1: "textures"}).round(1)
        t["objects_minus_textures"] = (t.get("objects") - t.get("textures")).round(1)
        t.to_csv(adir / "table_texture_object.csv")


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def write_stats(df, adir: Path):
    """Wilcoxon signed-rank over categories (MVTec AD and VisA pooled, n up to 27),
    one family of tests across all comparisons and k, Holm-corrected."""
    cm = category_means(df, "image_auroc")
    cm["unit"] = cm["dataset"] + "/" + cm["category"]
    wide = cm.pivot_table(index=["k", "unit"], columns="method", values="image_auroc", observed=True)
    rows = []
    for k in K_ORDER:
        if k not in wide.index.get_level_values(0):
            continue
        w = wide.loc[k]
        for ref, comp in COMPARISONS:
            if ref not in w or comp not in w:
                continue
            pair = w[[ref, comp]].dropna()
            diff = (pair[comp] - pair[ref]) * 100
            if len(pair) < 6 or np.allclose(diff, 0):
                p, stat = np.nan, np.nan
            else:
                stat, p = wilcoxon(pair[comp], pair[ref])
            rows.append({"k": k, "reference": ref, "comparator": comp, "n_categories": len(pair),
                         "median_diff_pp": round(float(np.median(diff)), 2) if len(pair) else np.nan,
                         "mean_diff_pp": round(float(np.mean(diff)), 2) if len(pair) else np.nan,
                         "comparator_wins": int((diff > 0).sum()), "W": stat, "p": p})
    st = pd.DataFrame(rows)
    if len(st):
        st["p_holm"] = holm(st["p"].values)
        st.to_csv(adir / "stats_wilcoxon_holm.csv", index=False)
    return st


def write_failures(df, out: str, adir: Path, k="4"):
    """Miss rate by defect type for the method with the highest mean AUROC at k."""
    d = df[df["k"] == k]
    if not len(d):
        return
    best = d.groupby("method")["image_auroc"].mean().idxmax()
    rows = []
    for f in glob.glob(os.path.join(out, "scores", f"*__{best}__k{k}__s*.npz")):
        ds, cat = Path(f).name.split("__")[:2]
        z = np.load(f, allow_pickle=False)
        s, y, t = z["scores"], z["labels"], z["defect_types"]
        thr = np.quantile(s[y == 0], 0.95)
        for dt in np.unique(t[y == 1]):
            sel = (t == dt) & (y == 1)
            rows.append({"dataset": ds, "category": cat, "defect_type": dt,
                         "n": int(sel.sum()), "miss_rate": float(np.mean(s[sel] <= thr))})
    if rows:
        r = pd.DataFrame(rows).groupby(["dataset", "category", "defect_type"]).agg(
            n=("n", "first"), miss_rate=("miss_rate", "mean")).reset_index()
        r.sort_values("miss_rate", ascending=False).assign(method=best).to_csv(
            adir / f"failures_k{k}.csv", index=False)


def write_figure(df, adir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [d for d in ["mvtec", "visa"] if d in df["dataset"].unique()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.5 * len(datasets), 4.5), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        mean, sd = summary_table(df[df["dataset"] == ds], "image_auroc")
        mean, sd = mean.loc[ds], sd.loc[ds]
        for m in order_methods(mean.index):
            ks = [k for k in K_ORDER if k in mean.columns and not np.isnan(mean.loc[m, k])]
            x = [K_ORDER.index(k) for k in ks]
            y = mean.loc[m, ks].values.astype(float)
            e = np.nan_to_num(sd.loc[m, ks].values.astype(float))
            ax.plot(x, y, marker="o", label=LABELS.get(m, m))
            ax.fill_between(x, y - e, y + e, alpha=0.15)
        ax.set_xticks(range(len(K_ORDER)), ["0", "1", "2", "4", "8", "16", "All"])
        ax.set_xlabel("Normal training images per category (k)")
        ax.set_ylabel("Image-level AUROC (%)")
        ax.set_title({"mvtec": "MVTec AD", "visa": "VisA"}[ds])
        ax.grid(alpha=0.3)
    axes[0][-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(adir / "fig1_auroc_vs_k.png", dpi=200)
    # PLOS requires TIFF (or EPS) figures at 300-600 dpi.
    fig.savefig(adir / "fig1_auroc_vs_k.tif", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)
    adir = Path(args.out) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    df = load(args.out)
    df.to_csv(adir / "s1_table_full.csv", index=False)
    write_tables(df, adir)
    st = write_stats(df, adir)
    write_failures(df, args.out, adir)
    write_figure(df, adir)
    print(f"Wrote analysis to {adir}")
    for f in sorted(adir.glob("table2_*.md")):
        print(f"\n{f.name}\n{f.read_text()}")
    if len(st):
        print("\nPre-specified comparisons (Holm-corrected):")
        print(st[["k", "reference", "comparator", "n_categories", "median_diff_pp", "p_holm"]]
              .to_string(index=False))


if __name__ == "__main__":
    main()
