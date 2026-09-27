"""Runs the full k-shot experiment grid and appends one row per run to a results CSV.

Examples
    python run.py --smoke                                   # 5-minute check on synthetic data
    python run.py --dataset mvtec --root /data/mvtec_ad
    python run.py --dataset visa  --root /data/VisA_20220922
    python run.py --dataset mvtec --root ... --methods patchcore,padim --categories bottle

Re-running the same command skips runs already in the results file, so an interrupted
session (e.g. a Colab disconnect) resumes where it stopped.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

import data as D
from metrics import evaluate

FIELDS = [
    "dataset", "category", "is_texture", "method", "k", "subset", "seed", "n_train",
    "n_test", "n_defective", "image_auroc", "pixel_auroc", "recall_oracle_5fpr",
    "recall_deploy", "fpr_deploy", "fit_seconds", "infer_ms_per_image",
]


SMOKE = False
K0_METHODS = {"winclip"}
FEW_SHOT_ONLY = {"winclip"}  # WinCLIP+ is a few-shot method; not run on the full training set
SEEDED_METHODS = {"patchcore", "patchcore_dinov2", "simclr_scratch_standard", "simclr_scratch_mild",
                  "simclr_finetune_standard", "simclr_finetune_mild"}


class PixelKNN:
    """Sanity-check baseline (not part of the study): nearest-neighbour distance on
    downsampled raw pixels. Needs only NumPy, so it is used for the smoke test and for
    testing the pipeline on machines without PyTorch."""
    name, produces_maps, supports_k0 = "pixel_knn", True, False

    def _f(self, x):
        x = x.reshape(len(x), 32, D.CACHE_SIZE // 32, 32, D.CACHE_SIZE // 32, 3).mean(axis=(2, 4))
        return x.reshape(len(x), -1).astype(np.float32) / 255.0

    def fit(self, train):
        self.bank = self._f(train)

    def score(self, images):
        f = self._f(images)
        d = np.sqrt(((f[:, None] - self.bank[None]) ** 2).sum(-1)).min(1)
        diff = np.abs(images.astype(np.float32) - images.mean(0, keepdims=True)).mean(-1)
        idx = np.linspace(0, D.CACHE_SIZE - 1, D.EVAL_SIZE).astype(int)
        return d, diff[:, idx][:, :, idx]

    def loo_scores(self, train):
        if len(train) < 2:
            return None
        f = self._f(train)
        d = np.sqrt(((f[:, None] - f[None]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        return d.min(1)


def build_method(name: str, category: D.CategoryData, seed: int):
    if name == "pixel_knn":
        return PixelKNN()
    import methods as M  # imported lazily so the smoke test runs without PyTorch
    if SMOKE:
        M.SIMCLR_STEPS = 20  # smoke test only checks that the code runs
    return M.build(name, category.prompt_name, seed)


def done_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {(r["dataset"], r["category"], r["method"], r["k"], r["subset"])
                for r in csv.DictReader(f)}


def append(path: str, row: dict, fields: list[str]) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def log_subset(path: str, dataset, category, k, subset, seed, paths) -> None:
    append(path, {"dataset": dataset, "category": category, "k": k, "subset": subset,
                  "seed": seed, "images": ";".join(os.path.basename(p) for p in paths)},
           ["dataset", "category", "k", "subset", "seed", "images"])


def run(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = str(out_dir / "results.csv")
    subsets_csv = str(out_dir / "subsets.csv")
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(exist_ok=True)
    finished = done_keys(results)
    logged_subsets = set()
    if os.path.exists(subsets_csv):
        with open(subsets_csv, newline="") as f:
            logged_subsets = {(r["dataset"], r["category"], r["k"], r["subset"]) for r in csv.DictReader(f)}

    ks = [k.strip() for k in args.ks.split(",")]
    method_names = [m.strip() for m in args.methods.split(",")]
    cats = args.categories.split(",") if args.categories else D.categories(args.dataset)

    for cat in cats:
        t0 = time.time()
        cd = D.load_category(args.dataset, args.root, cat)
        print(f"\n== {args.dataset}/{cat}: {len(cd.train_paths)} train, {len(cd.test_paths)} test "
              f"({cd.test_labels.sum()} defective) loaded in {time.time() - t0:.0f}s", flush=True)
        for mname in method_names:
            try:
                run_method(args, cd, mname, ks, finished, logged_subsets, results, subsets_csv, scores_dir)
            except Exception as e:  # keep going so one broken method does not hide the others
                import traceback
                tb = traceback.format_exc()
                print(f"  !! {mname} FAILED on {cat}: {type(e).__name__}: {e}", flush=True)
                append(str(out_dir / "errors.csv"), {"dataset": args.dataset, "category": cat,
                       "method": mname, "error": f"{type(e).__name__}: {e}", "traceback": tb},
                       ["dataset", "category", "method", "error", "traceback"])
        del cd


def run_method(args, cd, mname, ks, finished, logged_subsets, results, subsets_csv, scores_dir):
    cat = cd.category
    if True:
        if True:
            method = None
            for k in ks:
                n_subsets = 1 if k in ("0", "full") else args.subsets
                for s in range(n_subsets):
                    key = (args.dataset, cat, mname, k, str(s))
                    if key in finished:
                        continue
                    if k == "0":
                        idx = np.array([], dtype=int)
                        seed = 0
                    elif k == "full":
                        idx = np.arange(len(cd.train_paths))
                        seed = D.subset_seed(args.dataset, cat, k, s)
                    else:
                        kk = int(k)
                        if kk > len(cd.train_paths):
                            continue
                        seed = D.subset_seed(args.dataset, cat, k, s)
                        idx = D.sample_subset(len(cd.train_paths), kk, seed)
                    if k == "0" and mname not in K0_METHODS:
                        continue
                    if k == "full" and mname in FEW_SHOT_ONLY:
                        continue
                    # SimCLR training and PatchCore's coreset depend on the seed, so those
                    # methods are rebuilt per run; the others are built once per category.
                    if method is None or (mname in SEEDED_METHODS and method.seed != seed):
                        method = build_method(mname, cd, seed)
                    if (args.dataset, cat, k, str(s)) not in logged_subsets and k != "0":
                        log_subset(subsets_csv, args.dataset, cat, k, s, seed,
                                   [cd.train_paths[i] for i in idx])
                        logged_subsets.add((args.dataset, cat, k, str(s)))
                    train = cd.train_images[idx]

                    t = time.perf_counter()
                    method.fit(train)
                    fit_s = time.perf_counter() - t
                    t = time.perf_counter()
                    scores, maps = method.score(cd.test_images)
                    infer_ms = 1000 * (time.perf_counter() - t) / len(cd.test_images)
                    # Deployable threshold from training images only (k between 2 and 16).
                    loo = method.loo_scores(train) if k not in ("0", "full") else None

                    m = evaluate(np.asarray(scores), maps, cd.test_labels, cd.test_masks, loo)
                    row = {"dataset": args.dataset, "category": cat, "is_texture": int(cd.is_texture),
                           "method": mname, "k": k, "subset": s, "seed": seed, "n_train": len(idx),
                           "n_test": len(cd.test_labels), "n_defective": int(cd.test_labels.sum()),
                           "fit_seconds": round(fit_s, 3), "infer_ms_per_image": round(infer_ms, 3),
                           **{a: round(b, 6) for a, b in m.items()}}
                    append(results, row, FIELDS)
                    np.savez_compressed(
                        scores_dir / f"{args.dataset}__{cat}__{mname}__k{k}__s{s}.npz",
                        scores=np.asarray(scores, dtype=np.float32), labels=cd.test_labels,
                        defect_types=np.array(cd.test_defect_types),
                        test_files=np.array([os.path.relpath(p, args.root) for p in cd.test_paths]),
                        loo=np.array([] if loo is None else loo, dtype=np.float32))
                    print(f"  {mname:26s} k={k:>4s} s={s}  AUROC={m['image_auroc']:.3f}  "
                          f"pixAUROC={m['pixel_auroc']:.3f}  rec@5%={m['recall_oracle_5fpr']:.2f}  "
                          f"fit={fit_s:.1f}s", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["mvtec", "visa"], default="mvtec")
    ap.add_argument("--root", help="dataset root folder")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--ks", default="0,1,2,4,8,16,full")
    ap.add_argument("--subsets", type=int, default=5)
    ap.add_argument("--categories", default="")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny synthetic dataset; add --methods to also smoke-test the neural methods")
    args = ap.parse_args(argv)

    if args.smoke:
        global SMOKE
        SMOKE = True
        print("SMOKE TEST: synthetic data, SimCLR shortened to 20 steps. Not for real results.")
        root = os.path.join(args.out, "synthetic_mvtec")
        cats = D.make_synthetic_mvtec(root)
        args.dataset, args.root = "mvtec", root
        args.categories = ",".join(cats)
        args.ks = "0,2,full" if args.ks == "0,1,2,4,8,16,full" else args.ks
        args.subsets = 1
        if args.methods == "all":
            args.methods = "pixel_knn"
    elif not args.root:
        ap.error("--root is required unless --smoke is given")
    if args.methods == "all":
        import methods as M
        args.methods = ",".join(M.ALL_METHODS)
    run(args)


if __name__ == "__main__":
    sys.exit(main())
