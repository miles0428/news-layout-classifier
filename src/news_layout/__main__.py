#!/usr/bin/env python3
"""
CLI for news-layout-classifier.

    news-layout bench LEFT RIGHT OTHER [--rounds N] [--verbose]
    news-layout batch INPUT OUTPUT
    news-layout train LEFT RIGHT [--out FILE]
    news-layout single IMAGE --templates FILE
"""

import argparse, shutil, sys, random
from pathlib import Path
from collections import Counter

from news_layout import NewsLayoutClassifier, DEFAULT_ACTIVE_COLS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def train(args):
    clf = NewsLayoutClassifier(
        size=args.resolution,
        pixel_tol=args.pixel_tol,
        min_match=args.min_match,
        active_cols=args.active_cols if args.active_cols else DEFAULT_ACTIVE_COLS,
    )
    counts = clf.train(Path(args.left_dir), Path(args.right_dir))
    print(f"Trained: {counts}")
    if args.out:
        clf.save_templates(Path(args.out))
        print(f"Templates saved: {args.out}")


def batch(args):
    clf = NewsLayoutClassifier(
        size=args.resolution,
        pixel_tol=args.pixel_tol,
        min_match=args.min_match,
        active_cols=args.active_cols if args.active_cols else DEFAULT_ACTIVE_COLS,
    )
    if args.templates:
        clf.load_templates(Path(args.templates))
    else:
        clf.train(Path(args.left_dir), Path(args.right_dir))

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    buckets = {"left": [], "right": [], "other": []}

    for i, fpath in enumerate(files, 1):
        label = clf.classify(fpath)
        buckets[label].append(fpath.name)
        print(f"[{i}/{len(files)}] {fpath.name} -> {label}")

    for d in ["left", "right", "other"]:
        (output_dir / d).mkdir(parents=True, exist_ok=True)
    for label, names in buckets.items():
        for fname in names:
            src = input_dir / fname
            dst = output_dir / label / fname
            if args.move:
                shutil.move(str(src), str(dst))
            else:
                shutil.copy2(str(src), str(dst))

    print(f"\nTotal: {len(files)}")
    for d in ["left", "right", "other"]:
        print(f"  {d:8s}: {len(buckets[d])}")


def single(args):
    clf = NewsLayoutClassifier()
    clf.load_templates(Path(args.templates))
    label, scores = clf.classify_with_scores(args.image_path)
    print(f"Result: {label}")
    print(f"Scores: left={scores['left']:.4f}  right={scores['right']:.4f}")


def bench(args):
    clf = NewsLayoutClassifier(
        size=args.resolution,
        pixel_tol=args.pixel_tol,
        min_match=args.min_match,
        active_cols=args.active_cols if args.active_cols else DEFAULT_ACTIVE_COLS,
    )
    files_L = [f for f in Path(args.left_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS]
    files_R = [f for f in Path(args.right_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS]
    files_O = [f for f in Path(args.other_dir).iterdir() if f.suffix.lower() in IMAGE_EXTS]

    random.seed(42)
    all_results = []
    N = args.rounds
    SPLIT = 0.2

    for _ in range(N):
        idx_L = list(range(len(files_L))); random.shuffle(idx_L)
        idx_R = list(range(len(files_R))); random.shuffle(idx_R)
        idx_O = list(range(len(files_O))); random.shuffle(idx_O)
        sL = int(len(files_L) * SPLIT)
        sR = int(len(files_R) * SPLIT)
        sO = int(len(files_O) * SPLIT)
        train_L = [files_L[i] for i in idx_L[sL:]]; test_L = [files_L[i] for i in idx_L[:sL]]
        train_R = [files_R[i] for i in idx_R[sR:]]; test_R = [files_R[i] for i in idx_R[:sR]]
        train_O = [files_O[i] for i in idx_O[sO:]]; test_O = [files_O[i] for i in idx_O[:sO]]

        clf.train_from_lists(train_L, train_R, train_O)

        for f in test_L:
            all_results.append(("left", clf.classify(f), f))
        for f in test_R:
            all_results.append(("right", clf.classify(f), f))
        for f in test_O:
            all_results.append(("other", clf.classify(f), f))

    labels = ["left", "right", "other"]
    CM = {l: {m: 0 for m in labels} for l in labels}
    for true, pred, _ in all_results:
        CM[true][pred] += 1

    print(f"\n=== {N}x 80/20 Cross-Validation ===")
    hdr = f"{'True':10}" + "".join([f"{l:8}" for l in labels]) + f"{'Recall':8}"
    print(hdr)
    print("-" * len(hdr))
    for true in labels:
        row = "".join([f"{CM[true][m]:8}" for m in labels])
        rec = CM[true][true] / sum(CM[true].values()) * 100
        print(f"{true:10}{row}{rec:7.1f}%")
    correct = sum(1 for t, p, _ in all_results if t == p)
    print("-" * len(hdr))
    print(f"Overall: {correct}/{len(all_results)} = {correct/len(all_results)*100:.2f}%")

    if args.verbose:
        print(f"\n=== Per-Image Log ({len(all_results)} images) ===")
        print(f"{'Filename':<40} {'True':6} {'Pred':6} {'OK':3}")
        print("-" * 60)
        for true, pred, fpath in all_results:
            ok = "Y" if true == pred else "N"
            print(f"{fpath.name:<40} {true:6} {pred:6} {ok:3}")


def main():
    ap = argparse.ArgumentParser(description="news-layout-classifier")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train")
    p.add_argument("left_dir"); p.add_argument("right_dir")
    p.add_argument("--out")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--pixel-tol", type=float, default=0.05)
    p.add_argument("--min-match", type=float, default=0.30)
    p.add_argument("--active-cols", type=int, nargs="+", default=None)

    p = sub.add_parser("batch")
    p.add_argument("input_dir", type=Path); p.add_argument("output_dir", type=Path)
    p.add_argument("--templates"); p.add_argument("--left-dir"); p.add_argument("--right-dir")
    p.add_argument("--move", action="store_true")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--pixel-tol", type=float, default=0.05)
    p.add_argument("--min-match", type=float, default=0.30)
    p.add_argument("--active-cols", type=int, nargs="+", default=None)

    p = sub.add_parser("single")
    p.add_argument("image_path", type=Path); p.add_argument("--templates", required=True)

    p = sub.add_parser("bench")
    p.add_argument("left_dir"); p.add_argument("right_dir"); p.add_argument("other_dir")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--resolution", type=int, default=128)
    p.add_argument("--pixel-tol", type=float, default=0.05)
    p.add_argument("--min-match", type=float, default=0.30)
    p.add_argument("--active-cols", type=int, nargs="+", default=None)

    args = ap.parse_args()
    if args.command == "train": train(args)
    elif args.command == "batch": batch(args)
    elif args.command == "single": single(args)
    elif args.command == "bench": bench(args)


if __name__ == "__main__":
    main()
