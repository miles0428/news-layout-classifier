#!/usr/bin/env python3
"""
CLI for news-layout-classifier.

Usage:
    # Train from labelled directories
    python -m classify.main train LEFT_DIR RIGHT_DIR OTHER_DIR [--out templates.npz]

    # Classify a directory
    python -m classify.main batch INPUT_DIR OUTPUT_DIR [--templates templates.npz]
                                   [--left-dir L] [--right-dir R] [--other-dir O]

    # Single image
    python -m classify.main single IMAGE [--templates templates.npz]

    # Benchmark on a labelled directory
    python -m classify.main bench LEFT_DIR RIGHT_DIR OTHER_DIR [--rounds 5]
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))
from classify.classifier import NewsLayoutClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def train(args):
    clf = NewsLayoutClassifier(size=args.resolution)
    counts = clf.train(
        left_dir=Path(args.left_dir),
        right_dir=Path(args.right_dir),
        other_dir=Path(args.other_dir),
    )
    print(f"Trained: {counts}")

    if args.out:
        clf.save_templates(Path(args.out))
        print(f"Templates saved: {args.out}")

    if args.test_dirs:
        # Quick self-test on held-out directories
        l_dir, r_dir, o_dir = [Path(d) for d in args.test_dirs]
        files_l = [f for f in l_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        files_r = [f for f in r_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        files_o = [f for f in o_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
        results = []
        for f in files_l: results.append(("left",  clf.classify(f)))
        for f in files_r: results.append(("right", clf.classify(f)))
        for f in files_o: results.append(("other", clf.classify(f)))
        print("\nSelf-test results:")
        for true in ["left", "right", "other"]:
            preds = [r[1] for r in results if r[0] == true]
            c = Counter(preds)
            total = len(preds)
            acc = c.get(true, 0) / total * 100
            print(f"  {true}: {c.get(true,0)}/{total} = {acc:.1f}%")


def batch(args):
    templates_path = Path(args.templates) if args.templates else None
    clf = NewsLayoutClassifier(size=args.resolution)

    if templates_path and templates_path.exists():
        clf.load_templates(templates_path)
        logger.info(f"Loaded templates from {templates_path}")
    else:
        # Build inline templates from labelled subdirs if provided
        if args.left_dir and args.right_dir and args.other_dir:
            clf.train(Path(args.left_dir), Path(args.right_dir), Path(args.other_dir))
        else:
            logger.error("No templates provided and no --left-dir/--right-dir/--other-dir")
            sys.exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    total = len(files)
    logger.info(f"Classifying {total} images from {input_dir}")

    buckets = {"left": [], "right": [], "other": []}

    for i, fpath in enumerate(files, 1):
        t0 = time.time()
        label = clf.classify(fpath)
        elapsed = time.time() - t0
        buckets[label].append(fpath.name)
        logger.info(f"[{i}/{total}] {fpath.name} -> {label} ({elapsed*1000:.0f}ms)")

    # Write output
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

    # Report
    logger.info("=" * 50)
    logger.info(f"Total : {total}")
    for d in ["left", "right", "other"]:
        logger.info(f"  {d:8s}: {len(buckets[d])}")

    results_txt = output_dir / "classification_results.txt"
    with open(results_txt, "w", encoding="utf-8") as fh:
        fh.write(f"Input   : {input_dir}\nTotal   : {total}\n\n")
        for label, names in buckets.items():
            fh.write(f"# {label} ({len(names)})\n")
            for n in names: fh.write(f"  {n}\n")
            fh.write("\n")
    logger.info(f"Results: {results_txt}")


def single(args):
    templates_path = Path(args.templates) if args.templates else None
    clf = NewsLayoutClassifier()

    if templates_path and templates_path.exists():
        clf.load_templates(templates_path)
    else:
        logger.error("No templates file. Use --templates or train first.")
        sys.exit(1)

    path = Path(args.image_path)
    label, scores = clf.classify_with_scores(path)
    print(f"Result: {label}")
    print(f"Scores: left={scores['left']:.4f}  right={scores['right']:.4f}  other={scores['other']:.4f}")


def bench(args):
    clf = NewsLayoutClassifier(size=args.resolution)
    tpl_L = Path(args.left_dir)
    tpl_R = Path(args.right_dir)
    tpl_O = Path(args.other_dir)

    files_L = [f for f in tpl_L.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    files_R = [f for f in tpl_R.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    files_O = [f for f in tpl_O.iterdir() if f.suffix.lower() in IMAGE_EXTS]

    import random
    random.seed(42)
    all_results = []
    N = args.rounds
    SPLIT = 0.2

    for _ in range(N):
        idx_L = list(range(len(files_L))); random.shuffle(idx_L)
        idx_R = list(range(len(files_R))); random.shuffle(idx_R)
        idx_O = list(range(len(files_O))); random.shuffle(idx_O)
        sL = int(len(files_L)*SPLIT); sR = int(len(files_R)*SPLIT); sO = int(len(files_O)*SPLIT)
        train_L = [files_L[i] for i in idx_L[sL:]]; test_L = [files_L[i] for i in idx_L[:sL]]
        train_R = [files_R[i] for i in idx_R[sR:]]; test_R = [files_R[i] for i in idx_R[:sR]]
        train_O = [files_O[i] for i in idx_O[sO:]]; test_O = [files_O[i] for i in idx_O[:sO]]

        clf.train_from_lists(train_L, train_R, train_O)

        for f in test_L: all_results.append(("left",  clf.classify(f)))
        for f in test_R: all_results.append(("right", clf.classify(f)))
        for f in test_O: all_results.append(("other", clf.classify(f)))

    labels = ["left", "right", "other"]
    print("\n=== %dx 80/20 Cross-Validation ===" % N)
    header = "%-10s" % "True" + "".join(["%8s" % l for l in labels]) + "%8s" % "Acc"
    print(header)
    print("-" * len(header))
    for true in labels:
        preds = [r[1] for r in all_results if r[0] == true]
        c = Counter(preds)
        total = len(preds)
        acc = c.get(true, 0) / total * 100
        print("%-10s" % true + "".join(["%8d" % c.get(l, 0) for l in labels]) + "%7.1f%%" % acc)
    correct = sum(1 for r in all_results if r[0] == r[1])
    print("-" * len(header))
    print("Overall: %d/%d = %.1f%%" % (correct, len(all_results), correct/len(all_results)*100))


def main():
    ap = argparse.ArgumentParser(description="News Layout Classifier")
    sub = ap.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train templates from labelled directories")
    p_train.add_argument("left_dir",  help="Directory with LEFT-class images")
    p_train.add_argument("right_dir", help="Directory with RIGHT-class images")
    p_train.add_argument("other_dir", help="Directory with OTHER-class images")
    p_train.add_argument("--out", help="Save templates to .npz file")
    p_train.add_argument("--test-dirs", nargs=3, metavar=("L","R","O"), help="Quick self-test dirs")
    p_train.add_argument("--resolution", type=int, default=128)

    p_batch = sub.add_parser("batch", help="Classify a directory")
    p_batch.add_argument("input_dir",  type=Path)
    p_batch.add_argument("output_dir", type=Path)
    p_batch.add_argument("--templates", help=".npz template file")
    p_batch.add_argument("--left-dir")
    p_batch.add_argument("--right-dir")
    p_batch.add_argument("--other-dir")
    p_batch.add_argument("--move", action="store_true")
    p_batch.add_argument("--resolution", type=int, default=128)

    p_single = sub.add_parser("single", help="Classify one image")
    p_single.add_argument("image_path", type=Path)
    p_single.add_argument("--templates", required=True)

    p_bench = sub.add_parser("bench", help="Cross-validation benchmark")
    p_bench.add_argument("left_dir")
    p_bench.add_argument("right_dir")
    p_bench.add_argument("other_dir")
    p_bench.add_argument("--rounds", type=int, default=5)
    p_bench.add_argument("--resolution", type=int, default=128)

    args = ap.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "batch":
        if not args.templates and not (args.left_dir and args.right_dir and args.other_dir):
            print("Error: --templates or --left-dir/--right-dir/--other-dir required")
            sys.exit(1)
        batch(args)
    elif args.command == "single":
        single(args)
    elif args.command == "bench":
        bench(args)


if __name__ == "__main__":
    main()
