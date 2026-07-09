#!/usr/bin/env python3
"""
circle_analyzer.py

Detects concentric circle pairs (inner circle inside outer circle) in images,
filters out pairs that are out of focus (blurry), measures the area of each
circle, and outputs the inner/outer area ratio.

Usage:
    python3 circle_analyzer.py <input_path> [options]

    <input_path> can be a single image file or a directory of images.

Key options (all have sensible defaults, tune as needed for your images):
    --min-radius, --max-radius   Expected circle radius range in pixels
    --min-dist                   Minimum distance between detected circle centers
    --param1, --param2           Hough transform sensitivity params (see OpenCV docs)
    --center-tol                 Max center offset (px) to consider two circles concentric
    --focus-threshold             Laplacian-variance cutoff below which a circle is "blurry"
    --outdir                     Where to write CSV + annotated images

Output:
    - results.csv: one row per accepted circle pair with areas + ratio
    - <image>_annotated.png: visual showing kept (green) vs rejected (red) pairs
"""

import cv2
import numpy as np
import argparse
import os
import csv
import sys
import glob

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def detect_circles(gray, dp, min_dist, param1, param2, min_radius, max_radius):
    """Run Hough Circle Transform, return list of (x, y, r) ints."""
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    circles = np.round(circles[0, :]).astype(int)
    return [(int(x), int(y), int(r)) for x, y, r in circles]


def focus_score(gray, x, y, r):
    """
    Sharpness metric for the region around a circle, using the variance of
    the Laplacian. Higher = sharper/more in-focus, lower = blurrier.
    """
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(gray.shape[1], x + r), min(gray.shape[0], y + r)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def group_concentric(circles, center_tol):
    """
    Pair up circles that share (approximately) the same center, matching
    each larger circle to the nearest smaller circle within center_tol.
    Returns list of (outer, inner) tuples, each a (x, y, r) tuple.
    Unpaired circles are ignored (need both an inner and outer to form a ratio).
    """
    circles_sorted = sorted(circles, key=lambda c: c[2], reverse=True)
    used = set()
    pairs = []
    for i, outer in enumerate(circles_sorted):
        if i in used:
            continue
        best_j, best_dist = None, None
        for j, inner in enumerate(circles_sorted):
            if j <= i or j in used:
                continue
            if inner[2] >= outer[2]:
                continue
            dist = float(np.hypot(outer[0] - inner[0], outer[1] - inner[1]))
            if dist <= center_tol and (best_dist is None or dist < best_dist):
                best_dist, best_j = dist, j
        if best_j is not None:
            pairs.append((outer, circles_sorted[best_j]))
            used.add(i)
            used.add(best_j)
    return pairs


def process_image(path, args, csv_writer):
    img = cv2.imread(path)
    if img is None:
        print(f"  [skip] could not read image: {path}")
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = detect_circles(
        gray,
        dp=args.dp,
        min_dist=args.min_dist,
        param1=args.param1,
        param2=args.param2,
        min_radius=args.min_radius,
        max_radius=args.max_radius,
    )
    pairs = group_concentric(circles, center_tol=args.center_tol)

    annotated = img.copy()
    kept = 0
    fname = os.path.basename(path)

    for idx, (outer, inner) in enumerate(pairs):
        f_outer = focus_score(gray, *outer)
        f_inner = focus_score(gray, *inner)
        is_focused = f_outer >= args.focus_threshold and f_inner >= args.focus_threshold

        color = (0, 200, 0) if is_focused else (0, 0, 220)  # green vs red (BGR)
        cv2.circle(annotated, (outer[0], outer[1]), outer[2], color, 2)
        cv2.circle(annotated, (inner[0], inner[1]), inner[2], color, 2)
        cv2.putText(
            annotated, str(idx), (outer[0] - 8, outer[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        if not is_focused:
            continue

        area_outer = np.pi * (outer[2] ** 2)
        area_inner = np.pi * (inner[2] ** 2)
        ratio = area_inner / area_outer if area_outer else 0.0

        csv_writer.writerow({
            "image": fname,
            "pair_index": idx,
            "outer_x": outer[0], "outer_y": outer[1], "outer_r": outer[2],
            "inner_x": inner[0], "inner_y": inner[1], "inner_r": inner[2],
            "outer_area": round(area_outer, 2),
            "inner_area": round(area_inner, 2),
            "inner_outer_ratio": round(ratio, 4),
            "outer_focus_score": round(f_outer, 1),
            "inner_focus_score": round(f_inner, 1),
        })
        kept += 1

    out_path = os.path.join(args.outdir, f"{os.path.splitext(fname)[0]}_annotated.png")
    cv2.imwrite(out_path, annotated)
    print(f"  {fname}: {len(circles)} circles detected, {len(pairs)} pairs, "
          f"{kept} kept as in-focus -> {out_path}")
    return kept


def gather_images(input_path):
    if os.path.isdir(input_path):
        files = []
        for ext in IMAGE_EXTS:
            files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(files)
    else:
        return [input_path]


def main():
    parser = argparse.ArgumentParser(description="Analyze concentric circle images.")
    parser.add_argument("input_path", help="Image file or directory of images")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--dp", type=float, default=1.2, help="Hough dp param")
    parser.add_argument("--min-dist", type=int, default=20, dest="min_dist",
                         help="Min distance between detected circle centers")
    parser.add_argument("--param1", type=float, default=50)
    parser.add_argument("--param2", type=float, default=30)
    parser.add_argument("--min-radius", type=int, default=5, dest="min_radius")
    parser.add_argument("--max-radius", type=int, default=0, dest="max_radius",
                         help="0 = no upper limit")
    parser.add_argument("--center-tol", type=float, default=10, dest="center_tol",
                         help="Max center offset (px) for two circles to count as concentric")
    parser.add_argument("--focus-threshold", type=float, default=100.0, dest="focus_threshold",
                         help="Laplacian variance cutoff; below this a circle is 'blurry'")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    images = gather_images(args.input_path)
    if not images:
        print(f"No images found at {args.input_path}")
        sys.exit(1)

    csv_path = os.path.join(args.outdir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "image", "pair_index",
            "outer_x", "outer_y", "outer_r",
            "inner_x", "inner_y", "inner_r",
            "outer_area", "inner_area", "inner_outer_ratio",
            "outer_focus_score", "inner_focus_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_kept = 0
        print(f"Processing {len(images)} image(s)...")
        for path in images:
            total_kept += process_image(path, args, writer)

    print(f"\nDone. {total_kept} in-focus circle pair(s) written to {csv_path}")


if __name__ == "__main__":
    main()
