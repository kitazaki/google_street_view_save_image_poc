import cv2
import numpy as np
from pathlib import Path

def spaghetti_debug(img: np.ndarray, outdir="debug"):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True)

    h, w = img.shape[:2]

    # 1. ROI
    roi = img[: int(h * 0.6), :]
    cv2.imwrite(str(outdir / "01_roi.png"), roi)

    # 2. グレースケール
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(outdir / "02_gray.png"), gray)

    # 3. Canny
    edges = cv2.Canny(gray, 60, 160)
    cv2.imwrite(str(outdir / "03_edges.png"), edges)

    # 4. HoughLinesP 可視化
    vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        60,
        minLineLength=int(w * 0.06),
        maxLineGap=10,
    )

    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    cv2.imwrite(str(outdir / "04_lines.png"), vis)

    edge_density = edges.mean() / 255.0
    score = 4.0 * edge_density + 0.02 * line_count

    print(f"edge_density={edge_density:.4f}")
    print(f"line_count={line_count}")
    print(f"score={score:.2f}")

img = cv2.imread("sample.png")
spaghetti_debug(img, outdir="debug_sample")

