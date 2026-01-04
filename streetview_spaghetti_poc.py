#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
from playwright.sync_api import sync_playwright


# =========================
# URL 正規化（Street View に入りやすく）
# =========================
def normalize_streetview_url(url: str) -> str:
    try:
        u = urlparse(url)
        qs = parse_qs(u.query)
        if qs.get("map_action", [""])[0] == "pano" and "viewpoint" in qs:
            vp = qs["viewpoint"][0]
            return f"https://www.google.com/maps?layer=c&cbll={vp}"
    except Exception:
        pass
    return url


# =========================
# Street View が出るまで待つ
# =========================
def wait_until_streetview_like(page, timeout_s=60) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            html = page.content()
            if "Street View" in html or "ストリートビュー" in html:
                return True
        except Exception:
            pass

        try:
            canv = page.locator("canvas")
            for i in range(min(canv.count(), 10)):
                box = canv.nth(i).bounding_box()
                if box and box["width"] * box["height"] > 600 * 350:
                    return True
        except Exception:
            pass

        time.sleep(1.0)
    return False


# =========================
# Street View canvas にフォーカス
# =========================
def focus_streetview_canvas(page) -> bool:
    try:
        canv = page.locator("canvas")
        best_box = None
        best_area = 0
        for i in range(min(canv.count(), 10)):
            box = canv.nth(i).bounding_box()
            if box:
                area = box["width"] * box["height"]
                if area > best_area:
                    best_area = area
                    best_box = box
        if not best_box or best_area < 600 * 350:
            return False

        cx = int(best_box["x"] + best_box["width"] / 2)
        cy = int(best_box["y"] + best_box["height"] / 2)
        page.mouse.click(cx, cy)
        time.sleep(0.5)
        return True
    except Exception:
        return False


# =========================
# 回転操作
# =========================
def drag_rotate(page, dx: int, duration_ms: int = 220) -> bool:
    if not focus_streetview_canvas(page):
        return False

    vp = page.viewport_size
    cx = int(vp["width"] * 0.60)
    cy = int(vp["height"] * 0.55)

    page.mouse.move(cx, cy)
    page.mouse.down()

    steps = max(8, duration_ms // 25)
    for i in range(1, steps + 1):
        page.mouse.move(cx + int(dx * i / steps), cy)
        time.sleep(duration_ms / steps / 1000)

    page.mouse.up()
    return True


# =========================
# 初回のみ上向き補正
# =========================
def nudge_pitch_up(page, dy: int = 80, duration_ms: int = 180):
    if not focus_streetview_canvas(page):
        return
    vp = page.viewport_size
    cx = int(vp["width"] * 0.60)
    cy = int(vp["height"] * 0.55)

    page.mouse.move(cx, cy)
    page.mouse.down()
    steps = max(8, duration_ms // 25)
    for i in range(1, steps + 1):
        page.mouse.move(cx, cy + int(dy * i / steps))
        time.sleep(duration_ms / steps / 1000)
    page.mouse.up()
    time.sleep(0.4)


# =========================
# 未描画（真っ暗）チェック
# =========================
def is_blank_like(img: np.ndarray) -> bool:
    h, w = img.shape[:2]
    roi = img[int(h * 0.2):int(h * 0.7), int(w * 0.2):int(w * 0.8)]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return gray.std() < 5.0


# =========================
# メモリでスクショ（保存前判定用）
# =========================
def screenshot_when_ready_bytes(page, timeout_s=25) -> bytes:
    start = time.time()
    last = None
    while True:
        b = page.screenshot(type="png")
        last = b
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and not is_blank_like(img):
            return b
        if time.time() - start > timeout_s:
            return last
        time.sleep(0.8)


# =========================
# 電線スコア
# =========================
def spaghetti_score(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    roi = img[: int(h * 0.6), :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    edge_density = edges.mean() / 255.0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=int(w * 0.06), maxLineGap=10)
    line_count = 0 if lines is None else len(lines)

    return 4.0 * edge_density + 0.02 * line_count


# =========================
# main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out-dir", default="poc_out")
    ap.add_argument("--headful", action="store_true")
    ap.add_argument("--rot-steps", type=int, default=36)
    ap.add_argument("--rot-drag", type=int, default=220)
    ap.add_argument("--walk-steps", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=1.1)
    ap.add_argument("--min-score", type=float, default=1.2)
    ap.add_argument("--max-save", type=int, default=50)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    shots_dir = out_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headful)
        context = browser.new_context(viewport={"width": 1280, "height": 720})
        page = context.new_page()

        page.goto(normalize_streetview_url(args.url), wait_until="domcontentloaded")

        if not wait_until_streetview_like(page):
            raise RuntimeError("Street View に入れませんでした")

        focus_streetview_canvas(page)

        # ★初回のみ上向き補正
        nudge_pitch_up(page, dy=-100)
        nudge_pitch_up(page, dy=-100)

        for w in range(max(1, args.walk_steps + 1)):
            for r in range(args.rot_steps):
                png = screenshot_when_ready_bytes(page)
                img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                score = spaghetti_score(img)

                print(f"[SCAN] w{w:02d} r{r:03d}  score={score:.2f}")

                if score >= args.min_score:
                    path = shots_dir / f"w{w:02d}_r{r:03d}_s{score:.2f}.png"
                    path.write_bytes(png)
                    saved_count += 1

                    print(
                        f"[SAVE] {saved_count:3d} / {args.max_save:3d}  "
                        f"{path.name}  score={score:.2f}"
                    )

                    if saved_count >= args.max_save:
                        print(f"[DONE] reached max-save={args.max_save}")
                        context.close()
                        browser.close()
                        return

                drag_rotate(page, dx=args.rot_drag)
                time.sleep(args.sleep)

        context.close()
        browser.close()


if __name__ == "__main__":
    main()

