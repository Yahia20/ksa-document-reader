"""Turn a clean render into what a business actually receives: a scan or a phone photo.

    scan   slight skew, lower resolution, paper tint, JPEG
    photo  document on a desk, perspective, uneven light, blur, sensor noise, JPEG
    hard   stronger perspective and blur, a shadow, a glare spot, low resolution
"""
from __future__ import annotations

import random

import cv2
import numpy as np

LEVELS = ("scan", "photo", "hard")


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def _desk(h: int, w: int, rng: np.random.Generator, r: random.Random) -> np.ndarray:
    base = np.array([r.randint(40, 200), r.randint(40, 190), r.randint(40, 180)], np.float32)
    small = rng.normal(0, 18, (max(2, h // 40), max(2, w // 40))).astype(np.float32)
    tex = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]
    grain = rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    return np.clip(base + tex + grain, 0, 255).astype(np.uint8)


def _light(img: np.ndarray, lo: float, hi: float, r: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    gx, gy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    a, b = r.uniform(-1, 1), r.uniform(-1, 1)
    g = a * gx + b * gy
    g = (g - g.min()) / (np.ptp(g) + 1e-6)
    mult = lo + (hi - lo) * g
    tint = np.array([r.uniform(.92, 1.0), r.uniform(.95, 1.0), r.uniform(.95, 1.05)])
    out = img.astype(np.float32) * mult[..., None] * tint
    return np.clip(out, 0, 255).astype(np.uint8)


def scan(img: np.ndarray, r: random.Random, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    scale = r.uniform(0.6, 0.85)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    pad = int(0.04 * max(h, w))
    paper = np.full((h + 2 * pad, w + 2 * pad, 3), r.randint(232, 250), np.uint8)
    paper[pad:pad + h, pad:pad + w] = img
    M = cv2.getRotationMatrix2D((paper.shape[1] / 2, paper.shape[0] / 2), r.uniform(-1.5, 1.5), 1.0)
    paper = cv2.warpAffine(paper, M, (paper.shape[1], paper.shape[0]), borderValue=(245, 245, 245))
    paper = cv2.GaussianBlur(paper, (0, 0), r.uniform(0.3, 0.7))
    paper = _noise(paper, r.uniform(2, 5), rng)
    return _jpeg(paper, r.randint(70, 90))


def photo(img: np.ndarray, r: random.Random, rng: np.random.Generator, hard: bool = False) -> np.ndarray:
    h, w = img.shape[:2]
    # Document size in the photo. Pages are rendered at ~150 dpi and cards at ~254 dpi,
    # so pages keep most of their width and cards may shrink further.
    if h > 1.2 * w:  # A4 page or till receipt
        scale = r.uniform(0.72, 0.85) if hard else r.uniform(0.85, 1.0)
    else:
        target_w = r.randint(750, 950) if hard else r.randint(1000, 1400)
        scale = target_w / w
    target_w = int(w * scale)
    img = cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    margin = int(0.12 * max(h, w))
    H, W = h + 2 * margin, w + 2 * margin
    canvas = _desk(H, W, rng, r)

    jitter = r.uniform(0.07, 0.13) if hard else r.uniform(0.02, 0.07)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[margin + r.uniform(-jitter, jitter) * w, margin + r.uniform(-jitter, jitter) * h]
                      for _ in range(4)])
    dst += np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    angle = r.uniform(-12, 12) if hard else r.uniform(-7, 7)
    c = np.float32([W / 2, H / 2])
    rot = cv2.getRotationMatrix2D(tuple(c), angle, 1.0)
    dst = (np.hstack([dst, np.ones((4, 1), np.float32)]) @ rot.T).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    mask = cv2.GaussianBlur(mask, (0, 0), 1.2)[..., None] / 255.0
    out = (warped * mask + canvas * (1 - mask)).astype(np.uint8)

    out = _light(out, r.uniform(0.55, 0.75) if hard else r.uniform(0.72, 0.9), r.uniform(1.0, 1.12), r)
    if hard:
        # Shadow: a dark soft polygon across part of the page.
        sh = np.zeros((H, W), np.uint8)
        pts = np.int32([[r.randint(0, W), r.randint(0, H)] for _ in range(4)])
        cv2.fillPoly(sh, [cv2.convexHull(pts)], 255)
        sh = cv2.GaussianBlur(sh, (0, 0), 25)[..., None] / 255.0
        out = (out * (1 - 0.35 * sh)).astype(np.uint8)
        # Glare: a bright soft spot.
        gl = np.zeros((H, W), np.float32)
        cv2.circle(gl, (r.randint(0, W), r.randint(0, H)), r.randint(40, 120), 1.0, -1)
        gl = cv2.GaussianBlur(gl, (0, 0), 30)[..., None]
        out = np.clip(out + 140 * gl, 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (0, 0), r.uniform(1.0, 1.8) if hard else r.uniform(0.5, 1.1))
    out = _noise(out, r.uniform(4, 9) if hard else r.uniform(2, 6), rng)
    return _jpeg(out, r.randint(45, 65) if hard else r.randint(60, 85))


def degrade(img: np.ndarray, level: str, seed: int) -> np.ndarray:
    r = random.Random(seed)
    rng = np.random.default_rng(seed)
    if level == "scan":
        return scan(img, r, rng)
    if level == "photo":
        return photo(img, r, rng)
    if level == "hard":
        return photo(img, r, rng, hard=True)
    raise ValueError(level)
