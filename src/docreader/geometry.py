"""Find the document in a photo and flatten it, the way scanner apps do.

Edges -> largest four-cornered outline -> perspective warp to a rectangle.
A barcode or MRZ photographed at an angle reads far better once flattened.
"""
from __future__ import annotations

import cv2
import numpy as np


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def find_document(img: np.ndarray, min_area: float = 0.15) -> np.ndarray | None:
    """Corners of the largest convex quadrilateral covering at least min_area of the
    image, or None (a flat scan or a page that fills the frame)."""
    h, w = img.shape[:2]
    scale = 800 / max(h, w)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    grey = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    best, best_area = None, min_area * small.shape[0] * small.shape[1]
    for lo, hi in ((30, 90), (50, 150), (10, 50)):
        edges = cv2.dilate(cv2.Canny(grey, lo, hi), np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and area > best_area:
                best, best_area = approx, area
        if best is not None:
            break
    if best is None:
        return None
    quad = order_corners(best) / scale
    # A quad that is just the image border means there is nothing to flatten.
    if cv2.contourArea(quad) > 0.97 * h * w:
        return None
    return quad


def flatten(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rectify(img: np.ndarray) -> tuple[np.ndarray, bool]:
    quad = find_document(img)
    if quad is None:
        return img, False
    return flatten(img, quad), True
