"""
Template-based news layout classifier — pixel-wise match ratio (simplest version).

For each pixel in active columns:
    if |test[p] - template[p]| < pixel_tol → count as "matching"

Score = fraction of matching pixels (higher = better match).
Classification: higher score wins, but if both scores < min_match → other.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_SIZE = 128
PIXEL_TOL = 0.05       # |diff| < this → match (in [0,1] space)
MIN_MATCH = 0.30       # score must exceed this to NOT be "other"


def _match_score(img: np.ndarray, tpl: np.ndarray, col_mask: np.ndarray | None = None,
                 pixel_tol: float = PIXEL_TOL) -> float:
    """
    Fraction of pixels where |img - tpl| < pixel_tol, evaluated over col_mask.
    Returns value in [0, 1].
    """
    diff = np.abs(img - tpl)
    if col_mask is not None:
        diff = diff[:, col_mask]
    return (diff < pixel_tol).mean()


class NewsLayoutClassifier:
    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        pixel_tol: float = PIXEL_TOL,
        min_match: float = MIN_MATCH,
        active_cols: list[int] | None = None,
    ):
        self.size = size
        self.pixel_tol = pixel_tol
        self.min_match = min_match
        self._active_cols = active_cols
        self._col_mask: np.ndarray | None = None
        self._tpl_left:  np.ndarray | None = None
        self._tpl_right: np.ndarray | None = None
        self._trained = False

        if active_cols is not None:
            self._col_mask = np.zeros(size, dtype=bool)
            self._col_mask[active_cols] = True

        logger.debug(f"Classifier ready (size={size}, px_tol={pixel_tol}, "
                     f"min_match={min_match}, cols={active_cols})")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        left_dir: Path,
        right_dir: Path,
        other_dir: Path | None = None,
    ) -> dict[str, int]:
        def load_dir(d: Path) -> np.ndarray:
            files = [f for f in d.iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            imgs = [np.array(Image.open(f).convert("RGB").resize(
                (self.size, self.size), Image.LANCZOS), dtype=np.float64) / 255.0
                    for f in files]
            return np.array(imgs)

        logger.info(f"Training from: L={left_dir} R={right_dir}")
        L = load_dir(left_dir)
        R = load_dir(right_dir)
        self._tpl_left  = L.mean(axis=0)
        self._tpl_right = R.mean(axis=0)
        self._trained = True
        return {"left": len(L), "right": len(R)}

    def train_from_lists(
        self,
        left_paths: list[Path],
        right_paths: list[Path],
        other_paths: list[Path] | None = None,
    ) -> dict[str, int]:
        L = np.array([
            np.array(Image.open(p).convert("RGB").resize((self.size, self.size), Image.LANCZOS),
                     dtype=np.float64) / 255.0 for p in left_paths
        ])
        R = np.array([
            np.array(Image.open(p).convert("RGB").resize((self.size, self.size), Image.LANCZOS),
                     dtype=np.float64) / 255.0 for p in right_paths
        ])
        self._tpl_left  = L.mean(axis=0)
        self._tpl_right = R.mean(axis=0)
        self._trained = True
        return {"left": len(L), "right": len(R)}

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify(self, path: Path | str) -> Literal["left", "right", "other"]:
        if not self._trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        path = Path(path)
        img = (np.array(Image.open(path).convert("RGB").resize(
            (self.size, self.size), Image.LANCZOS), dtype=np.float64) / 255.0)

        sc_L = _match_score(img, self._tpl_left,  self._col_mask, self.pixel_tol)
        sc_R = _match_score(img, self._tpl_right, self._col_mask, self.pixel_tol)

        logger.debug(f"{path.name}: match_L={sc_L:.3f} match_R={sc_R:.3f}")

        if sc_L < self.min_match and sc_R < self.min_match:
            return "other"
        return "left" if sc_L > sc_R else "right"

    def classify_batch(
        self, paths: list[Path | str]
    ) -> list[Literal["left", "right", "other"]]:
        return [self.classify(p) for p in paths]

    def classify_with_scores(
        self, path: Path | str
    ) -> tuple[Literal["left", "right", "other"], dict[str, float]]:
        if not self._trained:
            raise RuntimeError("Classifier not trained.")
        path = Path(path)
        img = (np.array(Image.open(path).convert("RGB").resize(
            (self.size, self.size), Image.LANCZOS), dtype=np.float64) / 255.0)
        sc_L = _match_score(img, self._tpl_left,  self._col_mask, self.pixel_tol)
        sc_R = _match_score(img, self._tpl_right, self._col_mask, self.pixel_tol)
        scores = {"left": sc_L, "right": sc_R}
        label = self.classify(path)
        return label, scores

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save_templates(self, path: Path) -> None:
        np.savez(path, tpl_left=self._tpl_left, tpl_right=self._tpl_right)
        logger.info(f"Templates saved to {path}")

    def load_templates(self, path: Path) -> None:
        data = np.load(path)
        self._tpl_left  = data["tpl_left"]
        self._tpl_right = data["tpl_right"]
        self._trained = True
        logger.info(f"Templates loaded from {path}")
