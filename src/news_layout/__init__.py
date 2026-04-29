"""
news-layout-classifier: YouTube news layout classifier (left / right / other).

Usage (no install needed):
    from news_layout import NewsLayoutClassifier
    clf = NewsLayoutClassifier()
    label = clf.classify("image.jpg")

CLI after install:
    news-layout bench LEFT RIGHT OTHER --rounds 5 --verbose
    news-layout batch INPUT OUTPUT
    news-layout single IMAGE --templates templates.npz
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_SIZE = 128
PIXEL_TOL = 0.05
MIN_MATCH = 0.30
DEFAULT_ACTIVE_COLS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    97, 100,
    120, 121, 122, 123, 124, 125, 126, 127,
]
DEFAULT_TEMPLATE = Path(__file__).parent / "templates" / "news_layout_templates.npz"


def _match_score(
    img: np.ndarray,
    tpl: np.ndarray,
    col_mask: np.ndarray | None = None,
    pixel_tol: float = PIXEL_TOL,
) -> float:
    """Fraction of pixels where |img - tpl| < pixel_tol, over col_mask."""
    diff = np.abs(img - tpl)
    if col_mask is not None:
        diff = diff[:, col_mask]
    return float((diff < pixel_tol).mean())


class NewsLayoutClassifier:
    """
    Pixel-wise match-ratio news layout classifier.

    Usage:
        clf = NewsLayoutClassifier()          # auto-loades built-in templates
        clf = NewsLayoutClassifier(custom_params...)  # override defaults
        label = clf.classify("image.jpg")     # left | right | other
        label, scores = clf.classify_with_scores("image.jpg")
    """

    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        pixel_tol: float = PIXEL_TOL,
        min_match: float = MIN_MATCH,
        active_cols: list[int] | None = DEFAULT_ACTIVE_COLS,
        template: Path | str | None = None,
    ):
        self.size = size
        self.pixel_tol = pixel_tol
        self.min_match = min_match
        self._active_cols = active_cols
        self._col_mask: np.ndarray | None = None
        self._tpl_left: np.ndarray | None = None
        self._tpl_right: np.ndarray | None = None
        self._trained = False

        if active_cols is not None:
            self._col_mask = np.zeros(size, dtype=bool)
            self._col_mask[active_cols] = True

        # Auto-load built-in templates if they exist
        tpl_path = Path(template) if template else DEFAULT_TEMPLATE
        if tpl_path.exists():
            self.load_templates(tpl_path)
        else:
            logger.debug(
                "Classifier ready (size=%d, px_tol=%s, min_match=%s, cols=%s)",
                size, pixel_tol, min_match, active_cols,
            )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        left_dir: Path,
        right_dir: Path,
        other_dir: Path | None = None,
    ) -> dict[str, int]:
        """Build templates from labelled directories."""
        def load_dir(d: Path) -> np.ndarray:
            files = [
                f for f in d.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            imgs = [
                np.array(
                    Image.open(f).convert("RGB").resize(
                        (self.size, self.size), Image.LANCZOS
                    ),
                    dtype=np.float64,
                )
                / 255.0
                for f in files
            ]
            return np.array(imgs)

        logger.info("Training from: L=%s R=%s", left_dir, right_dir)
        L = load_dir(left_dir)
        R = load_dir(right_dir)
        self._tpl_left = L.mean(axis=0)
        self._tpl_right = R.mean(axis=0)
        self._trained = True
        return {"left": len(L), "right": len(R)}

    def train_from_lists(
        self,
        left_paths: list[Path],
        right_paths: list[Path],
        other_paths: list[Path] | None = None,
    ) -> dict[str, int]:
        """Build templates from explicit path lists."""
        L = np.array([
            np.array(
                Image.open(p).convert("RGB").resize((self.size, self.size), Image.LANCZOS),
                dtype=np.float64,
            )
            / 255.0
            for p in left_paths
        ])
        R = np.array([
            np.array(
                Image.open(p).convert("RGB").resize((self.size, self.size), Image.LANCZOS),
                dtype=np.float64,
            )
            / 255.0
            for p in right_paths
        ])
        self._tpl_left = L.mean(axis=0)
        self._tpl_right = R.mean(axis=0)
        self._trained = True
        return {"left": len(L), "right": len(R)}

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify(self, path: Path | str) -> Literal["left", "right", "other"]:
        """Classify a single image. Returns 'left', 'right', or 'other'."""
        if not self._trained:
            raise RuntimeError("Classifier not trained. Call train() or load_templates() first.")
        path = Path(path)
        img = (
            np.array(
                Image.open(path).convert("RGB").resize((self.size, self.size), Image.LANCZOS),
                dtype=np.float64,
            )
            / 255.0
        )
        sc_L = _match_score(img, self._tpl_left, self._col_mask, self.pixel_tol)
        sc_R = _match_score(img, self._tpl_right, self._col_mask, self.pixel_tol)

        logger.debug("%s: match_L=%.3f match_R=%.3f", path.name, sc_L, sc_R)

        if sc_L < self.min_match and sc_R < self.min_match:
            return "other"
        return "left" if sc_L > sc_R else "right"

    def classify_batch(
        self, paths: list[Path | str]
    ) -> list[Literal["left", "right", "other"]]:
        """Classify a list of image paths."""
        return [self.classify(p) for p in paths]

    def classify_with_scores(
        self, path: Path | str
    ) -> tuple[Literal["left", "right", "other"], dict[str, float]]:
        """Classify and also return raw match scores."""
        if not self._trained:
            raise RuntimeError("Classifier not trained.")
        path = Path(path)
        img = (
            np.array(
                Image.open(path).convert("RGB").resize((self.size, self.size), Image.LANCZOS),
                dtype=np.float64,
            )
            / 255.0
        )
        sc_L = _match_score(img, self._tpl_left, self._col_mask, self.pixel_tol)
        sc_R = _match_score(img, self._tpl_right, self._col_mask, self.pixel_tol)
        scores = {"left": sc_L, "right": sc_R}
        return self.classify(path), scores

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save_templates(self, path: Path) -> None:
        """Save L/R templates to a .npz file."""
        np.savez(path, tpl_left=self._tpl_left, tpl_right=self._tpl_right)
        logger.info("Templates saved: %s", path)

    def load_templates(self, path: Path) -> None:
        """Load pre-trained L/R templates from .npz file."""
        data = np.load(path)
        self._tpl_left = data["tpl_left"]
        self._tpl_right = data["tpl_right"]
        self._trained = True
        logger.info("Templates loaded: %s", path)
