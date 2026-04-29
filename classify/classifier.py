"""
Template-based news layout classifier.

Uses normalized cross-correlation with class-mean templates:
  - "left"  : anchor on LEFT  (UI frame on RIGHT)
  - "right" : anchor on RIGHT (UI frame on LEFT)
  - "other" : generic news footage (weather graphics, motion graphics)

Templates are built from training images at SIZE x SIZE resolution.
A tolerance gap between best and second-best score triggers "other".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
DEFAULT_SIZE = 128            # resolution for template matching
TOLERANCE_MARGIN = 0.05       # min score gap (best - 2nd) to avoid "other"
MIN_TPL_MATCH = 0.15          # absolute score below this -> "other"
# ---------------------------------------------------------------------------


def _corr(img_flat: np.ndarray, tpl_flat: np.ndarray) -> float:
    """Normalized cross-correlation between two flat arrays."""
    i_c = img_flat - img_flat.mean()
    t_c = tpl_flat - tpl_flat.mean()
    num = np.dot(i_c, t_c)
    denom = np.sqrt(np.dot(i_c, i_c) * np.dot(t_c, t_c)) + 1e-12
    return float(num / denom)


def _load_gray_rgb(path: Path, size: int) -> np.ndarray:
    """Load image as float64 [0,1] RGB."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float64) / 255.0


# ---------------------------------------------------------------------------
# Main classifier
# ---------------------------------------------------------------------------
class NewsLayoutClassifier:
    """
    Template-based news layout classifier.

    Classifies a news broadcast frame into:
      - "left"  : anchor on LEFT  (UI frame on RIGHT)
      - "right" : anchor on RIGHT (UI frame on LEFT)
      - "other" : no fixed studio frame detected
    """

    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        tolerance_margin: float = TOLERANCE_MARGIN,
        min_template_match: float = MIN_TPL_MATCH,
    ):
        self.size = size
        self.tolerance_margin = tolerance_margin
        self.min_template_match = min_template_match
        self._tpl_left:  np.ndarray | None = None
        self._tpl_right: np.ndarray | None = None
        self._tpl_other: np.ndarray | None = None
        self._trained = False
        logger.debug(f"Classifier ready (size={size})")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        left_dir: Path,
        right_dir: Path,
        other_dir: Path,
    ) -> dict[str, int]:
        """
        Build class templates from labelled image directories.

        Parameters
        ----------
        left_dir, right_dir, other_dir : Path
            Directories containing .jpg/.png images for each class.

        Returns
        -------
        dict with keys 'left', 'right', 'other' mapping to image counts.
        """
        def load_dir(d: Path) -> list[np.ndarray]:
            files = [f for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            return [_load_gray_rgb(f, self.size) for f in files]

        logger.info(f"Training from: L={left_dir} R={right_dir} O={other_dir}")
        L = load_dir(left_dir)
        R = load_dir(right_dir)
        O = load_dir(other_dir)

        self._tpl_left  = np.array(L).mean(axis=0)
        self._tpl_right = np.array(R).mean(axis=0)
        self._tpl_other = np.array(O).mean(axis=0)
        self._trained = True

        counts = {"left": len(L), "right": len(R), "other": len(O)}
        logger.info(f"Training done: {counts}")
        return counts

    def train_from_lists(
        self,
        left_paths: list[Path],
        right_paths: list[Path],
        other_paths: list[Path],
    ) -> dict[str, int]:
        """Build templates from explicit lists of image paths."""
        def from_paths(paths: list[Path]) -> np.ndarray:
            return np.array([_load_gray_rgb(p, self.size) for p in paths])

        L = from_paths(left_paths)
        R = from_paths(right_paths)
        O = from_paths(other_paths)
        self._tpl_left  = L.mean(axis=0)
        self._tpl_right = R.mean(axis=0)
        self._tpl_other = O.mean(axis=0)
        self._trained = True
        counts = {"left": len(L), "right": len(R), "other": len(O)}
        logger.info(f"Training done: {counts}")
        return counts

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def classify(self, path: Path | str) -> Literal["left", "right", "other"]:
        """
        Classify a single image.

        Raises
        ------
        RuntimeError if classifier has not been trained.
        """
        if not self._trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        path = Path(path)
        img = _load_gray_rgb(path, self.size)

        sl = _corr(img.reshape(-1), self._tpl_left.reshape(-1))
        sr = _corr(img.reshape(-1), self._tpl_right.reshape(-1))
        so = _corr(img.reshape(-1), self._tpl_other.reshape(-1))

        scores = {"left": sl, "right": sr, "other": so}
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]
        sorted_scores = sorted(scores.values(), reverse=True)

        logger.debug(
            f"{path.name}: left={sl:.3f} right={sr:.3f} other={so:.3f} -> {best_label}"
        )

        # Tolerance: if the gap between best and 2nd-best is too small -> "other"
        if len(sorted_scores) >= 2:
            gap = sorted_scores[0] - sorted_scores[1]
            if gap < self.tolerance_margin:
                return "other"

        # Absolute threshold
        if best_score < self.min_template_match:
            return "other"

        return best_label

    def classify_batch(
        self, paths: list[Path | str]
    ) -> list[Literal["left", "right", "other"]]:
        """Classify a list of image paths."""
        return [self.classify(p) for p in paths]

    def classify_with_scores(
        self, path: Path | str
    ) -> tuple[Literal["left", "right", "other"], dict[str, float]]:
        """Classify and also return raw template match scores."""
        if not self._trained:
            raise RuntimeError("Classifier not trained.")

        path = Path(path)
        img = _load_gray_rgb(path, self.size)
        sl = _corr(img.reshape(-1), self._tpl_left.reshape(-1))
        sr = _corr(img.reshape(-1), self._tpl_right.reshape(-1))
        so = _corr(img.reshape(-1), self._tpl_other.reshape(-1))
        scores = {"left": sl, "right": sr, "other": so}
        label = self.classify(path)
        return label, scores

    # ------------------------------------------------------------------
    # Save / load templates
    # ------------------------------------------------------------------
    def save_templates(self, path: Path) -> None:
        """Save templates to a .npz file."""
        np.savez(
            path,
            tpl_left=self._tpl_left,
            tpl_right=self._tpl_right,
            tpl_other=self._tpl_other,
        )
        logger.info(f"Templates saved to {path}")

    def load_templates(self, path: Path) -> None:
        """Load pre-trained templates from .npz file."""
        data = np.load(path)
        self._tpl_left  = data["tpl_left"]
        self._tpl_right = data["tpl_right"]
        self._tpl_other = data["tpl_other"]
        self._trained = True
        logger.info(f"Templates loaded from {path}")
