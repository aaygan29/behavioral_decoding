"""Facial video loading.

Two feature families, kept separate on purpose:

``geometric``
    Action-unit intensities and landmark-derived measures. Interpretable, low
    dimensional, and directly comparable to the affect ratings in the behaviour
    block. This is the family you can defend in a paper.

``embedding``
    Vision-transformer embeddings of sampled frames. Higher ceiling, but the
    features are not interpretable and the encoder is a confound: two subjects
    can differ in embedding space because of lighting or identity rather than
    expression. Run the geometric arm as the control before believing the
    embedding arm.

The loader samples frames within each trial's time window and pools across
frames, because the model contract is one row per trial, not one row per frame.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..utils.progress import progress
from .base import FACE, BaseLoader, ModalityBlock


class FaceLoader(BaseLoader):
    """Turn per-trial facial video segments into trial-by-feature rows."""

    name = FACE

    def __init__(
        self,
        frames_per_trial: int = 8,
        pooling: str = "mean_std",
        encoder: Optional[object] = None,
    ) -> None:
        """
        Parameters
        ----------
        frames_per_trial:
            Frames sampled uniformly across the trial window.
        pooling:
            How to collapse frames to one row: ``"mean"``, ``"max"``, or
            ``"mean_std"``. ``mean_std`` keeps within-trial variability, which
            carries expression *dynamics* that a mean alone destroys.
        encoder:
            Object with an ``encode(frames) -> (n_frames, d)`` method. Defaults
            to :class:`behavioral_decoding.features.vit_encoder.ViTEncoder`.
        """
        if pooling not in {"mean", "max", "mean_std"}:
            raise ValueError("pooling must be one of 'mean', 'max', 'mean_std'")
        self.frames_per_trial = frames_per_trial
        self.pooling = pooling
        self._encoder = encoder

    @property
    def encoder(self) -> object:
        if self._encoder is None:
            from ..features.vit_encoder import ViTEncoder

            self._encoder = ViTEncoder()
        return self._encoder

    def _pool(self, frame_features: np.ndarray) -> np.ndarray:
        """Collapse ``(n_frames, d)`` to a single vector."""
        if self.pooling == "mean":
            return frame_features.mean(axis=0)
        if self.pooling == "max":
            return frame_features.max(axis=0)
        return np.concatenate([frame_features.mean(axis=0), frame_features.std(axis=0)])

    def _pooled_names(self, base_names: Sequence[str]) -> List[str]:
        if self.pooling == "mean_std":
            return [f"{n}_mean" for n in base_names] + [
                f"{n}_std" for n in base_names
            ]
        return [f"{n}_{self.pooling}" for n in base_names]

    def from_geometric(
        self,
        au_features: np.ndarray,
        subject_ids: Sequence,
        stimulus_ids: Sequence,
        feature_names: Optional[List[str]] = None,
        source: str = "arrays",
    ) -> ModalityBlock:
        """Build a block from precomputed AU or landmark features.

        ``au_features`` may be ``(n_trials, d)`` (already pooled) or
        ``(n_trials, n_frames, d)`` (pooled here).
        """
        arr = np.asarray(au_features, dtype=float)
        if arr.ndim == 3:
            base_names = feature_names or [f"au_{i:02d}" for i in range(arr.shape[2])]
            pooled = np.vstack([self._pool(trial) for trial in arr])
            names = self._pooled_names(base_names)
        elif arr.ndim == 2:
            pooled = arr
            names = feature_names or [f"au_{i:02d}" for i in range(arr.shape[1])]
        else:
            raise ValueError(
                f"expected (n_trials, d) or (n_trials, n_frames, d), got shape {arr.shape}"
            )

        return ModalityBlock(
            name=self.name,
            X=pooled,
            subject_ids=np.asarray(subject_ids),
            stimulus_ids=np.asarray(stimulus_ids),
            feature_names=list(names),
            provenance=self._provenance(
                source=source, family="geometric", pooling=self.pooling
            ),
        )

    def from_frames(
        self,
        frame_stacks: Sequence[np.ndarray],
        subject_ids: Sequence,
        stimulus_ids: Sequence,
        source: str = "frames",
    ) -> ModalityBlock:
        """Encode per-trial frame stacks with the vision transformer.

        ``frame_stacks[i]`` has shape ``(n_frames, H, W, 3)``.
        """
        rows: List[np.ndarray] = []
        for stack in progress(frame_stacks, desc="face trials", total=len(frame_stacks)):
            embeddings = self.encoder.encode(np.asarray(stack))
            rows.append(self._pool(np.asarray(embeddings, dtype=float)))

        X = np.vstack(rows)
        d = X.shape[1] // (2 if self.pooling == "mean_std" else 1)
        base_names = [f"vit_{i:03d}" for i in range(d)]

        return ModalityBlock(
            name=self.name,
            X=X,
            subject_ids=np.asarray(subject_ids),
            stimulus_ids=np.asarray(stimulus_ids),
            feature_names=self._pooled_names(base_names),
            provenance=self._provenance(
                source=source,
                family="embedding",
                pooling=self.pooling,
                encoder=getattr(self.encoder, "describe", lambda: "unknown")(),
            ),
        )

    def load(
        self,
        video_paths: Sequence[str],
        trial_windows: Sequence[Sequence[Tuple[float, float]]],
        subject_ids: Sequence,
        stimulus_ids: Sequence[Sequence],
    ) -> ModalityBlock:
        """Read videos, slice trial windows, and encode. Requires ``opencv-python``.

        ``trial_windows[i]`` is a list of ``(t_start_s, t_end_s)`` for video
        ``i``; ``stimulus_ids[i]`` is the matching list of stimulus labels.
        """
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "FaceLoader.load requires opencv-python. Install with "
                "`pip install '.[face]'`, or use from_frames / from_geometric."
            ) from exc

        stacks: List[np.ndarray] = []
        subj_out: List[object] = []
        stim_out: List[object] = []

        for vid_idx, path in enumerate(
            progress(video_paths, desc="videos", total=len(video_paths))
        ):
            cap = cv2.VideoCapture(path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            try:
                for (t0, t1), stim in zip(trial_windows[vid_idx], stimulus_ids[vid_idx]):
                    idxs = np.linspace(t0 * fps, t1 * fps, self.frames_per_trial).astype(int)
                    frames = []
                    for fi in idxs:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                        ok, frame = cap.read()
                        if ok:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if not frames:
                        continue
                    stacks.append(np.stack(frames))
                    subj_out.append(subject_ids[vid_idx])
                    stim_out.append(stim)
            finally:
                cap.release()

        if not stacks:
            raise ValueError("no frames could be read from the given videos")

        return self.from_frames(stacks, subj_out, stim_out, source="video")
