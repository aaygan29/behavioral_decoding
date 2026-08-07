"""Vision-transformer frame encoder.

Used for the facial-video block, and available for any other image-like input
(stimulus images, video thumbnails, rendered fMRI slices).

Honesty note, and it matters: if ``torch`` and ``transformers`` are missing this
class does **not** silently substitute something and call itself a ViT. It falls
back to a fixed random projection, sets ``backend="random_projection_fallback"``,
and every downstream provenance record says so. A fallback run is a smoke test
of the plumbing, never a result. :meth:`assert_real_encoder` exists so scripts
that produce reportable numbers can refuse to run on the fallback.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..utils.progress import progress

DEFAULT_MODEL = "google/vit-base-patch16-224-in21k"
FALLBACK_BACKEND = "random_projection_fallback"


class ViTEncoder:
    """Encode ``(n_frames, H, W, 3)`` uint8 frames into ``(n_frames, d)`` features."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch_size: int = 16,
        pooling: str = "cls",
        fallback_dim: int = 64,
        seed: int = 0,
        allow_fallback: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        pooling:
            ``"cls"`` takes the class token; ``"mean"`` averages patch tokens.
            Mean pooling is usually the safer default for faces, where the
            signal is spatially distributed rather than global.
        allow_fallback:
            If ``False``, a missing torch/transformers install raises instead of
            degrading to the random projection.
        """
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.pooling = pooling
        self.fallback_dim = fallback_dim
        self.seed = seed
        self.allow_fallback = allow_fallback

        self.backend = "uninitialised"
        self._model = None
        self._processor = None
        self._projection: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ setup

    def _ensure_backend(self, input_dim: Optional[int] = None) -> None:
        if self.backend != "uninitialised":
            return
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._processor = AutoImageProcessor.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
            self.backend = "transformers"
        except Exception as exc:  # noqa: BLE001 - any failure means no real encoder
            if not self.allow_fallback:
                raise RuntimeError(
                    f"ViTEncoder could not load {self.model_name!r} and allow_fallback=False: {exc}"
                ) from exc
            rng = np.random.default_rng(self.seed)
            if input_dim is not None:
                self._projection = rng.normal(
                    0.0, 1.0 / np.sqrt(input_dim), size=(input_dim, self.fallback_dim)
                )
            self.backend = FALLBACK_BACKEND

    @property
    def is_real_encoder(self) -> bool:
        return self.backend == "transformers"

    def assert_real_encoder(self) -> None:
        """Raise if running on the fallback. Call this before reporting results."""
        self._ensure_backend()
        if not self.is_real_encoder:
            raise RuntimeError(
                f"ViTEncoder is running on the {self.backend} backend, which produces "
                "meaningless features. Install torch and transformers "
                "(`pip install '.[vision]'`) before generating reportable "
                "numbers."
            )

    def describe(self) -> Dict[str, object]:
        """Provenance record. Gets embedded in every block this encoder touches."""
        self._ensure_backend()
        return {
            "encoder": "ViTEncoder",
            "backend": self.backend,
            "model_name": self.model_name if self.is_real_encoder else None,
            "pooling": self.pooling,
            "reportable": self.is_real_encoder,
        }

    # ----------------------------------------------------------------- encode

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """Encode a stack of frames.

        Parameters
        ----------
        frames:
            ``(n_frames, H, W, 3)`` uint8 or float array.

        Returns
        -------
        ``(n_frames, d)`` float array.
        """
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"expected (n_frames, H, W, 3), got shape {frames.shape}"
            )
        flat_dim = int(np.prod(frames.shape[1:]))
        self._ensure_backend(input_dim=flat_dim)

        if self.backend == FALLBACK_BACKEND:
            return self._encode_fallback(frames, flat_dim)
        return self._encode_transformers(frames)

    def _encode_fallback(self, frames: np.ndarray, flat_dim: int) -> np.ndarray:
        if self._projection is None or self._projection.shape[0] != flat_dim:
            rng = np.random.default_rng(self.seed)
            self._projection = rng.normal(
                0.0, 1.0 / np.sqrt(flat_dim), size=(flat_dim, self.fallback_dim)
            )
        flat = frames.reshape(frames.shape[0], -1).astype(np.float32) / 255.0
        return flat @ self._projection

    def _encode_transformers(self, frames: np.ndarray) -> np.ndarray:
        import torch

        outputs: List[np.ndarray] = []
        n_batches = int(np.ceil(len(frames) / self.batch_size))
        batches = range(0, len(frames), self.batch_size)
        with torch.no_grad():
            for start in progress(list(batches), desc="ViT encode", total=n_batches):
                batch = list(frames[start : start + self.batch_size])
                inputs = self._processor(images=batch, return_tensors="pt").to(self.device)
                hidden = self._model(**inputs).last_hidden_state  # (b, tokens, d)
                if self.pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    pooled = hidden[:, 1:, :].mean(dim=1)
                outputs.append(pooled.cpu().numpy())
        return np.vstack(outputs)
