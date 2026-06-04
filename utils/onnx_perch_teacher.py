"""
ONNX-based PerchTeacher for Perch v2 knowledge distillation.

The training script uses a PyTorch-native PerchTeacher that requires:
    src/model/perch_pytorch.py    (PerchNet class)
    weights/perch_jax_backbone/perch_backbone_params.pkl

Neither is available in our codebase. This class uses ONNX runtime with
CUDA I/O binding instead — GPU-resident tensors, no CPU roundtrip, same API.

Verified: cosine similarity > 0.9999 vs PyTorch reference (Hengck's benchmark,
see https://www.kaggle.com/competitions/birdclef-2026/discussion/...).

Usage:
    from onnx_perch_teacher import PerchTeacher
    perch = PerchTeacher(onnx_path="path/to/perch_v2_no_dft.onnx", device_str="cuda:0")
    emb = perch.embed(waveforms_5s)   # (B, 160000) → (B, 1536) on same GPU
"""

import os
from pathlib import Path

import numpy as np
import torch

PERCH_EMBED_DIM = 1536


class PerchTeacher:
    """Frozen Perch v2 via ONNX with CUDA I/O binding.

    ONNX Perch v2 teacher for knowledge distillation. Same API:
        embed(waveforms_5s: (B, 160000) float32) -> (B, 1536) float32 on same device.

    Why I/O binding: standard onnxruntime CUDA execution still does
    host-device copies for inputs/outputs. iobinding pins GPU pointers
    so the data never leaves VRAM. Critical for training-time speed.
    """

    _DEFAULT_ONNX_PATH = os.environ.get(
        "PERCH_ONNX_PATH",
        "/mnt/local-scratch/perch-v2-no-dft-onnx/perch_v2_no_dft.onnx",
    )

    def __init__(self, onnx_path=None, device_str="cuda:0", fp16=False):
        import onnxruntime as ort

        if onnx_path is None:
            onnx_path = self._DEFAULT_ONNX_PATH
        onnx_path = Path(onnx_path)
        assert onnx_path.exists(), f"Perch ONNX not found at {onnx_path}"

        self.device = torch.device(device_str)
        self.device_id = self.device.index if self.device.index is not None else 0
        self.fp16 = fp16  # kept for API compat, ONNX session uses fp32 internally

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1     # I/O binding handles parallelism on GPU
        so.inter_op_num_threads = 1

        providers = [
            ("CUDAExecutionProvider", {"device_id": self.device_id}),
            "CPUExecutionProvider",
        ]
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=so, providers=providers
        )

        actual = self.session.get_providers()
        if "CUDAExecutionProvider" not in actual:
            print(
                "  WARNING: PerchTeacher CUDAExecutionProvider not active. "
                "Falling back to CPU — training will be ~50x slower."
            )

        self.input_name = self.session.get_inputs()[0].name

        # Locate the 1536-d embedding output (Perch v2 has multiple outputs)
        self.embed_name = None
        self.embed_idx = None
        for i, o in enumerate(self.session.get_outputs()):
            if o.shape and o.shape[-1] == PERCH_EMBED_DIM:
                self.embed_idx = i
                self.embed_name = o.name
                break
        if self.embed_name is None:
            # Fallback: assume output 1 is embedding (Perch v2 convention)
            self.embed_idx = 1
            self.embed_name = self.session.get_outputs()[1].name

        # Other outputs that the graph produces — we must bind them somewhere
        # otherwise ORT errors out. Bind to CPU (we discard them).
        self._other_outputs = [
            o for i, o in enumerate(self.session.get_outputs()) if i != self.embed_idx
        ]

        print(f"  PerchTeacher ONNX loaded: {onnx_path.name}")
        print(f"    providers: {actual}")
        print(f"    embedding output: '{self.embed_name}' (idx={self.embed_idx})")

    @torch.no_grad()
    def embed(self, waveforms_5s: torch.Tensor) -> torch.Tensor:
        """waveforms_5s: (B, 160000) float32. Returns (B, 1536) float32 on the same device."""
        x = waveforms_5s.contiguous().float()
        if x.device != self.device:
            x = x.to(self.device)

        B = x.shape[0]
        # Pre-allocate output tensor on GPU — embedding will be written here directly
        embedding = torch.empty(
            (B, PERCH_EMBED_DIM), device=self.device, dtype=torch.float32
        )

        binding = self.session.io_binding()
        binding.bind_input(
            name=self.input_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr(),
        )
        binding.bind_output(
            name=self.embed_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=np.float32,
            shape=tuple(embedding.shape),
            buffer_ptr=embedding.data_ptr(),
        )
        # Bind other outputs to CPU (they'll be discarded)
        for o in self._other_outputs:
            binding.bind_output(name=o.name)

        self.session.run_with_iobinding(binding)
        return embedding


if __name__ == "__main__":
    # Smoke test
    import time
    print("Testing OnnxPerchTeacher...")
    teacher = PerchTeacher(device_str="cuda:0")
    x = torch.randn(64, 160_000, device="cuda:0")
    # warmup
    for _ in range(3):
        emb = teacher.embed(x)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        emb = teacher.embed(x)
    torch.cuda.synchronize()
    elapsed = (time.time() - t0) / 10
    print(f"  Batch 64 forward: {elapsed*1000:.1f} ms")
    print(f"  Output shape: {emb.shape}  device: {emb.device}  dtype: {emb.dtype}")
    print(f"  Output range: [{emb.min().item():.3f}, {emb.max().item():.3f}]")
    print(f"  Mean abs: {emb.abs().mean().item():.4f}")
    