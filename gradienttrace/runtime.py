from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Runtime:
    kind: str  # cuda | xla | cpu
    name: str
    precision: str  # fp16 | bf16 | no
    micro_bs: int
    grad_accum: int
    workers: int
    pad_to_max: bool

    @property
    def effective_batch(self) -> int:
        return self.micro_bs * self.grad_accum


def detect_runtime() -> Runtime:
    if _xla_requested():
        return Runtime(
            kind="xla",
            name="TPU (torch_xla)",
            precision="bf16",
            micro_bs=8,
            grad_accum=8,
            workers=0,
            pad_to_max=True,
        )
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            turing = "T4" in name.upper()
            return Runtime(
                kind="cuda",
                name=name,
                precision="fp16" if turing else "bf16",
                micro_bs=16,
                grad_accum=4,
                workers=2,
                pad_to_max=False,
            )
    except Exception:
        pass
    return Runtime(
        kind="cpu",
        name="cpu",
        precision="no",
        micro_bs=8,
        grad_accum=1,
        workers=0,
        pad_to_max=False,
    )


def _xla_requested() -> bool:
    if not (
        os.environ.get("PJRT_DEVICE")
        or os.environ.get("COLAB_TPU_ADDR")
        or os.environ.get("XRT_TPU_CONFIG")
        or os.environ.get("TPU_NAME")
    ):
        return False
    os.environ.setdefault("PJRT_DEVICE", "TPU")  # must be set before torch_xla is imported
    try:
        import torch_xla  # noqa: F401

        return True
    except Exception:
        return False


def resolve_device(runtime: Runtime):
    if runtime.kind == "xla":
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    import torch

    return torch.device("cuda" if runtime.kind == "cuda" else "cpu")
