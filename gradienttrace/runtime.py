from __future__ import annotations

import os
from dataclasses import dataclass

# The native TPU backend (torch_tpu) loads the compiled extension matching the
# installed torch build; these are the torch minors the Colab image ships.
NATIVE_TPU_TORCH_MINORS = ("2.11", "2.12", "2.13", "2.14", "2.15")


@dataclass
class Runtime:
    kind: str  # tpu | xla | cuda | cpu
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
    tpu_name = _native_tpu()
    if tpu_name:
        return Runtime(
            kind="tpu",
            name=tpu_name,
            precision="bf16",
            micro_bs=8,
            grad_accum=8,
            workers=0,
            pad_to_max=True,
        )
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


def _tpu_env() -> bool:
    return bool(
        os.environ.get("COLAB_TPU_ADDR")
        or os.environ.get("PJRT_DEVICE") == "TPU"
        or os.environ.get("XRT_TPU_CONFIG")
        or os.environ.get("TPU_NAME")
    )


def _native_tpu() -> str | None:
    """Name of the native torch_tpu accelerator, or None when absent."""
    try:
        import torch
    except Exception as exc:
        if _tpu_env():
            raise RuntimeError(
                "TPU environment detected but `import torch` failed. The native TPU "
                "backend ships one compiled extension per torch build ("
                + ", ".join(NATIVE_TPU_TORCH_MINORS)
                + "); reinstall a torch build it supports."
            ) from exc
        return None
    try:
        accelerator = getattr(torch, "accelerator", None)
        if accelerator is not None:
            current = accelerator.current_accelerator()
            device_type = getattr(current, "type", current)
            if device_type == "tpu" and accelerator.is_available():
                return "TPU (torch_tpu)"
        tpu = getattr(torch, "tpu", None)
        if tpu is not None and tpu.is_available():
            return "TPU (torch_tpu)"
    except Exception:
        return None
    return None


def _xla_requested() -> bool:
    if not _tpu_env():
        return False
    os.environ.setdefault("PJRT_DEVICE", "TPU")  # must be set before torch_xla is imported
    try:
        import torch_xla  # noqa: F401

        return True
    except Exception as exc:
        raise RuntimeError(
            "TPU environment detected but neither the native torch_tpu backend nor "
            "torch_xla is usable. torch_tpu needs a torch build it ships an "
            "extension for ("
            + ", ".join(NATIVE_TPU_TORCH_MINORS)
            + "); torch_xla needs an exactly matching torch build."
        ) from exc


def resolve_device(runtime: Runtime):
    import torch

    if runtime.kind == "tpu":
        return torch.device("tpu:0")
    if runtime.kind == "xla":
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    return torch.device("cuda" if runtime.kind == "cuda" else "cpu")
