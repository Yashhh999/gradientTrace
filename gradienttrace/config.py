from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class HeadCfg:
    dropout: float = 0.1
    pooling: str = "cls"


@dataclass
class TrainCfg:
    output_dir: str = "runs/gradientTrace"
    epochs: float = 3.0
    lr: float = 1.5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    per_device_batch_size: Any = "auto"
    grad_accum: Any = "auto"
    precision: str = "auto"  # auto | fp16 | bf16 | no
    gradient_checkpointing: bool = True
    early_stopping_patience: int = 2
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    dataloader_workers: Any = "auto"
    log_every: int = 25
    max_steps: int = -1


@dataclass
class DataCfg:
    total_samples: int = 150_000
    min_chars: int = 200
    max_chars: int = 12_000
    mix: dict[str, float] = field(
        default_factory=lambda: {"human": 0.40, "ai_foundational": 0.40, "ai_frontier": 0.20}
    )
    val_frac: float = 0.10
    test_frac: float = 0.10
    max_humanized_frac: float = 0.25
    held_out_generators: list[str] = field(
        default_factory=lambda: ["llama-2", "mistral", "gemini"]
    )
    frontier_patterns: list[str] = field(
        default_factory=lambda: [
            "gpt-4",
            "gpt-5",
            "gpt4",
            "o1",
            "o3",
            "claude",
            "gemini",
            "grok",
            "llama-3",
            "llama3",
            "mistral-large",
            "deepseek",
            "qwen",
            "command-r",
        ]
    )
    sources: list[str] = field(default_factory=lambda: ["hc3", "pile", "local_jsonl"])
    local_jsonl: str = "data/local.jsonl"
    scan_limit_per_source: int = 400_000
    ood_cap: int = 8_000


@dataclass
class HubCfg:
    model_id: str = "Yashhh999/gradientTrace"
    push: bool = False
    private: bool = False
    checkpoint_strategy: str = "every_save"  # every_save | end | all_checkpoints


@dataclass
class WandbCfg:
    enabled: bool = False
    project: str = "gradienttrace"


@dataclass
class Config:
    seed: int = 42
    backbone: str = "microsoft/deberta-v3-large"
    max_length: int = 512
    head: HeadCfg = field(default_factory=HeadCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
    hub: HubCfg = field(default_factory=HubCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "head": HeadCfg,
    "train": TrainCfg,
    "data": DataCfg,
    "hub": HubCfg,
    "wandb": WandbCfg,
}


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cfg = Config()
    for name in ("seed", "backbone", "max_length"):
        if name in raw:
            setattr(cfg, name, raw[name])
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        obj = cls()
        allowed = {f.name for f in fields(cls)}
        unknown = set(section) - allowed
        if unknown:
            raise KeyError(f"Unknown keys in config section '{name}': {sorted(unknown)}")
        for key, value in section.items():
            setattr(obj, key, value)
        setattr(cfg, name, obj)
    return cfg
