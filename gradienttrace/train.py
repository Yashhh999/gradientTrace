"""Training entrypoint for gradientTrace.

Fine-tunes an encoder backbone (DeBERTa-v3-large by default) as a binary AI-text
detector on prompt-paired data, with generator-disjoint evaluation and HF Hub
checkpointing. Runs unchanged on CUDA (2x T4: fp16 + DDP), TPU v5e-1 (bf16 +
torch_xla), or CPU.

Smoke run (no network data, ~40 steps, verifies the whole pipeline):
    python -m gradienttrace.train --config configs/base.yaml --smoke --max-steps 40
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .data import build_datasets, samples_to_dataset, tokenize_for_training
from .evaluate import evaluate_slices
from .metrics import binary_detection_metrics
from .model import AIDetector, build_tokenizer
from .runtime import Runtime, detect_runtime

LOG = logging.getLogger("gradienttrace.train")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the gradientTrace AI-text detector")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--smoke", action="store_true", help="network-free synthetic run to verify the pipeline")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _resolve_runtime(cfg: Config, args: argparse.Namespace) -> Runtime:
    runtime = detect_runtime()
    if cfg.train.per_device_batch_size != "auto":
        runtime.micro_bs = int(cfg.train.per_device_batch_size)
    if cfg.train.grad_accum != "auto":
        runtime.grad_accum = int(cfg.train.grad_accum)
    if cfg.train.dataloader_workers != "auto":
        runtime.workers = int(cfg.train.dataloader_workers)
    if cfg.train.precision != "auto":
        if runtime.kind == "xla" and cfg.train.precision == "fp16":
            LOG.warning("fp16 is not supported on TPU/XLA; using bf16")
        else:
            runtime.precision = cfg.train.precision
    if args.smoke:
        runtime.micro_bs = min(runtime.micro_bs, 4)
        runtime.grad_accum = 1
        runtime.workers = 0
    LOG.info(
        "runtime=%s (%s) | precision=%s | micro_batch=%d x accum=%d (effective %d)",
        runtime.kind,
        runtime.name,
        runtime.precision,
        runtime.micro_bs,
        runtime.grad_accum,
        runtime.effective_batch,
    )
    return runtime


def _report_to(cfg: Config) -> list[str]:
    report_to = ["tensorboard"]
    if not cfg.wandb.enabled:
        return report_to
    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline"):
        LOG.warning("wandb.enabled=true but WANDB_API_KEY is unset; logging to tensorboard only")
        return report_to
    try:
        import wandb  # noqa: F401
    except ImportError:
        LOG.warning("wandb is enabled but not installed; logging to tensorboard only")
        return report_to
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)
    return report_to + ["wandb"]


def _build_training_args(
    cfg: Config,
    runtime: Runtime,
    max_steps: int,
    steps_mode: bool,
    push: bool,
    hub_id: str,
    train_size: int,
):
    from transformers import TrainingArguments

    kwargs = dict(
        output_dir=cfg.train.output_dir,
        per_device_train_batch_size=runtime.micro_bs,
        per_device_eval_batch_size=runtime.micro_bs,
        gradient_accumulation_steps=runtime.grad_accum,
        learning_rate=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        max_grad_norm=cfg.train.max_grad_norm,
        lr_scheduler_type="cosine",
        num_train_epochs=cfg.train.epochs,
        max_steps=max_steps,
        fp16=runtime.precision == "fp16",
        bf16=runtime.precision == "bf16",
        gradient_checkpointing=False,  # enabled explicitly on the model before Trainer wraps it
        logging_steps=cfg.train.log_every,
        save_total_limit=cfg.train.save_total_limit,
        metric_for_best_model="auroc",
        greater_is_better=True,
        load_best_model_at_end=True,
        label_names=["labels"],
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=runtime.workers,
        dataloader_pin_memory=runtime.kind == "cuda",
        report_to=_report_to(cfg),
        push_to_hub=push,
    )
    if push:
        kwargs["hub_model_id"] = hub_id
        kwargs["hub_private_repo"] = cfg.hub.private
    if steps_mode:
        step = 15 if max_steps <= 100 else max(50, max_steps // 10)
        kwargs.update(save_strategy="steps", save_steps=step, eval_steps=step)
        eval_strategy = "steps"
    else:
        kwargs["save_strategy"] = cfg.train.save_strategy
        eval_strategy = cfg.train.eval_strategy

    signature = inspect.signature(TrainingArguments.__init__).parameters
    eval_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    kwargs[eval_key] = eval_strategy

    if "warmup_ratio" in signature:
        kwargs["warmup_ratio"] = cfg.train.warmup_ratio
    else:  # transformers v5 removed warmup_ratio in favor of warmup_steps
        if max_steps > 0:
            total_steps = max_steps
        else:
            steps_per_epoch = math.ceil(train_size / max(1, runtime.effective_batch))
            total_steps = max(1, int(steps_per_epoch * cfg.train.epochs))
        kwargs["warmup_steps"] = max(1, int(cfg.train.warmup_ratio * total_steps))

    strategies = [cfg.hub.checkpoint_strategy, "every_save", "end", None] if push else [None]
    last_error: Exception | None = None
    for strategy in strategies:
        attempt = dict(kwargs)
        if strategy:
            attempt["hub_strategy"] = strategy
        try:
            return TrainingArguments(**attempt)
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise last_error  # type: ignore[misc]


def _build_trainer(model, tokenizer, train_ds, val_ds, train_args, cfg: Config):
    from transformers import DataCollatorWithPadding, EarlyStoppingCallback, Trainer

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))
        labels = np.asarray(eval_pred.label_ids).astype(int)
        metrics = binary_detection_metrics(labels, probs)
        keys = ("auroc", "auprc", "accuracy", "tpr_at_fpr_0.01", "ece")
        return {key: float(metrics[key]) for key in keys}

    callbacks = []
    if cfg.train.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience)
        )

    trainer_kwargs = dict(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**trainer_kwargs)


def _fmt(value, spec: str = ".4f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "n/a"


def _model_card(cfg: Config, report: dict, runtime: Runtime) -> str:
    rows = []
    for name, metrics in report.items():
        overall = metrics.get("overall", {})
        rows.append(
            f"| {name} | {overall.get('n', 0)} | {_fmt(overall.get('auroc'))} "
            f"| {_fmt(overall.get('tpr_at_fpr_0.01'), '.3f')} |"
        )
    table = "\n".join(rows)
    return f"""---
library_name: transformers
pipeline_tag: text-classification
base_model: {cfg.backbone}
tags: [ai-text-detection, gradienttrace]
---

# gradientTrace

Binary AI-text detector with a `{cfg.backbone}` backbone, trained on prompt-paired
human/model text with humanized-text augmentation and generator-disjoint splits.

## Held-out evaluation

| slice | n | AUROC | TPR @ 1% FPR |
|---|---|---|---|
{table}

Full report: `eval_report.json`. Data manifest: `data_manifest.json`.

## Intended use

Estimate P(machine-generated) for passages up to {cfg.max_length} tokens.
Scores are evidence, not proof: false positives occur on atypical human writing,
and paraphrased/humanized machine text is the known hard failure mode.
Do not use as the sole basis for academic-integrity or moderation decisions.

## Training

Runtime: {runtime.name}; precision {runtime.precision}; effective batch
{runtime.effective_batch}. Full configuration is in `config.json` and in the
gradientTrace repository.
"""


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args(argv)
    cfg = load_config(args.config)

    if args.backbone:
        cfg.backbone = args.backbone
    if args.seed is not None:
        cfg.seed = args.seed
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.output_dir:
        cfg.train.output_dir = args.output_dir

    max_steps = args.max_steps if args.max_steps is not None else cfg.train.max_steps
    if args.smoke:
        cfg.train.epochs = 1.0
        cfg.train.early_stopping_patience = 0
        cfg.train.log_every = 5
        max_steps = args.max_steps if args.max_steps is not None else 40
        if not args.output_dir:
            cfg.train.output_dir = cfg.train.output_dir.rstrip("/") + "-smoke"
    steps_mode = args.smoke or max_steps > 0

    if cfg.max_length > 512 and "deberta" in cfg.backbone.lower():
        LOG.warning("DeBERTa-v3 is trained at 512 positions; max_length=%d may degrade quality", cfg.max_length)

    runtime = _resolve_runtime(cfg, args)

    hub_id = args.hub_model_id or cfg.hub.model_id
    if args.smoke and args.hub_model_id is None and hub_id:
        hub_id = hub_id.rstrip("/") + "-smoke"
    push = bool(args.push_to_hub or cfg.hub.push)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if push and not hf_token:
        raise SystemExit("--push-to-hub requires HF_TOKEN (write scope) in the environment.")
    if hf_token:
        from huggingface_hub import login

        login(token=hf_token, add_to_git_credential=False)
    if push:
        from huggingface_hub import HfApi

        HfApi(token=hf_token).create_repo(
            hub_id, repo_type="model", private=cfg.hub.private, exist_ok=True
        )
        LOG.info("pushing checkpoints to https://huggingface.co/%s", hub_id)

    bundle = build_datasets(cfg, smoke=args.smoke)
    LOG.info(
        "data: train=%d val=%d slices=%s",
        len(bundle.train),
        len(bundle.val),
        {name: len(items) for name, items in bundle.slices.items()},
    )

    tokenizer = build_tokenizer(cfg.backbone)
    train_ds = tokenize_for_training(
        samples_to_dataset(bundle.train), tokenizer, cfg.max_length, pad_to_max=runtime.pad_to_max
    )
    val_ds = tokenize_for_training(
        samples_to_dataset(bundle.val), tokenizer, cfg.max_length, pad_to_max=runtime.pad_to_max
    )

    model = AIDetector.from_backbone(
        cfg.backbone, head_dropout=cfg.head.dropout, pooling=cfg.head.pooling
    )
    model.config.gradienttrace_max_length = cfg.max_length
    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    train_args = _build_training_args(
        cfg, runtime, max_steps, steps_mode, push, hub_id, train_size=len(train_ds)
    )
    trainer = _build_trainer(model, tokenizer, train_ds, val_ds, train_args, cfg)

    trainer.train(**({"resume_from_checkpoint": True} if args.resume else {}))

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    trainer.save_state()
    tokenizer.save_pretrained(output_dir)

    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    eval_batch = runtime.micro_bs if runtime.kind == "xla" else max(runtime.micro_bs, 16)
    report = evaluate_slices(
        unwrapped,
        tokenizer,
        bundle.slices,
        trainer.accelerator.device,
        max_length=cfg.max_length,
        batch_size=eval_batch,
        pad_to_max=runtime.pad_to_max,
    )

    final_report = {
        "hub_model_id": hub_id if push else None,
        "runtime": asdict(runtime),
        "config": cfg.to_dict(),
        "splits": {name: len(items) for name, items in {"train": bundle.train, "val": bundle.val, **bundle.slices}.items()},
        "threshold": report.get("val", {}).get("overall", {}).get("best_threshold"),
        "metrics": report,
    }
    (output_dir / "eval_report.json").write_text(
        json.dumps(final_report, indent=2, default=float), encoding="utf-8"
    )
    (output_dir / "data_manifest.json").write_text(
        json.dumps(bundle.manifest, indent=2, default=float), encoding="utf-8"
    )

    if push:
        from huggingface_hub import HfApi

        api = HfApi(token=hf_token)
        trainer.push_to_hub(commit_message=f"gradientTrace best model ({cfg.backbone})")
        api.upload_file(
            path_or_fileobj=str(output_dir / "eval_report.json"),
            path_in_repo="eval_report.json",
            repo_id=hub_id,
            commit_message="gradientTrace eval report",
        )
        api.upload_file(
            path_or_fileobj=str(output_dir / "data_manifest.json"),
            path_in_repo="data_manifest.json",
            repo_id=hub_id,
            commit_message="gradientTrace data manifest",
        )
        api.upload_file(
            path_or_fileobj=_model_card(cfg, report, runtime).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=hub_id,
            commit_message="gradientTrace model card",
        )
        LOG.info("model + artifacts pushed to https://huggingface.co/%s", hub_id)

    LOG.info("slice                   n      auroc   tpr@1%%fpr")
    for name, metrics in report.items():
        overall = metrics.get("overall", {})
        LOG.info(
            "%-22s %6d   %s    %s",
            name,
            overall.get("n", 0),
            _fmt(overall.get("auroc")),
            _fmt(overall.get("tpr_at_fpr_0.01"), ".3f"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
