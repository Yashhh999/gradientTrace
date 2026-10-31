"""Standalone evaluation: AUROC / TPR@FPR by generator, domain and attack slice."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .data import Sample, build_datasets, norm_text
from .metrics import sliced_report
from .model import AIDetector, build_tokenizer
from .runtime import detect_runtime, resolve_device

LOG = logging.getLogger("gradienttrace.evaluate")


def _progress(total: int, desc: str):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return None
    return tqdm(total=total, desc=desc, unit="batch", dynamic_ncols=True)


def predict_probs(
    model,
    tokenizer,
    samples,
    device,
    max_length: int = 512,
    batch_size: int = 16,
    pad_to_max: bool = False,
    desc: str = "eval",
) -> np.ndarray:
    import torch

    model.eval()
    device = torch.device(device)
    count = len(samples)
    probs = np.zeros(count, dtype=np.float32)
    if count == 0:
        return probs

    # Length-sorted batches keep padding low with dynamic padding, which is
    # several times cheaper than padding every batch to max_length.
    order = sorted(range(count), key=lambda i: len(samples[i].text))
    starts = range(0, count, batch_size)
    batches = len(starts)
    use_amp = device.type == "cuda"
    is_xla = device.type == "xla"
    bar = _progress(batches, desc)
    report_every = max(1, batches // 20)
    started = time.monotonic()

    with torch.inference_mode():
        for step, start in enumerate(starts, 1):
            index = order[start : start + batch_size]
            enc = tokenizer(
                [samples[i].text for i in index],
                truncation=True,
                max_length=max_length,
                padding="max_length" if pad_to_max else True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**enc).logits.float()
            if use_amp and not torch.isfinite(logits).all():
                with torch.autocast(device_type="cuda", enabled=False):
                    logits = model(**enc).logits.float()
            probs[index] = torch.sigmoid(logits).reshape(-1).cpu().numpy()
            if is_xla:
                import torch_xla.core.xla_model as xm

                xm.mark_step()
            if bar is not None:
                bar.update(1)
            elif step % report_every == 0 or step == batches:
                LOG.info("%s: %d/%d batches (%.0fs)", desc, step, batches, time.monotonic() - started)

    if bar is not None:
        bar.close()
    return probs


def evaluate_slices(
    model,
    tokenizer,
    slices: dict[str, list[Sample]],
    device,
    max_length: int = 512,
    batch_size: int = 16,
    pad_to_max: bool = False,
) -> dict:
    report: dict = {}
    for name, samples in slices.items():
        if not samples:
            report[name] = {"overall": {"n": 0}}
            LOG.warning("slice '%s' is empty", name)
            continue
        started = time.monotonic()
        probs = predict_probs(
            model, tokenizer, samples, device, max_length, batch_size, pad_to_max, desc=f"eval {name}"
        )
        labels = np.array([s.label for s in samples])
        generators = np.array([s.generator for s in samples])
        domains = np.array([s.domain for s in samples])
        report[name] = sliced_report(labels, probs, {"generator": generators, "domain": domains})
        overall = report[name]["overall"]
        LOG.info(
            "slice %-22s n=%6d auroc=%s tpr@1%%fpr=%s (%.0fs)",
            name,
            overall.get("n", 0),
            _fmt(overall.get("auroc")),
            _fmt(overall.get("tpr_at_fpr_0.01"), ".3f"),
            time.monotonic() - started,
        )
    return report


def load_model_and_tokenizer(model_path: str):
    path = Path(model_path)
    if path.is_dir() and not (path / "config.json").exists():
        checkpoints = sorted(
            path.glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[-1])
        )
        if checkpoints:
            LOG.info("no model at %s; using latest checkpoint %s", path, checkpoints[-1])
            path = checkpoints[-1]
        else:
            raise FileNotFoundError(f"{path} has no config.json and no checkpoint-* directories")
    model = AIDetector.from_pretrained(str(path))
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(path))
    except Exception:
        tokenizer = build_tokenizer(getattr(model.config, "backbone_name", "microsoft/deberta-v3-large"))
    return model, tokenizer


def _load_custom(path: str) -> list[Sample]:
    samples: list[Sample] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("label") not in (0, 1) or not isinstance(rec.get("text"), str):
                continue
            label = int(rec["label"])
            samples.append(
                Sample(
                    text=norm_text(rec["text"]),
                    label=label,
                    generator=str(rec.get("generator") or ("human" if label == 0 else "unknown")),
                    source="custom",
                    prompt=str(rec.get("prompt") or ""),
                    domain=str(rec.get("domain") or "custom"),
                    variant=str(rec.get("variant") or "raw"),
                )
            )
    return samples


def _fmt(value, spec: str = ".4f") -> str:
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return "n/a"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a gradientTrace detector")
    parser.add_argument("--model", required=True, help="Hub model id or local checkpoint dir")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--smoke", action="store_true", help="evaluate on the synthetic smoke slices")
    parser.add_argument("--eval-jsonl", default=None, help="extra eval file with the local_jsonl schema")
    parser.add_argument("--out", default="eval_report.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = load_config(args.config)
    runtime = detect_runtime()
    device = resolve_device(runtime)

    bundle = build_datasets(cfg, smoke=args.smoke)
    slices = dict(bundle.slices)
    if args.eval_jsonl:
        slices["custom"] = _load_custom(args.eval_jsonl)

    model, tokenizer = load_model_and_tokenizer(args.model)
    model.to(device)
    max_length = int(getattr(model.config, "gradienttrace_max_length", cfg.max_length))
    report = evaluate_slices(
        model,
        tokenizer,
        slices,
        device,
        max_length=max_length,
        batch_size=args.batch_size,
        pad_to_max=runtime.pad_to_max,
    )

    out = Path(args.out)
    out.write_text(
        json.dumps({"model": args.model, "metrics": report}, indent=2, default=float),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            name: {
                "n": metrics.get("overall", {}).get("n", 0),
                "auroc": metrics.get("overall", {}).get("auroc"),
                "tpr_at_fpr_0.01": metrics.get("overall", {}).get("tpr_at_fpr_0.01"),
            }
            for name, metrics in report.items()
        },
        indent=2,
        default=float,
    ))
    LOG.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
