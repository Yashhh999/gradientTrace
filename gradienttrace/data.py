"""Data pipeline: paired human/AI text, generator-disjoint splits, humanized slice.

Every sample carries (label, generator, source, prompt, domain, variant, bucket).
Splits are grouped by prompt so the same prompt never straddles train/test.
Samples from `held_out_generators` never enter training; they form the OOD test.

Source schema:
  hc3          Hello-SimpleAI/HC3 (question + human_answers + chatgpt_answers)
  pile         artem9k/ai-text-detection-pile (text + source)
  local_jsonl  data/local.jsonl with keys:
               {text, label(0|1), generator?, prompt?, domain?, variant?, bucket?}
  hf:<org/name>[:split]   generic adapter: finds text/label/generator/prompt columns
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .config import Config

LOG = logging.getLogger("gradienttrace.data")

BUCKETS = ("human", "ai_foundational", "ai_frontier")
_WS = re.compile(r"\s+")


@dataclass
class Sample:
    text: str
    label: int
    generator: str
    source: str
    prompt: str = ""
    domain: str = "general"
    variant: str = "raw"
    bucket: str = ""


@dataclass
class Bundle:
    train: list[Sample]
    val: list[Sample]
    slices: dict[str, list[Sample]]
    manifest: dict


def norm_text(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def text_key(text: str) -> str:
    return _sha(norm_text(text).lower())


def group_key(sample: Sample, seed: int) -> int:
    if sample.prompt:
        base = f"{norm_text(sample.prompt).lower()}|{sample.domain}"
    else:
        base = text_key(sample.text)
    return int(_sha(f"{seed}|{base}"), 16) % 10_000


def matches_any(name: str, patterns: Iterable[str]) -> bool:
    low = (name or "").lower()
    return any(p and p.lower() in low for p in patterns)


def ai_bucket(generator: str, frontier_patterns: Iterable[str]) -> str:
    return "ai_frontier" if matches_any(generator, frontier_patterns) else "ai_foundational"


def load_hc3(cfg: Config, scan_limit: int) -> Iterator[Sample]:
    from datasets import load_dataset

    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= scan_limit:
            return
        prompt = row.get("question") or ""
        domain = row.get("source") or "general"
        for answer in (row.get("human_answers") or [])[:2]:
            yield Sample(norm_text(answer), 0, "human", "hc3", prompt=prompt, domain=domain, bucket="human")
        for answer in (row.get("chatgpt_answers") or [])[:2]:
            yield Sample(norm_text(answer), 1, "chatgpt", "hc3", prompt=prompt, domain=domain)


def load_pile(cfg: Config, scan_limit: int) -> Iterator[Sample]:
    from datasets import load_dataset

    ds = load_dataset("artem9k/ai-text-detection-pile", split="train", streaming=True)
    for i, row in enumerate(ds):
        if i >= scan_limit:
            return
        generator = str(row.get("source") or "unknown")
        label = 0 if generator.lower() == "human" else 1
        yield Sample(
            norm_text(row.get("text") or ""),
            label,
            "human" if label == 0 else generator,
            "pile",
            domain="web",
        )


def load_local_jsonl(cfg: Config, scan_limit: int) -> Iterator[Sample]:
    path = Path(cfg.data.local_jsonl)
    if not path.exists():
        LOG.info("local_jsonl not found at %s; skipping", path)
        return
    bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= scan_limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(rec.get("text"), str) or rec.get("label") not in (0, 1):
                bad += 1
                continue
            label = int(rec["label"])
            yield Sample(
                text=norm_text(rec["text"]),
                label=label,
                generator=str(rec.get("generator") or ("human" if label == 0 else "unknown")),
                source="local",
                prompt=str(rec.get("prompt") or ""),
                domain=str(rec.get("domain") or "general"),
                variant=str(rec.get("variant") or "raw"),
                bucket=str(rec.get("bucket") or ""),
            )
    if bad:
        LOG.warning("local_jsonl: skipped %d malformed records", bad)


def _parse_generic(spec: str) -> tuple[str, str | None, str]:
    body = spec[len("hf:"):]
    config_name = None
    if "?" in body:
        body, query = body.split("?", 1)
        for part in query.split("&"):
            if part.startswith("config="):
                config_name = part.split("=", 1)[1]
    split = "train"
    if ":" in body:
        body, split = body.split(":", 1)
    return body, config_name, split


def _label_from(value) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in (0, 1):
        return int(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"human", "real", "0", "false"}:
            return 0
        if low in {"ai", "machine", "generated", "model", "1", "true", "chatgpt"}:
            return 1
    return None


def load_generic(spec: str, cfg: Config, scan_limit: int) -> Iterator[Sample]:
    from datasets import load_dataset

    name, config_name, split = _parse_generic(spec)
    ds = (
        load_dataset(name, config_name, split=split, streaming=True)
        if config_name
        else load_dataset(name, split=split, streaming=True)
    )
    features = set(ds.features or [])
    text_col = next((c for c in ("text", "generation", "response", "content", "answer") if c in features), None)
    label_col = next((c for c in ("label", "is_ai", "machine_generated", "is_machine") if c in features), None)
    gen_col = next((c for c in ("model", "generator", "source", "model_name") if c in features), None)
    prompt_col = next((c for c in ("prompt", "question", "instruction") if c in features), None)
    domain_col = next((c for c in ("domain", "topic") if c in features), None)
    if text_col is None:
        raise ValueError(f"{spec}: no text column in {sorted(features)}")
    for i, row in enumerate(ds):
        if i >= scan_limit:
            return
        text = row.get(text_col)
        if not isinstance(text, str):
            continue
        generator = str(row.get(gen_col) or "unknown") if gen_col else "unknown"
        label = _label_from(row.get(label_col)) if label_col else None
        if label is None:
            if matches_any(generator, ("human",)):
                label = 0
            elif generator.lower() not in {"", "unknown"}:
                label = 1
            else:
                continue
        if label == 0:
            generator = "human"
        yield Sample(
            text=norm_text(text),
            label=label,
            generator=generator,
            source=f"hf:{name}",
            prompt=str(row.get(prompt_col) or "") if prompt_col else "",
            domain=str(row.get(domain_col) or "general") if domain_col else "general",
        )


def _make_loader(spec: str, cfg: Config):
    if spec == "hc3":
        return load_hc3
    if spec == "pile":
        return load_pile
    if spec == "local_jsonl":
        return load_local_jsonl
    if spec.startswith("hf:"):
        return lambda _cfg, scan_limit: load_generic(spec, _cfg, scan_limit)
    raise ValueError(f"unknown data source '{spec}'")


def _targets(total: int, mix: dict[str, float]) -> dict[str, int]:
    unknown = set(mix) - set(BUCKETS)
    if unknown:
        raise ValueError(f"mix keys must be within {BUCKETS}; got {sorted(unknown)}")
    weight = sum(mix.values())
    if weight <= 0:
        raise ValueError("mix must sum to > 0")
    return {bucket: int(total * w / weight) for bucket, w in mix.items()}


def build_datasets(cfg: Config, smoke: bool = False) -> Bundle:
    if smoke:
        return build_smoke_bundle(cfg)

    dcfg = cfg.data
    if not (0 < dcfg.val_frac < 1 and 0 < dcfg.test_frac < 1 and dcfg.val_frac + dcfg.test_frac < 1):
        raise ValueError("val_frac/test_frac must be positive and sum to < 1")

    targets = _targets(dcfg.total_samples, dcfg.mix)
    caps = {bucket: int(n * 1.4) + 64 for bucket, n in targets.items()}
    pools: dict[str, list[Sample]] = {b: [] for b in BUCKETS}
    ood_ai: list[Sample] = []
    seen: set[str] = set()
    manifest: dict = {
        "smoke": False,
        "seed": cfg.seed,
        "backbone": cfg.backbone,
        "targets": targets,
        "sources": {},
        "errors": {},
        "dropped": {"short": 0, "long": 0, "duplicate": 0, "bad_label": 0},
        "held_out_generators": list(dcfg.held_out_generators),
    }

    for spec in dcfg.sources:
        loader = _make_loader(spec, cfg)
        loaded = 0
        try:
            for sample in loader(cfg, dcfg.scan_limit_per_source):
                sample.text = norm_text(sample.text)
                if len(sample.text) < dcfg.min_chars:
                    manifest["dropped"]["short"] += 1
                    continue
                if len(sample.text) > dcfg.max_chars:
                    manifest["dropped"]["long"] += 1
                    continue
                if sample.label not in (0, 1):
                    manifest["dropped"]["bad_label"] += 1
                    continue
                key = text_key(sample.text)
                if key in seen:
                    manifest["dropped"]["duplicate"] += 1
                    continue
                seen.add(key)

                if sample.label == 0:
                    sample.generator = "human"
                    sample.bucket = "human"
                else:
                    if sample.bucket not in ("ai_foundational", "ai_frontier"):
                        sample.bucket = ai_bucket(sample.generator, dcfg.frontier_patterns)
                    if matches_any(sample.generator, dcfg.held_out_generators):
                        if len(ood_ai) < dcfg.ood_cap:
                            ood_ai.append(sample)
                        loaded += 1
                        continue

                pool = pools[sample.bucket]
                if len(pool) < caps[sample.bucket]:
                    pool.append(sample)
                    loaded += 1
                if all(len(pools[b]) >= caps[b] for b in BUCKETS):
                    break
        except Exception as exc:  # a flaky hub dataset must not kill the run
            LOG.warning("source '%s' failed: %r", spec, exc)
            manifest["errors"][spec] = repr(exc)
        manifest["sources"][spec] = loaded
        if all(len(pools[b]) >= caps[b] for b in BUCKETS):
            break

    if not pools["human"]:
        raise RuntimeError(
            "No human samples loaded. Add a human source (hc3/pile/local_jsonl) "
            "or provide data/local.jsonl."
        )
    if not pools["ai_foundational"] and not pools["ai_frontier"]:
        raise RuntimeError(
            "No AI samples loaded. Add an AI source or provide data/local.jsonl."
        )
    if not pools["ai_frontier"]:
        LOG.warning(
            "No ai_frontier samples loaded: modern-model robustness depends on this slice. "
            "Add prompt-paired frontier outputs via data/local.jsonl."
        )

    rng = random.Random(cfg.seed)
    chosen: dict[str, list[Sample]] = {}
    humanized_reserve: list[Sample] = []
    for bucket, target in targets.items():
        pool = pools[bucket][:]
        if not pool:
            LOG.warning("bucket '%s' has no samples; effective mix is rebalanced", bucket)
            chosen[bucket] = []
            continue
        rng.shuffle(pool)
        if bucket != "human":
            humanized = [s for s in pool if s.variant == "humanized"]
            regular = [s for s in pool if s.variant != "humanized"]
            keep = humanized[: int(target * dcfg.max_humanized_frac)]
            humanized_reserve.extend(humanized[len(keep):])
            pool = regular[: target - len(keep)] + keep
            rng.shuffle(pool)
        chosen[bucket] = pool[:target]

    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []
    val_cut = int(dcfg.val_frac * 10_000)
    test_cut = int((dcfg.val_frac + dcfg.test_frac) * 10_000)
    for bucket in BUCKETS:
        for sample in chosen[bucket]:
            key = group_key(sample, cfg.seed)
            if key < val_cut:
                val.append(sample)
            elif key < test_cut:
                test.append(sample)
            else:
                train.append(sample)

    if not train or not val:
        raise RuntimeError(
            f"Split too small (train={len(train)}, val={len(val)}). "
            "Increase data.total_samples or lower val_frac/test_frac."
        )

    humans_eval = [s for s in val + test if s.label == 0]
    rng.shuffle(humans_eval)
    ood_humans = humans_eval[: len(ood_ai)]
    humanized_ai = [s for s in humanized_reserve + test if s.label == 1 and s.variant == "humanized"]
    rng.shuffle(humanized_ai)
    humanized_ai = humanized_ai[: dcfg.ood_cap]
    rng.shuffle(humans_eval)
    humanized_humans = humans_eval[: len(humanized_ai)]

    slices = {
        "val": val,
        "test_indomain": test,
        "test_ood_generators": ood_ai + ood_humans,
        "test_humanized": humanized_ai + humanized_humans,
    }

    manifest["splits"] = {
        "train": len(train),
        "val": len(val),
        "test_indomain": len(test),
        "test_ood_generators": len(slices["test_ood_generators"]),
        "test_humanized": len(slices["test_humanized"]),
    }
    manifest["train_buckets"] = {b: sum(1 for s in train if s.bucket == b) for b in BUCKETS}
    manifest["train_generators"] = _generator_counts(train)
    return Bundle(train=train, val=val, slices=slices, manifest=manifest)


def _generator_counts(samples: list[Sample], top: int = 25) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.generator] = counts.get(sample.generator, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
    return dict(ordered)


_HUMAN = [
    "So I tried this recipe last weekend and honestly it was a mess at first. The dough kept sticking "
    "to the counter, and I burned the first batch because I walked away to answer a text. Second try "
    "went better once I chilled the butter properly, though the edges still came out a little dark. "
    "Nobody at the table complained, which I am counting as a win.",
    "My commute takes about forty minutes each way, mostly on the old blue line that rattles past the "
    "warehouses. I usually read or just watch the city wake up. There is a bakery near the third stop "
    "that opens stupidly early, and sometimes I grab a cheap coffee there if I am running ahead of "
    "schedule, which is rare.",
    "I have been keeping a little notebook of things my grandma says, because half of them make no "
    "sense until you think about them for a while. Last week she told me that patience is just "
    "curiosity wearing comfortable shoes. I wrote it down twice to make sure I had it right, and then "
    "I sat there trying to figure out what she meant.",
    "The project started as a joke between me and my roommate, mostly because we were bored and had a "
    "spare Saturday. We built the thing out of cardboard and duct tape and it somehow survived the "
    "whole summer on the porch. Looking back, I think the fun part was never finishing it properly, "
    "just adding silly features whenever we felt like it.",
    "When I moved into my first apartment I owned exactly two plates, one pan, and a folding chair I "
    "found by the dumpster. I told myself it was minimalist living, but really I was just broke. It "
    "took almost a year before the place felt like mine, and the day I finally hung a picture on the "
    "wall, I stood back and laughed at how long it took.",
    "The trail behind the library floods every spring, which turns a boring walk into a small "
    "adventure. Last April I saw a heron standing in the middle of the path, completely unbothered by "
    "the water or by me. I waited a long time for it to move and eventually just turned around and "
    "took the long way home, which was fine by me.",
]

_OPENERS = [
    "In today's fast-paced world,",
    "In the realm of modern technology,",
    "It is important to note that",
    "From a comprehensive perspective,",
    "Navigating the complexities of everyday life,",
]

_MIDDLES = [
    "it is essential to consider the multifaceted nature of this subject.",
    "one must acknowledge the pivotal role of continuous improvement.",
    "stakeholders must align on a holistic strategy that fosters sustainable outcomes.",
    "the interplay between innovation and tradition remains a central consideration.",
    "a structured approach can significantly enhance overall effectiveness.",
    "careful evaluation of competing priorities enables informed decision-making.",
]

_CLOSERS = [
    "Ultimately, the key lies in balancing competing priorities while maintaining a clear vision.",
    "In conclusion, thoughtful planning and adaptability are vital for long-term success.",
    "By embracing these principles, individuals and organizations can unlock their full potential.",
    "Overall, a nuanced understanding of these dynamics is indispensable.",
    "To summarize, consistent effort and strategic thinking yield meaningful results.",
]


def _ai_text(i: int) -> str:
    opener = _OPENERS[i % len(_OPENERS)]
    mid1 = _MIDDLES[(i // 3) % len(_MIDDLES)]
    mid2 = _MIDDLES[(i * 7 + 1) % len(_MIDDLES)]
    closer = _CLOSERS[(i // 5) % len(_CLOSERS)]
    return f"{opener} {mid1} {mid2} {closer} Scenario {i} is examined here."


def build_smoke_bundle(cfg: Config) -> Bundle:
    """Synthetic, network-free dataset that exercises every code path."""
    samples: list[Sample] = []
    for i in range(220):
        text = f"{_HUMAN[i % len(_HUMAN)]} Entry {i}."
        samples.append(
            Sample(text, 0, "human", "smoke", prompt=f"smoke prompt {i % 140}", domain="smoke", bucket="human")
        )
    for i in range(180):
        generator = "smoke-base-1" if i % 2 == 0 else "smoke-base-2"
        samples.append(
            Sample(_ai_text(i), 1, generator, "smoke", prompt=f"smoke prompt {i % 140}", domain="smoke", bucket="ai_foundational")
        )
    for i in range(60):
        samples.append(
            Sample(_ai_text(1000 + i), 1, "smoke-frontier-1", "smoke", prompt=f"smoke frontier {i}", domain="smoke", bucket="ai_frontier")
        )
    for i in range(40):
        samples.append(
            Sample(_ai_text(2000 + i), 1, "smoke-ood-v1", "smoke", prompt=f"smoke ood {i}", domain="smoke", bucket="ai_foundational")
        )
    for i in range(60):
        text = _ai_text(3000 + i).replace("Ultimately,", "I mean,").replace("In conclusion,", "So anyway,")
        samples.append(
            Sample(text, 1, "smoke-base-1", "smoke", prompt=f"smoke humanized {i}", domain="smoke", variant="humanized", bucket="ai_foundational")
        )

    rng = random.Random(0)
    train: list[Sample] = []
    val: list[Sample] = []
    test: list[Sample] = []
    ood_ai: list[Sample] = []
    for sample in samples:
        if sample.label == 1 and matches_any(sample.generator, ["smoke-ood"]):
            ood_ai.append(sample)
            continue
        key = group_key(sample, cfg.seed)
        if key < 1500:
            val.append(sample)
        elif key < 3000:
            test.append(sample)
        else:
            train.append(sample)

    humans_eval = [s for s in val + test if s.label == 0]
    rng.shuffle(humans_eval)
    humanized_eval = [s for s in val + test if s.variant == "humanized"]
    slices = {
        "val": val,
        "test_indomain": test,
        "test_ood_generators": ood_ai + humans_eval[: len(ood_ai)],
        "test_humanized": humanized_eval + humans_eval[: len(humanized_eval)],
    }
    manifest = {
        "smoke": True,
        "seed": cfg.seed,
        "note": "synthetic data; verifies the pipeline end to end, not model quality",
        "splits": {name: len(items) for name, items in slices.items()} | {"train": len(train)},
    }
    return Bundle(train=train, val=val, slices=slices, manifest=manifest)


def samples_to_dataset(samples: list[Sample]):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "text": [s.text for s in samples],
            "label": [float(s.label) for s in samples],
            "source": [s.source for s in samples],
            "generator": [s.generator for s in samples],
            "domain": [s.domain for s in samples],
            "variant": [s.variant for s in samples],
            "bucket": [s.bucket for s in samples],
        }
    )


def tokenize_for_training(dataset, tokenizer, max_length: int, pad_to_max: bool):
    def fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length" if pad_to_max else False,
        )

    ds = dataset.map(
        fn,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c != "label"],
        desc="tokenize",
    )
    return ds.rename_column("label", "labels")
