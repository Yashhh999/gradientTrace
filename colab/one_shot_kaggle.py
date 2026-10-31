# gradientTrace - one-shot Kaggle cell
# Paste this whole file into a single Kaggle notebook cell.
# Notebook settings: Accelerator = GPU T4 x2, Internet = On.
# Add-ons > Secrets: HF_TOKEN (write scope) for Hub checkpoints; WANDB_API_KEY optional.
RUN = "smoke"                        # "smoke" (~10 min sanity) | "full" (real training)
CONFIG = "base"                      # base | v2 | pilot -> configs/<CONFIG>.yaml
HUB_ID = ""                          # "" = per-CONFIG default (gradientTrace / -v2 / -v2-pilot)
RAID_HUB = "Yashhh999/gradientTrace" # repo holding v2/raid_v2.jsonl from colab/build_raid.py
NEWS_HUB = "Yashhh999/gradientTrace" # repo holding v2/news_pairs.jsonl from colab/build_news.py
PUSH_SMOKE = False                   # also push the smoke model to "<HUB_ID>-smoke"
MAX_STEPS = 0                        # full run: stop after this many optimizer steps per session
                                     # (RESUME keeps counting from the checkpoint); 0 = all EPOCHS
SAVE_EVERY = 300                     # full run: checkpoint + eval every N steps, pushed to the Hub
EPOCHS = None                        # None = config (3); e.g. 1 for a single full-data pass
RESUME = False                       # continue from the newest checkpoint: local first, else the Hub
MICRO_BATCH = 4                      # per GPU; 4 x 8 accum x 2 GPUs = 64 global batch
GRAD_ACCUM = 8
LOCAL_JSONL = ""                     # e.g. "/kaggle/input/gradienttrace/local.jsonl"

import collections
import json
import os, shutil, socket, subprocess, sys


def run(command, check=True):
    """Stream a child's output through the notebook; repeat the traceback on failure."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    recent = collections.deque(maxlen=4000)
    with subprocess.Popen(
        command,
        shell=isinstance(command, str),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    ) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            recent.append(line)
        proc.wait()
    if check and proc.returncode:
        tail = list(recent)
        start = max(
            (i for i, line in enumerate(tail) if "Traceback (most recent call last)" in line),
            default=max(0, len(tail) - 80),
        )
        print("\n[gradientTrace] child failure (reprinted so it is not scrolled away):")
        print("".join(tail[start : start + 120]), flush=True)
        raise RuntimeError(f"command failed with exit {proc.returncode}: {command}")
    return proc.returncode


def check_hub(repo_id):
    """Prove the token can create and write the repo before a two-hour run starts."""
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        me = api.whoami()
        role = ((me.get("auth") or {}).get("accessToken") or {}).get("role")
        print(f"[gradientTrace] hub token: user={me.get('name')} role={role}", flush=True)
        if role == "read":
            raise RuntimeError("the token is read-only")
        api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=b"ok",
            path_in_repo=".preflight",
            repo_id=repo_id,
            commit_message="gradientTrace preflight",
        )
        api.delete_file(".preflight", repo_id=repo_id, commit_message="gradientTrace preflight cleanup")
    except Exception as exc:
        raise RuntimeError(
            f"Hub preflight failed for {repo_id}: {exc!r}. Create it once at "
            f"https://huggingface.co/new (model, private, name '{repo_id.split('/')[-1]}'), "
            "or use a token with write scope and repo creation; then re-run this cell."
        ) from exc
    print(f"[gradientTrace] hub preflight ok -> https://huggingface.co/{repo_id}", flush=True)


try:  # Kaggle secrets -> env
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            value = client.get_secret(key)
        except Exception:
            continue
        if value:
            os.environ[key] = value
except ImportError:
    print("[gradientTrace] kaggle_secrets unavailable; using env vars only")

try:  # Kaggle silently blocks the clone/HF download when Internet is off
    socket.create_connection(("huggingface.co", 443), timeout=10).close()
except OSError as exc:
    raise RuntimeError(
        "No internet: set Settings > Internet > On (Kaggle asks for phone verification), "
        "then re-run this cell."
    ) from exc

REPO_DIR = "/kaggle/working/gradientTrace"
if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
    if os.path.isdir(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    subprocess.run(
        f"git clone --depth 1 https://github.com/Yashhh999/gradientTrace.git {REPO_DIR}",
        shell=True,
        check=True,
    )
else:  # keep runs/ so RESUME works within a session
    subprocess.run(f"git -C {REPO_DIR} pull --ff-only", shell=True, check=True)
os.chdir(REPO_DIR)
run(f"{sys.executable} -m pip -q install -r requirements.txt")

assert CONFIG in ("base", "v2", "pilot"), f"unknown CONFIG '{CONFIG}'"
DEFAULT_HUB = {
    "base": "Yashhh999/gradientTrace",
    "v2": "Yashhh999/gradientTrace-v2",
    "pilot": "Yashhh999/gradientTrace-v2-pilot",
}
HUB_ID = HUB_ID or DEFAULT_HUB[CONFIG]
push = RUN == "full" or PUSH_SMOKE
hub = HUB_ID if RUN == "full" else HUB_ID + "-smoke"
if push:
    assert os.environ.get("HF_TOKEN"), "Add HF_TOKEN (write scope) in Add-ons > Secrets to push checkpoints."
    check_hub(hub)

if CONFIG in ("v2", "pilot"):  # paired RAID slice; build it once with colab/build_raid.py
    from huggingface_hub import hf_hub_download

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN is required to download the RAID slice."
    os.makedirs(os.path.join(REPO_DIR, "data"), exist_ok=True)
    try:
        raid_src = hf_hub_download(
            repo_id=RAID_HUB, filename="v2/raid_v2.jsonl", token=os.environ["HF_TOKEN"]
        )
    except Exception as exc:
        raise RuntimeError(
            f"RAID slice missing: run colab/build_raid.py with HUB_ID={RAID_HUB} first ({exc!r})"
        ) from exc
    raid_path = os.path.join(REPO_DIR, "data/raid_v2.jsonl")
    shutil.copy(raid_src, raid_path)
    rows = humans = 0
    human_domains = collections.Counter()
    with open(raid_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            rec = json.loads(line)
            if rec.get("label") == 0:
                humans += 1
                human_domains[rec.get("domain")] += 1
    if humans < 10000 or len(human_domains) < 7:
        # the first slice had 1,233 humans in 3 domains; v2 needs every domain populated
        raise RuntimeError(
            f"RAID slice at {RAID_HUB}/v2/raid_v2.jsonl has {humans} human rows across"
            f" {len(human_domains)} domains; rebuild it with colab/build_raid.py"
            " before running v2/pilot."
        )
    print(
        f"[gradientTrace] raid slice <- {RAID_HUB}/v2/raid_v2.jsonl"
        f" ({rows} rows, {humans} human in {len(human_domains)} domains,"
        f" {os.path.getsize(raid_path) / 1e6:.1f} MB)"
    )

    if not LOCAL_JSONL:  # clean paired news; without it the model re-learns "clean prose = AI"
        try:
            news_src = hf_hub_download(
                repo_id=NEWS_HUB, filename="v2/news_pairs.jsonl", token=os.environ["HF_TOKEN"]
            )
        except Exception as exc:
            raise RuntimeError(
                f"paired clean news missing: run colab/build_news.py with HUB_ID={NEWS_HUB}"
                f" first, or set LOCAL_JSONL to your own paired file ({exc!r})"
            ) from exc
        news_path = os.path.join(REPO_DIR, "data/local.jsonl")
        shutil.copy(news_src, news_path)
        news_humans = news_ai = 0
        news_generators = collections.Counter()
        with open(news_path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("label") == 0:
                    news_humans += 1
                elif rec.get("label") == 1:
                    news_ai += 1
                    news_generators[rec.get("generator")] += 1
        if news_humans < 1000 or news_ai < 1000:
            raise RuntimeError(
                f"paired clean news at {NEWS_HUB}/v2/news_pairs.jsonl has {news_humans} human"
                f" and {news_ai} AI rows; rebuild it with colab/build_news.py"
            )
        print(
            f"[gradientTrace] news pairs <- {NEWS_HUB}/v2/news_pairs.jsonl"
            f" ({news_humans} human, {news_ai} AI from {dict(news_generators)},"
            f" {os.path.getsize(news_path) / 1e6:.1f} MB)"
        )
        if len(news_generators) < 2:
            print("[gradientTrace] warning: one news generator only; the paired AI side is less diverse")

import torch  # safe here: only the TPU runtime cares about a kernel-side torch import

assert torch.cuda.is_available(), "Set Settings > Accelerator > GPU T4 x2, then re-run this cell."
world = torch.cuda.device_count()
print(
    f"[gradientTrace] {world} x {torch.cuda.get_device_name(0)} | torch {torch.__version__}"
    f" | global batch {MICRO_BATCH * GRAD_ACCUM * world}"
)
free_gib, total_gib = (value / 2**30 for value in torch.cuda.mem_get_info(0))
print(f"[gradientTrace] GPU 0 free {free_gib:.1f}/{total_gib:.1f} GiB before launch", flush=True)
if free_gib < 13.0:  # a clean T4 shows ~14 GiB; less means an earlier in-kernel model is resident
    raise RuntimeError(
        f"GPU 0 has only {free_gib:.1f} GiB free. This notebook kernel is still holding GPU "
        "memory (usually a gate/eval model from an earlier cell in the same session). "
        "Restart the session (Run > Restart session) and re-run this cell; killing torchrun "
        "alone will not free the kernel's memory."
    )

if LOCAL_JSONL:
    os.makedirs(os.path.join(REPO_DIR, "data"), exist_ok=True)
    shutil.copy(LOCAL_JSONL, os.path.join(REPO_DIR, "data/local.jsonl"))
    print(f"[gradientTrace] local_jsonl <- {LOCAL_JSONL}")

# torchrun needs a script path; a repo-root launcher keeps the package importable.
launcher = os.path.join(REPO_DIR, "kaggle_launch.py")
with open(launcher, "w", encoding="utf-8") as handle:
    handle.write("from gradienttrace.train import main\n\nraise SystemExit(main())\n")

cmd = [
    sys.executable, "-m", "torch.distributed.run", "--standalone",
    "--nproc_per_node", str(world), launcher,
    "--config", f"configs/{CONFIG}.yaml", "--hub-model-id", hub,
    "--batch-size", str(MICRO_BATCH), "--grad-accum", str(GRAD_ACCUM),
]
if RUN == "smoke":
    cmd += ["--smoke", "--max-steps", "40"]
else:
    if MAX_STEPS:
        cmd += ["--max-steps", str(MAX_STEPS)]
    if SAVE_EVERY:
        cmd += ["--save-steps", str(SAVE_EVERY)]
    if EPOCHS:
        cmd += ["--epochs", str(EPOCHS)]
if push:
    cmd.append("--push-to-hub")
if RESUME:
    cmd.append("--resume")

print("[gradientTrace]", " ".join(cmd))
run(cmd)
print("[gradientTrace] done" + (f" -> https://huggingface.co/{hub}" if push else " (local run, no push)"))
