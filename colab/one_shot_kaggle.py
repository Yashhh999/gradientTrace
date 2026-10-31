# gradientTrace - one-shot Kaggle cell
# Paste this whole file into a single Kaggle notebook cell.
# Notebook settings: Accelerator = GPU T4 x2, Internet = On.
# Add-ons > Secrets: HF_TOKEN (write scope) for Hub checkpoints; WANDB_API_KEY optional.
RUN = "smoke"                        # "smoke" (~10 min sanity) | "full" (real training)
CONFIG = "base"                      # base | v2 | pilot -> configs/<CONFIG>.yaml
HUB_ID = ""                          # "" = per-CONFIG default (gradientTrace / -v2 / -v2-pilot)
RAID_HUB = "Yashhh999/gradientTrace" # repo holding v2/raid_v2.jsonl from colab/build_raid.py
PUSH_SMOKE = False                   # also push the smoke model to "<HUB_ID>-smoke"
MAX_STEPS = 0                        # full run: stop after this many optimizer steps per session
                                     # (RESUME keeps counting from the checkpoint); 0 = all EPOCHS
SAVE_EVERY = 300                     # full run: checkpoint + eval every N steps, pushed to the Hub
EPOCHS = None                        # None = config (3); e.g. 1 for a single full-data pass
RESUME = False                       # continue from the newest checkpoint: local first, else the Hub
MICRO_BATCH = 4                      # per GPU; 4 x 8 accum x 2 GPUs = 64 global batch
GRAD_ACCUM = 8
LOCAL_JSONL = ""                     # e.g. "/kaggle/input/gradienttrace/local.jsonl"

import os, shutil, socket, subprocess, sys


def run(command, check=True):
    """Stream a child's output through the notebook; Kaggle hides raw child fds."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
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
        proc.wait()
    if check and proc.returncode:
        raise RuntimeError(f"command failed with exit {proc.returncode}: {command}")
    return proc.returncode


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

if CONFIG in ("v2", "pilot"):  # paired RAID slice; build it once with colab/build_raid.py
    from huggingface_hub import hf_hub_download

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN is required to download the private RAID slice."
    os.makedirs(os.path.join(REPO_DIR, "data"), exist_ok=True)
    try:
        raid_src = hf_hub_download(
            repo_id=RAID_HUB, filename="v2/raid_v2.jsonl", token=os.environ["HF_TOKEN"]
        )
    except Exception as exc:
        raise RuntimeError(
            f"RAID slice missing: run colab/build_raid.py with HUB_ID={RAID_HUB} first ({exc!r})"
        ) from exc
    shutil.copy(raid_src, os.path.join(REPO_DIR, "data/raid_v2.jsonl"))
    print(
        f"[gradientTrace] raid slice <- {RAID_HUB}/v2/raid_v2.jsonl"
        f" ({os.path.getsize(os.path.join(REPO_DIR, 'data/raid_v2.jsonl')) / 1e6:.1f} MB)"
    )

import torch  # safe here: only the TPU runtime cares about a kernel-side torch import

assert torch.cuda.is_available(), "Set Settings > Accelerator > GPU T4 x2, then re-run this cell."
world = torch.cuda.device_count()
print(
    f"[gradientTrace] {world} x {torch.cuda.get_device_name(0)} | torch {torch.__version__}"
    f" | global batch {MICRO_BATCH * GRAD_ACCUM * world}"
)

if LOCAL_JSONL:
    os.makedirs(os.path.join(REPO_DIR, "data"), exist_ok=True)
    shutil.copy(LOCAL_JSONL, os.path.join(REPO_DIR, "data/local.jsonl"))
    print(f"[gradientTrace] local_jsonl <- {LOCAL_JSONL}")

# torchrun needs a script path; a repo-root launcher keeps the package importable.
launcher = os.path.join(REPO_DIR, "kaggle_launch.py")
with open(launcher, "w", encoding="utf-8") as handle:
    handle.write("from gradienttrace.train import main\n\nraise SystemExit(main())\n")

push = RUN == "full" or PUSH_SMOKE
hub = HUB_ID if RUN == "full" else HUB_ID + "-smoke"
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
    assert os.environ.get("HF_TOKEN"), "Add HF_TOKEN (write scope) in Add-ons > Secrets to push checkpoints."
    cmd.append("--push-to-hub")
if RESUME:
    cmd.append("--resume")

print("[gradientTrace]", " ".join(cmd))
run(cmd)
print("[gradientTrace] done" + (f" -> https://huggingface.co/{hub}" if push else " (local run, no push)"))
