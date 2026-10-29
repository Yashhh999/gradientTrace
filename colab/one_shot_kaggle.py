# gradientTrace - one-shot Kaggle cell
# Paste this whole file into a single Kaggle notebook cell.
# Notebook settings: Accelerator = GPU T4 x2, Internet = On.
# Add-ons > Secrets: HF_TOKEN (write scope) for Hub checkpoints; WANDB_API_KEY optional.
RUN = "smoke"                        # "smoke" (~10 min sanity) | "full" (real training)
HUB_ID = "Yashhh999/gradientTrace"   # HF model repo to receive checkpoints
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
    from kaggle_secrets import UserSecretClient

    client = UserSecretClient()
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            value = client.get_secret(key)
        except Exception:
            continue
        if value:
            os.environ[key] = value
except ImportError:
    pass

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
    "--config", "configs/base.yaml", "--hub-model-id", hub,
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
