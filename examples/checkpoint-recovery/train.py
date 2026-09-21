# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded, single-worker training with fail-closed complete checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import torch

FORMAT = 1
CHECKPOINT_NAME = re.compile(r"step-[0-9]{8}")


def emit(event: str, **fields) -> None:
    print(
        json.dumps(
            {"event": event, "unix_ns": time.time_ns(), "pid": os.getpid(), **fields},
            sort_keys=True,
        ),
        flush=True,
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sync_directory(directory: Path) -> None:
    if os.name != "nt":
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def save_checkpoint(root: Path, state: dict) -> Path:
    """Publish one immutable directory; readers never inspect .pending-*."""
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step-{state['step']:08d}"
    if destination.exists():
        raise ValueError(f"refusing to overwrite checkpoint: {destination}")
    payload = io.BytesIO()
    torch.save(state, payload)
    raw = payload.getvalue()
    manifest = {"format": FORMAT, "sha256": sha256(raw), "step": state["step"]}
    pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=root))
    archive = pending / "checkpoint.zip"
    with archive.open("wb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as bundle:
            bundle.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
            bundle.writestr("state.pt", raw)
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(pending)
    pending.rename(destination)
    sync_directory(root)
    emit(
        "checkpoint_committed",
        step=state["step"],
        checkpoint=str(destination),
        artifact_sha256=sha256((destination / "checkpoint.zip").read_bytes()),
        state_sha256=manifest["sha256"],
    )
    return destination


def read_checkpoint(directory: Path) -> tuple[dict, Path, str]:
    if not directory.is_dir():
        raise ValueError(f"checkpoint directory missing: {directory}")
    if not CHECKPOINT_NAME.fullmatch(directory.name):
        candidates = sorted(
            p for p in directory.iterdir() if CHECKPOINT_NAME.fullmatch(p.name)
        )
        if not candidates:
            raise ValueError(f"no complete checkpoint in {directory}")
        directory = candidates[-1]
    # Fail on a damaged newest publication, rather than quietly loading older state.
    artifact = (directory / "checkpoint.zip").read_bytes()
    with zipfile.ZipFile(io.BytesIO(artifact)) as bundle:
        if sorted(bundle.namelist()) != ["manifest.json", "state.pt"]:
            raise ValueError("checkpoint archive members mismatch")
        manifest = json.loads(bundle.read("manifest.json"))
        raw = bundle.read("state.pt")
    if (
        set(manifest) != {"format", "sha256", "step"}
        or type(manifest["format"]) is not int
        or manifest["format"] != FORMAT
    ):
        raise ValueError("checkpoint format mismatch")
    if sha256(raw) != manifest["sha256"]:
        raise ValueError("checkpoint integrity mismatch")
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("step") != manifest["step"]:
        raise ValueError("checkpoint progress mismatch")
    if directory.name != f"step-{state['step']:08d}":
        raise ValueError("checkpoint directory/progress mismatch")
    return state, directory, sha256(artifact)


def validate_tree(actual, expected, location: str) -> None:
    """Check the entire known state schema before any load_state_dict call."""
    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or not torch.isfinite(actual).all()
        ):
            raise ValueError(f"checkpoint tensor mismatch: {location}")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"checkpoint state keys mismatch: {location}")
        for key, value in expected.items():
            validate_tree(actual[key], value, f"{location}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError(f"checkpoint sequence mismatch: {location}")
        for index, value in enumerate(expected):
            validate_tree(actual[index], value, f"{location}[{index}]")
    elif type(actual) is not type(expected):
        raise ValueError(f"checkpoint type mismatch: {location}")
    elif isinstance(actual, float) and not math.isfinite(actual):
        raise ValueError(f"checkpoint nonfinite value: {location}")


def build_training(device: str):
    model = torch.nn.Sequential(torch.nn.Linear(3, 1), torch.nn.Dropout(0.2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8)
    return model, optimizer, scheduler


def validate_state(state: dict, config: dict, device: str) -> None:
    required = {
        "format",
        "config",
        "step",
        "model",
        "optimizer",
        "scheduler",
        "torch_rng",
        "python_rng",
        "cuda_rng",
        "data_rng",
        "samples_seen",
    }
    if (
        state.keys() != required
        or type(state["format"]) is not int
        or state["format"] != FORMAT
    ):
        raise ValueError("checkpoint state/format mismatch")
    if state["config"] != config:
        raise ValueError("checkpoint config/code/runtime identity mismatch")
    step = state["step"]
    if type(step) is not int or not 0 < step <= config["steps"]:
        raise ValueError("checkpoint step out of range")
    if type(state["samples_seen"]) is not int or state["samples_seen"] != step * 8:
        raise ValueError("checkpoint data progress mismatch")
    # Isolated CPU objects provide PyTorch's exact version-specific state schema.
    probe, optimizer, scheduler = build_training("cpu")
    probe(torch.zeros(8, 3)).sum().backward()
    optimizer.step()
    scheduler.step()
    validate_tree(state["model"], probe.state_dict(), "model")
    validate_tree(state["optimizer"], optimizer.state_dict(), "optimizer")
    validate_tree(state["scheduler"], scheduler.state_dict(), "scheduler")
    for slot in state["optimizer"]["state"].values():
        if slot["step"].item() != step:
            raise ValueError("checkpoint optimizer step mismatch")
        if (slot["exp_avg_sq"] < 0).any():
            raise ValueError("checkpoint optimizer second moment must be nonnegative")
    expected_scheduler = scheduler.state_dict()
    expected_scheduler.update(
        last_epoch=step, _step_count=step + 1, _last_lr=[0.03 * 0.8 ** (step // 4)]
    )
    for key, value in expected_scheduler.items():
        actual = state["scheduler"][key]
        if key == "_last_lr":
            if not math.isclose(actual[0], value[0], rel_tol=1e-12):
                raise ValueError("checkpoint scheduler learning rate mismatch")
        elif actual != value:
            raise ValueError(f"checkpoint scheduler state mismatch: {key}")
    expected_group = optimizer.state_dict()["param_groups"][0]
    for key, value in expected_group.items():
        actual = state["optimizer"]["param_groups"][0][key]
        if key == "lr":
            if not math.isclose(
                actual, expected_scheduler["_last_lr"][0], rel_tol=1e-12
            ):
                raise ValueError("checkpoint optimizer learning rate mismatch")
        elif actual != value:
            raise ValueError(f"checkpoint optimizer configuration mismatch: {key}")
    for key in ("torch_rng", "data_rng"):
        validate_tree(state[key], torch.get_rng_state(), key)
        torch.Generator().set_state(state[key])
    random.Random().setstate(state["python_rng"])
    if device == "cpu":
        if state["cuda_rng"] is not None:
            raise ValueError("checkpoint unexpected CUDA RNG")
    else:
        validate_tree(state["cuda_rng"], torch.cuda.get_rng_state(), "cuda_rng")
        torch.Generator(device=device).set_state(state["cuda_rng"])


def train(args: argparse.Namespace) -> None:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA required; refusing CPU fallback")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = {
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": 8,
        "dataset": "analytic-32-v1",
        "model": "linear3-dropout-adam-steplr-v1",
        "device": args.device,
        "torch": str(torch.__version__),
        "python": tuple(sys.version_info[:3]),
        "source_sha256": args.source_sha256,
    }
    model, optimizer, scheduler = build_training(args.device)
    data_rng = torch.Generator().manual_seed(args.seed + 1)
    start = 0
    resume_from = os.environ.get("TAU_RESUME_FROM")
    root = args.checkpoint_dir
    if resume_from is not None:
        if not resume_from.strip() or not Path(resume_from).is_absolute():
            raise ValueError("TAU_RESUME_FROM must be a nonempty absolute directory")
        state, selected, digest = read_checkpoint(Path(resume_from))
        validate_state(state, config, args.device)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        data_rng.set_state(state["data_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        start = state["step"]
        emit(
            "restored",
            step=start,
            checkpoint=str(selected),
            artifact_sha256=digest,
            optimizer_steps=[slot["step"].item() for slot in optimizer.state.values()],
            samples_seen=state["samples_seen"],
        )
    elif root.exists() and any(root.iterdir()):
        raise ValueError(
            "fresh run requires an empty checkpoint directory; set TAU_RESUME_FROM"
        )
    emit(
        "training_started",
        start_step=start,
        device=args.device,
        retry_attempt=os.environ.get("TAU_RETRY_ATTEMPT"),
        retry_max=os.environ.get("TAU_RETRY_MAX"),
        retry_reason=os.environ.get("TAU_RETRY_REASON"),
    )
    values = torch.arange(96, dtype=torch.float32).reshape(32, 3) / 96
    targets = values @ torch.tensor([[0.4], [-0.2], [0.8]]) + 0.1
    for step in range(start + 1, args.steps + 1):
        indices = torch.randint(32, (8,), generator=data_rng)
        x = (values[indices] * (0.99 + random.random() * 0.02)).to(args.device)
        y = targets[indices].to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(x), y)
        if not torch.isfinite(loss):
            raise ValueError("training produced nonfinite loss")
        loss.backward()
        optimizer.step()
        scheduler.step()
        emit("update", step=step, loss=loss.item())
        if step % args.checkpoint_every == 0 or step == args.steps:
            state = {
                "format": FORMAT,
                "config": config,
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state()
                if args.device == "cuda"
                else None,
                "data_rng": data_rng.get_state(),
                "samples_seen": step * 8,
            }
            save_checkpoint(root, state)
        if args.pause_after_step == step:
            emit("paused_for_local_test", step=step)
            if input() != "continue":
                raise ValueError("local test continuation not received")
        if args.step_delay:
            time.sleep(args.step_delay)
    emit("completed", step=args.steps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=os.environ.get("RECOVERY_CHECKPOINT_DIR")
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=os.environ.get("RECOVERY_DEVICE", "cpu"),
    )
    parser.add_argument(
        "--steps", type=int, default=int(os.environ.get("RECOVERY_STEPS", "24"))
    )
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument(
        "--step-delay",
        type=float,
        default=float(os.environ.get("RECOVERY_STEP_DELAY", "0")),
    )
    parser.add_argument(
        "--pause-after-step",
        type=int,
        default=0,
        help="offline harness only: block on stdin after this update",
    )
    args = parser.parse_args()
    if args.checkpoint_dir is None or not args.checkpoint_dir.is_absolute():
        parser.error("--checkpoint-dir / RECOVERY_CHECKPOINT_DIR must be absolute")
    if not 1 <= args.steps <= 400 or not 1 <= args.checkpoint_every <= args.steps:
        parser.error("require 1 <= checkpoint-every <= steps <= 400")
    if not 0 <= args.step_delay <= 1:
        parser.error("step-delay must be between 0 and 1 seconds")
    if not 0 <= args.pause_after_step < args.steps:
        parser.error("pause-after-step must be zero or less than steps")
    args.source_sha256 = sha256(Path(__file__).read_bytes())
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
