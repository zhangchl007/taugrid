# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline process acceptance, not a Kubernetes/Ray/Spot recovery test."""

import copy
import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import torch
import train


class RecoveryAcceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.baseline = cls.root / "baseline"
        cls.baseline_result = cls.run_process(cls.baseline)
        if cls.baseline_result.returncode:
            raise RuntimeError(cls.baseline_result.stdout)
        cls.reference, _, _ = train.read_checkpoint(cls.baseline)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def command(output, *extra):
        # Windows venv python.exe is a redirector. Launch the actual interpreter
        # so terminate() targets training itself, not a wrapper leaving it alive.
        executable = (
            str(Path(sys.base_prefix) / "python.exe")
            if os.name == "nt"
            else sys.executable
        )
        return [
            executable,
            str(Path(train.__file__).resolve()),
            "--checkpoint-dir",
            str(output),
            *extra,
        ]

    @staticmethod
    def environment(resume=None):
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("RECOVERY_", "TAU_RESUME_", "TAU_RETRY_"))
        }
        env["PYTHONPATH"] = os.pathsep.join(sys.path)
        if resume is not None:
            env["TAU_RESUME_FROM"] = str(resume)
        return env

    @classmethod
    def run_process(cls, output, resume=None, *extra):
        return subprocess.run(
            cls.command(output, *extra),
            env=cls.environment(resume),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
            check=False,
        )

    def assert_state_equal(self, actual, expected, path="state"):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected), path)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys(), path)
            for key in expected:
                self.assert_state_equal(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected), path)
            self.assertEqual(len(actual), len(expected), path)
            for i, value in enumerate(expected):
                self.assert_state_equal(actual[i], value, f"{path}[{i}]")
        else:
            self.assertEqual(actual, expected, path)

    def test_interrupted_process_exact_continuity(self):
        output = self.root / "interrupted"
        events = []
        lines = []
        inbox = queue.Queue()
        process = subprocess.Popen(
            self.command(output, "--pause-after-step", "6"),
            env=self.environment(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        def read_lines():
            for line in process.stdout:
                inbox.put(line)
            inbox.put(None)

        reader = threading.Thread(target=read_lines, daemon=True)
        reader.start()
        try:
            deadline = time.monotonic() + 90
            while True:
                line = inbox.get(timeout=max(0.01, deadline - time.monotonic()))
                self.assertIsNotNone(line, "trainer exited before test pause")
                lines.append(line)
                if not line.startswith("{"):
                    continue
                event = json.loads(line)
                events.append(event)
                if event["event"] == "paused_for_local_test":
                    self.assertEqual(event["pid"], process.pid)
                    break
            interrupted_ns = time.time_ns()
            process.terminate()  # Only this harness-owned process, never a cluster object.
            process.wait(timeout=15)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)
            process.stdin.close()
            reader.join(timeout=15)
            process.stdout.close()
        self.assertNotEqual(process.returncode, 0)
        saved, selected, saved_hash = train.read_checkpoint(output)
        self.assertEqual(saved["step"], 4)
        self.assertEqual(events[-1]["step"], 6)
        resumed = self.run_process(output, selected)
        self.assertEqual(resumed.returncode, 0, resumed.stdout)
        resumed_events = [
            json.loads(line)
            for line in resumed.stdout.splitlines()
            if line.startswith("{")
        ]
        restored = next(e for e in resumed_events if e["event"] == "restored")
        first_update = next(e for e in resumed_events if e["event"] == "update")
        self.assertEqual(restored["step"], 4)
        self.assertEqual(restored["optimizer_steps"], [4.0, 4.0])
        self.assertEqual(restored["samples_seen"], 32)
        self.assertEqual(first_update["step"], 5)
        self.assertGreater(first_update["step"], restored["step"])
        final, final_dir, final_hash = train.read_checkpoint(output)
        self.assertEqual(final["step"], 24)
        self.assert_state_equal(final, self.reference)
        self.assertFalse(
            torch.equal(final["model"]["0.weight"], saved["model"]["0.weight"])
        )
        self.assertEqual(final["optimizer"]["state"][0]["step"].item(), 24)
        self.assertGreater(final["step"], restored["step"])
        # Re-read the subsequent publication in yet another process.
        reread = self.run_process(self.root / "reread", final_dir)
        self.assertEqual(reread.returncode, 0, reread.stdout)
        self.assertIn('"event": "restored"', reread.stdout)
        evidence = {
            "scenario": "direct-trainer-manual-resume",
            "scope": "CPU/local filesystem/separate processes; NOT tau or AKS recovery",
            "result": "PASS",
            "torch": str(torch.__version__),
            "source_sha256": final["config"]["source_sha256"],
            "checkpoint_step": 4,
            "last_update_before_interruption": 6,
            "restored_step": restored["step"],
            "first_resumed_update": first_update["step"],
            "final_step": final["step"],
            "lost_updates_replayed": 2,
            "interrupted_unix_ns": interrupted_ns,
            "restored_unix_ns": restored["unix_ns"],
            "first_resumed_update_unix_ns": first_update["unix_ns"],
            "recovery_seconds": (first_update["unix_ns"] - interrupted_ns) / 1e9,
            "original_pid": process.pid,
            "resumed_pid": restored["pid"],
            "checkpoint_artifact_sha256": saved_hash,
            "final_artifact_sha256": final_hash,
            "exact_state_comparison": [
                "model",
                "Adam steps/exp_avg/exp_avg_sq/param_groups",
                "scheduler",
                "global step",
                "Python RNG",
                "torch RNG",
                "data sampler RNG",
                "samples_seen",
                "config/code/runtime identity",
            ],
            "business_slo": "NOT VALIDATED: thresholds not approved",
            "automatic_tau_recovery": "NOT EXECUTED",
            "gpu_pvc_spot": "NOT EXECUTED",
        }
        destination = os.environ.get("RECOVERY_EVIDENCE_DIR")
        if destination:
            destination = Path(destination)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "process-acceptance.json").write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
            (destination / "interrupted.log").write_text(
                "".join(lines), encoding="utf-8"
            )
            (destination / "resumed.log").write_text(resumed.stdout, encoding="utf-8")
            (destination / "baseline.log").write_text(
                self.baseline_result.stdout, encoding="utf-8"
            )
            shutil.copy2(selected / "checkpoint.zip", destination / "step-4.zip")
            shutil.copy2(
                final_dir / "checkpoint.zip", destination / "resumed-step-24.zip"
            )
            shutil.copy2(
                self.baseline / "step-00000024" / "checkpoint.zip",
                destination / "baseline-step-24.zip",
            )

    def assert_rejected(self, directory, expected):
        result = self.run_process(self.root / "unused-output", directory)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(expected, result.stdout)
        self.assertNotIn('"event": "restored"', result.stdout)
        self.assertNotIn('"event": "update"', result.stdout)

    def checkpoint_copy(self, label):
        destination = self.root / label / "step-00000004"
        shutil.copytree(self.baseline / "step-00000004", destination)
        return destination

    def mutate(self, label, change, manifest_change=None):
        directory = self.checkpoint_copy(label)
        state, _, _ = train.read_checkpoint(directory)
        change(state)
        payload = io.BytesIO()
        torch.save(state, payload)
        raw = payload.getvalue()
        manifest = {
            "format": train.FORMAT,
            "step": state["step"],
            "sha256": train.sha256(raw),
        }
        if manifest_change:
            manifest_change(manifest)
        with zipfile.ZipFile(directory / "checkpoint.zip", "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            archive.writestr("state.pt", raw)
        return directory

    def test_missing_checkpoint(self):
        self.assert_rejected(self.root / "missing", "checkpoint directory missing")

    def test_resume_requires_absolute_nonempty_directory(self):
        for value in ("", "relative-checkpoint"):
            with self.subTest(value=value):
                self.assert_rejected(
                    value, "TAU_RESUME_FROM must be a nonempty absolute directory"
                )

    def test_partial_publication(self):
        directory = self.root / "partial"
        (directory / ".pending-test").mkdir(parents=True)
        (directory / ".pending-test" / "checkpoint.zip").write_bytes(b"partial")
        self.assert_rejected(directory, "no complete checkpoint")

    def test_corrupt_publication_no_fallback(self):
        directory = self.checkpoint_copy("corrupt")
        newer = directory.parent / "step-00000008"
        newer.mkdir()
        (newer / "checkpoint.zip").write_bytes(b"truncated")
        self.assert_rejected(directory.parent, "File is not a zip file")

    def test_integrity_mismatch(self):
        directory = self.mutate(
            "integrity", lambda s: None, lambda m: m.update(sha256="0" * 64)
        )
        self.assert_rejected(directory, "checkpoint integrity mismatch")

    def test_format_mismatch(self):
        directory = self.mutate(
            "format", lambda s: None, lambda m: m.update(format=999)
        )
        self.assert_rejected(directory, "checkpoint format mismatch")

    def test_config_and_code_mismatch(self):
        for field, value in (
            ("seed", -1),
            ("source_sha256", "different-code"),
            ("torch", "incompatible"),
        ):
            with self.subTest(field=field):
                directory = self.mutate(
                    field,
                    lambda s, field=field, value=value: s["config"].update(
                        {field: value}
                    ),
                )
                self.assert_rejected(
                    directory, "checkpoint config/code/runtime identity mismatch"
                )

    def test_incomplete_and_incompatible_state(self):
        cases = [
            (
                "missing-optimizer",
                lambda s: s.pop("optimizer"),
                "state/format mismatch",
            ),
            (
                "optimizer-counter",
                lambda s: s["optimizer"]["state"][0]["step"].fill_(0),
                "optimizer step mismatch",
            ),
            (
                "optimizer-moment",
                lambda s: s["optimizer"]["state"][0].pop("exp_avg"),
                "state keys mismatch",
            ),
            (
                "negative-moment",
                lambda s: s["optimizer"]["state"][0]["exp_avg_sq"].fill_(-1),
                "second moment must be nonnegative",
            ),
            (
                "model-shape",
                lambda s: s["model"].update({"0.weight": torch.zeros(2, 3)}),
                "tensor mismatch",
            ),
            (
                "scheduler",
                lambda s: s["scheduler"].update(last_epoch=0),
                "scheduler state mismatch",
            ),
            (
                "data-progress",
                lambda s: s.update(samples_seen=0),
                "data progress mismatch",
            ),
            (
                "rng",
                lambda s: s.update(torch_rng=torch.zeros(2, dtype=torch.uint8)),
                "tensor mismatch",
            ),
            (
                "nonfinite",
                lambda s: s["model"]["0.bias"].fill_(float("nan")),
                "tensor mismatch",
            ),
        ]
        for label, change, expected in cases:
            with self.subTest(label=label):
                self.assert_rejected(self.mutate(label, change), expected)

    def test_read_permission_failure_propagates(self):
        # Portable I/O unit contract; NOT a PVC identity/ACL acceptance claim.
        with (
            mock.patch.object(
                Path, "read_bytes", side_effect=PermissionError("test denied")
            ),
            self.assertRaisesRegex(PermissionError, "test denied"),
        ):
            train.read_checkpoint(self.baseline / "step-00000004")

    def test_fresh_run_cannot_overwrite_checkpoint(self):
        directory = self.checkpoint_copy("overwrite")
        before = (directory / "checkpoint.zip").read_bytes()
        result = self.run_process(directory.parent)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh run requires an empty checkpoint directory", result.stdout)
        self.assertEqual(before, (directory / "checkpoint.zip").read_bytes())

    def test_invalid_state_does_not_apply_model(self):
        state = copy.deepcopy(self.reference)
        state["scheduler"]["last_epoch"] = 0
        with mock.patch.object(torch.nn.Module, "load_state_dict") as apply:
            with self.assertRaisesRegex(ValueError, "scheduler state mismatch"):
                train.validate_state(state, self.reference["config"], "cpu")
            apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
