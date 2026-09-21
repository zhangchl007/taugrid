# Checkpoint / recovery acceptance

> **Status:** `operator runbook`
> **Intended use:** deterministic offline state-continuity acceptance and an
> approval-gated, single-GPU durable recovery exercise.
> **Not for:** claiming GPU/PVC/Spot recovery from a restarted Pod or CPU test.

This is separate from `aks-gpu-quickstart/train.py`: its historical
`gpu-evidence.json` is not a training checkpoint. Nothing here changes that smoke.

## Checkpoint contract

`train.py` trains a tiny real PyTorch linear model with dropout, Adam and StepLR.
Synthetic data is fixed; Python RNG changes input scaling, the torch RNG drives
dropout, and a separate saved generator samples data. Checkpoints contain model
weights, both Adam moments and per-parameter step counters, optimizer settings,
scheduler, global step, samples seen, Python/torch/device RNG, sampler RNG,
format version, training config, runtime versions and the exact source hash.
There is no AMP scaler or epoch permutation to restore in this trainer.

Every four updates (and at completion), the writer creates a `.pending-*`
directory, writes a ZIP with a SHA-256-checked `state.pt` and manifest, flushes and
fsyncs, then atomically renames the directory to `step-00000004`, etc. POSIX
directory fsync is included; Windows tests establish process-interruption
behavior, not power-loss durability. Verify the actual PVC filesystem's rename
and fsync guarantees before calling this durable. There must be **one writer**
per checkpoint root; no overwrite or multi-writer coordination is supported.

`TAU_RESUME_FROM` must be an absolute directory, either an exact `step-*`
publication (preferred for manual recovery) or the scenario's root (automatic
retry selects the highest published step). `.pending-*` is never selected. A
corrupt newest publication fails; there is no silent fallback to an older step,
CPU fallback, or fresh training. An empty resume variable also fails. A fresh
run refuses a nonempty checkpoint root. Keep pending files for diagnosis.

ZIP integrity and SHA-256 are checked before `torch.load(weights_only=True)`.
The complete model/optimizer/scheduler/RNG/data schema and progress consistency
are validated before applying training state. Hashes detect corruption, **not
authenticity**: only load your own trusted test artifacts. Code, configuration,
device and runtime changes are incompatible; this example has no migration path.

JSON events include `checkpoint_committed` with the actual ZIP hash, `restored`
with optimizer counters and step, `update`, and `completed`. Wall-clock timestamps
and process IDs support correlation; Kubernetes UIDs and node identities must
be collected separately. This tiny trainer is not a GPU performance benchmark.

## Offline acceptance (no cluster required)

From the repository root, using Python 3.12:

```powershell
python -m venv .venv-recovery
.\.venv-recovery\Scripts\python -m pip install -r examples\checkpoint-recovery\requirements.txt --index-url https://download.pytorch.org/whl/cpu
$env:RECOVERY_EVIDENCE_DIR = Join-Path $env:TEMP "tau-recovery-evidence"
.\.venv-recovery\Scripts\python -m unittest discover -s examples\checkpoint-recovery -v
```

On Linux use `.venv-recovery/bin/python` and native path separators. Evidence is
optional; without `RECOVERY_EVIDENCE_DIR`, all data remains in temporary test
directories. The tests launch a baseline and an independent trainer, pause only
the owned test child at update 6 after publishing step 4, terminate that child,
and explicitly resume in another process. They require restored step 4, first
new update 5, final checkpoint 24 and exact tensor/state equality with baseline.
The two uncheckpointed updates are replayed; decreasing loss alone is not proof.
A fourth process re-reads the final checkpoint.

Negative tests use copies: missing, partial, corrupt newest, checksum, format,
config/code/runtime, missing optimizer, counter/moment/shape/scheduler/data/RNG
mismatch, nonfinite state, invalid resume path, and accidental fresh overwrite.
Permission failure is a mocked I/O unit test, **not** a new PVC identity test.

The separate offline CLI contract tests use the real config loader, Job renderer,
resume preflight, retry loop and environment helper with synthetic status/hooks.
No cluster status is patched:

```powershell
Push-Location cli
go test -count=1 ./internal/cli -run 'Test(CheckpointRecovery|Resume|RunResume|ResolveResume|Retry|AppendRetry)'
go test -count=1 ./internal/resume ./internal/jobrender
Pop-Location
Push-Location core
go test -count=1 ./runconfig
Pop-Location
```

These assert manual/automatic config isolation, durable `/data`, a single GPU
Job with Kubernetes `backoffLimit: 0` and `restartPolicy: Never`, one configured
Tau retry, injected resume/attempt/max/reason values, bounded exhaustion and
rejection of `Unknown`. They are **simulated CLI orchestration**, not real
automatic process, Kubernetes, capacity or Spot recovery.

## Approval-gated AKS proposal (NOT EXECUTED)

**Commit, push and review before deployment.** Owner approval is separately
required for billable storage, workload submission and the exact fault. An
unavailable owner is not approval. Do not provision, delete Pods, scale or evict
nodes, restart stopped resources, or alter the deployed platform from this guide
without the corresponding approval.

| Item | Exact proposed scope |
| --- | --- |
| Existing target | `taugrid-phase1-itn`, RG `rg-taugrid-phase1-itn`, Italy North; recheck before use |
| Namespace / queue | Existing `taugrid-default` / `jobqueue`; verify current applicable Job/GPU profile |
| Workloads | `recovery-manual`, then `recovery-auto`; never simultaneously |
| PVC | `recovery-checkpoints`, `pvc-gated.yaml`, 1Gi requested, `ReadWriteOnce`, class `default` |
| Expected class | Azure Disk `disk.csi.azure.com`, `StandardSSD_LRS`, `WaitForFirstConsumer`, reclaim `Delete`; stop if different |
| Billing / topology | 1Gi request may bill the minimum managed-disk tier (typically E1, 4Gi); verify current regional pricing. Zonal RWO allows sequential same-zone reattachment, not concurrent multi-node or cross-zone recovery |
| Per Job | 1 GPU; requests 1 CPU / 2Gi RAM; limits 2 CPU / 4Gi RAM; 400 updates, 0.25s pacing; checkpoint every 4 updates |
| Runtime | MCR `acpt-pytorch-2.8-cuda12.6:17` pinned by digest in configs; registry metadata verified Linux amd64 / Python 3.10 / CUDA 12.6.3; container/GPU execution still unverified |
| Retention | Retain PVC and logs through owner evidence sign-off; a retained disk continues billing |

The local PyTorch 2.8 CPU test is a different Python/device build, not proof of
the container's memory limit or GPU determinism. At approved startup verify the
actual `python3`, torch version, CUDA device, available memory and filesystem
permissions. Runtime versions are persisted in checkpoints; compare GPU baseline
and resumed GPU state on the same pinned runtime. Do not reuse CPU checkpoints.
Direct Jobs cannot use `runtime.pip`. A RayJob is deliberately not used: the
current renderer mounts the PVC on both a system-pool head and GPU worker, which
cannot share this RWO disk across those nodes.

### Read-only preflight

Use a verified isolated kubeconfig/context; never paste its contents or credentials
into logs. Current source contracts, not an old installed binary, are authoritative:

```powershell
tau run resume --help
tau run validate --config examples\checkpoint-recovery\tau-manual.yaml
tau run validate --config examples\checkpoint-recovery\tau-auto.yaml
kubectl --context taugrid-phase1-itn-admin get nodes
kubectl --context taugrid-phase1-itn-admin get storageclass default -o yaml
kubectl --context taugrid-phase1-itn-admin -n taugrid-default get pvc,job,pod
```

Before an approved submission, resolve the current Ready workload profile and
server-dry-run each config using that reviewed routing. Profile selection,
admission, image availability and mounts must pass; no invented profile names.
`WaitForFirstConsumer` means a new claim may remain Pending until its approved
consumer is scheduled; verify it is Bound before accepting checkpoint durability.

### Commands reserved for an approved operator

The following create billable storage / workloads. **Do not run now.** Check every
exit code and stop on error. Run one scenario to completion before the other.

```powershell
kubectl --context taugrid-phase1-itn-admin apply -f examples\checkpoint-recovery\pvc-gated.yaml
tau run --config examples\checkpoint-recovery\tau-manual.yaml --context taugrid-phase1-itn-admin --namespace taugrid-default --dry-run=server
tau run --config examples\checkpoint-recovery\tau-manual.yaml --context taugrid-phase1-itn-admin --namespace taugrid-default
```

Archive Job/Pod UIDs, node, full logs, events and a published checkpoint hash
**before** any approved fault. First approve only interruption of the named
trainer process / its single Pod, not a node or cluster-wide cleanup.
**This scope does not guarantee a retryable failure:** ordinary process exit or
Pod deletion can classify as `Unknown`. In that case record the failed recovery
precondition; do not patch status, widen `retry_on`, force an OOM, or call it
automatic recovery. A genuine observed `Preempted` or `Evicted` terminal failure
is required for the configs below. Any broader fault technique needs a new
specific owner gate. No fault command is supplied that disguises this mismatch.

For a still-existing, genuinely retryable failed **manual** Job only, select the
exact committed directory from its log (replace the illustrative step):

```powershell
tau run status recovery-manual --context taugrid-phase1-itn-admin -n taugrid-default
tau run resume recovery-manual --config examples\checkpoint-recovery\tau-manual.yaml --from /data/checkpoints/finetunes/recovery-manual/step-00000004 --context taugrid-phase1-itn-admin -n taugrid-default
```

`--config` is mandatory. A succeeded workload does not qualify. Resume validates
the replacement, deletes the failed Job, then resubmits: archive evidence first.
OOM requires `--force` and a resolved resource issue; it is not the proposed test.

For the distinct **automatic** scenario, keep the submitting CLI alive:

```powershell
tau run --config examples\checkpoint-recovery\tau-auto.yaml --context taugrid-phase1-itn-admin --namespace taugrid-default --dry-run=server
tau run --config examples\checkpoint-recovery\tau-auto.yaml --context taugrid-phase1-itn-admin --namespace taugrid-default
```

There is no `tau run retry` command and no independent retry daemon. Do not
manually resubmit this scenario. Require actual failure classification, a new
Job UID, retry environment, restored step greater than zero, first later update,
new checkpoint and terminal success. No approved fault / no capacity means
Blocked or Not executed, never PASS. `max_retries: 1` bounds replacements,
**not pending admission time**: use an owner-agreed observation deadline and
record capacity waiting separately; do not claim a queue timeout was tested.

After copying and independently validating evidence, obtain explicit cleanup
approval. Delete only the two named test Jobs (using current approved UID checks)
and this named PVC; never delete the namespace or unrelated objects. PVC deletion
with reclaim `Delete` destroys the disk and checkpoint copies. Wait for disk
deletion confirmation; do not remove the sole good checkpoint before export.

## Evidence boundaries

Record scenario, original/replacement identities, node/zone, checkpoint/hash/step,
last pre-fault update, first resumed update, terminal outcome and timestamps.
Recovery time is interruption to the first successful new update; report
capacity waiting separately. Lost work is last update minus checkpoint step.
All timings are measurements, not a business SLO PASS without agreed thresholds.
Zero real recovery samples means **Not validated**, not a 0% or 100% success rate.

| Surface | What offline acceptance establishes | Still needs an approved live exercise |
| --- | --- | --- |
| Save/load | Complete local atomic publication and fail-closed validation | PVC durability, permissions and cross-Pod read |
| Manual | New CPU process restores full state and exactly matches baseline | Real `tau run resume`, GPU/PVC and worker identity |
| Automatic | Simulated real CLI policy/env/bounded retries | Observed classified failure and unattended GPU completion |
| Capacity / Spot | Nothing | Capacity wait, replacement node/zone, genuine interruption |
