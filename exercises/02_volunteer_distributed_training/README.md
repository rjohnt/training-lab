# Volunteer GPU training across Windows PCs

Status: **planned; not implemented or executed**. No worker image, launcher,
coordinator, deployment, invitation or benchmark result exists yet.

Question: can 2–3 friends contribute their Windows GPUs to one training run with
little setup, clear control over participation, and a measurable improvement in
time to a target validation quality?

## Intended participant experience

Assume NVIDIA GPUs initially; confirm GPU models, VRAM, Windows versions, available
disk space and upload speeds before selecting the model or runtime. Other GPU
vendors require a separate compatibility decision.

1. Complete a one-time setup of Docker Desktop with its WSL2 backend and a
   compatible Windows NVIDIA driver. Setup can require administrator access and
   a restart; this is not a zero-install application.
2. Download a reviewed release with `start-worker.ps1` and `stop-worker.ps1`.
3. Start the worker, enter an individual join code through a non-echoing prompt,
   select a GPU and choose a session duration (initial default: 30 minutes).
4. See local progress, resource use, download size and remaining session time.
5. Stop at any time. No automatic startup or unattended installation of updates.

The launch script will check compatibility and pull an immutable container image
pinned by digest. The image will contain Python, PyTorch, CUDA runtime libraries,
the training program and its communication client. The host still supplies the
GPU driver; participants do not need a separate host CUDA development toolkit.
Any custom extensions must be built for the supported GPUs or compiled inside
the image, with first-run costs reported separately.

Model weights and approved dataset shards will download separately into a
dedicated cache, with revisions/checksums recorded. Confirm model and dataset
redistribution/access terms before release. If the image is private, document
registry access separately; public source/image publication is not authorized
by this plan.

## Coordinator and network design

Use one coordinator with authenticated HTTPS connections initiated by workers.
Each friend's worker and the owner's worker contact that endpoint independently.
Participants do not need incoming ports, router changes, SSH access or a VPN.
Hosting, TLS, bandwidth and availability of the coordinator are our responsibility;
its placement and funding remain undecided.

```text
Windows worker A ── outbound HTTPS ──┐
Windows worker B ── outbound HTTPS ──┼── Coordinator ── checkpoints / metrics
Windows worker C ── outbound HTTPS ──┤
Owner's worker  ── outbound HTTPS ──┘
```

Workers authenticate, fetch an agreed run manifest and model version, train a
local round, upload updates, and fetch the next accepted version. Polling or a
worker-initiated persistent connection can carry control messages. The coordinator
issues declarative work for the fixed reviewed training program, not arbitrary
shell commands or remotely supplied executable code.

This is a proposed application-level training protocol, **not ordinary PyTorch
DDP transported over HTTPS**. Docker packages dependencies; it does not supply
the aggregation algorithm, fault tolerance or secure coordinator. A private
Tailscale connection to the coordinator is an optional alternative, not a
participant prerequisite for the HTTPS design.

## Initial training algorithm

Start with LoRA fine-tuning of a small model that fits the least-capable accepted
GPU. Every worker holds the same frozen base model and compatible adapter
structure; this combines compute, not VRAM. Local accumulation can reduce local
memory requirements for a chosen effective batch, while periodic adapter updates
reduce communication compared with full-parameter synchronization.

Begin with a documented, synchronous round-based local-update baseline. Workers
start from the same adapter version, run a specified number of local optimizer
steps on assigned examples, and return adapter deltas. Aggregate accepted deltas
weighted by the number of examples actually processed. Fix sequence length and
loss normalization initially, and record non-padding tokens as well as examples.
Local updates change the optimization trajectory; they are not equivalent to
accumulating gradients at unchanged weights.

Specify the optimizer, learning rates, local-step count, adapter rank/targets,
optimizer-state reset or retention policy, aggregation weights and participant
sampling before execution. Averaging LoRA factors is a baseline to evaluate:
it is not mathematically identical to averaging their dense weight updates.
DiLoCo-style outer optimization is a later comparison, not a label for arbitrary
adapter averaging.

Use versioned rounds, deadlines, a minimum accepted contribution policy, and
idempotent uploads. Reject stale, duplicate, non-finite or incompatible updates.
Late workers must fetch the current version before rejoining. Record dropped
contributions and slower-worker exclusion, since these can bias data exposure.
Checkpoint the global adapters, coordinator state and relevant optimizer state.
A worker's departure must not corrupt a checkpoint; a round may wait, continue
with eligible contributors, or abort according to a predeclared policy.

## Participant control and credentials

- Use a dedicated cache mount, a non-root container process where supported,
  and no home-directory, Docker-socket or privileged host access.
- Use individual, scoped, revocable worker credentials. Exchange short-lived
  join codes for worker credentials and store them outside Git using local secret
  storage. Do not put secrets in launch arguments, images, logs or screenshots.
- Pin the image digest and link it to reviewed source. Updates require an
  explicit new release/start decision. Containers reduce installation impact
  but do not remove the need to trust GPU-executing code.
- Bound session time, CPU/RAM use and workload size. Consumer GPUs generally do
  not offer a hard per-container VRAM or utilization quota; a conservative batch
  size is not a guaranteed partition. Schedule sessions while participants are
  away initially, because desktop/gaming responsiveness can suffer.
- Show cache size and provide explicit cache cleanup instructions separately
  from stopping the worker. Preserve the participant's ability to interrupt work.
- Report only consented experiment metadata. Keep identities, addresses,
  credentials, private connection configuration and raw logs out of published
  artifacts. Use anonymous worker labels in reviewed results.

## Reproducible experiment

Compare a single GPU baseline, synchronized distributed training where feasible,
and periodic local updates with 2–3 friend workers plus the owner's worker.
The synchronized control needs its own validated transport; do not assume DDP
will work through the outbound-only coordinator. An application-level gradient
aggregation control is another option if its numerics are validated.

Keep the base model, initial adapters, tokenizer, train/validation split, data
ordering rules, precision and evaluation procedure fixed and versioned. Record
effective global batch and total training tokens; distinguish batch-size effects
from communication effects. Retain both fixed-token-budget comparisons and time
to a predeclared validation target, with repeated seeds and run variability.

Measure:

- Held-out validation loss against elapsed time and tokens processed.
- Time to the same validation target; explicitly report runs that never reach it.
- Aggregate tokens/second, per-worker compute/wait time and participation.
- Bytes uploaded/downloaded, synchronization duration and coordinator traffic.
- Peak GPU memory, hardware/runtime versions and disconnect recovery behavior.
- Cold onboarding/download time separately from warm training time. Count
  required communication, evaluation and checkpoint work in end-to-end results.

First test protocol correctness locally: one-worker equivalence to a reference,
known weighted aggregation, duplicate/stale update handling and checkpoint
restart. Then run a short Windows pilot with one friend before the full cohort.
Choose sustained run durations after the pilot, freeze the protocol, and keep
profiling separate from uninstrumented timing. Do not poll expensive GPU metrics
inside every training step.

Overlay methods on the same axes for validation loss versus time, loss versus
tokens, throughput and communication costs. Include a participant/coordinator
diagram and a measured compute/wait timeline. Generate PNG and SVG artifacts,
embed PNGs here and link SVGs when results exist. No measured visuals are claimed
by this planning document.

## Planned deliverables and implementation order

1. Hardware/network inventory and compatible model selection; no private
   inventory details committed.
2. Single-GPU LoRA reference and frozen experiment manifest.
3. Versioned coordinator protocol, aggregation reference and recovery checks.
4. Pinned container build, Windows launch/stop scripts and scoped authentication.
5. One-friend onboarding pilot, resource checks and interruption test.
6. Full reproducible comparisons, audited summaries and embedded charts.

Keep future source, non-sensitive configs, documentation and reviewed results in
this exercise. Keep weights, datasets, credentials, private endpoints, environments
and raw logs in ignored local storage, following [SECURITY.md](../../SECURITY.md).

## References

- [Docker Desktop GPU support on Windows](https://docs.docker.com/desktop/features/gpu/)
- [Docker Desktop WSL2 backend](https://docs.docker.com/desktop/features/wsl/)
- [NVIDIA Container Toolkit installation (Linux hosts)](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [LoRA concepts](https://huggingface.co/docs/peft/main/conceptual_guides/lora)
- [PyTorch gradient accumulation and DDP synchronization](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [DiLoCo: low-communication training](https://arxiv.org/abs/2311.08105)
- [Hivemind: decentralized PyTorch training building blocks](https://github.com/learning-at-home/hivemind)
- [Optional Tailscale coordinator sharing](https://tailscale.com/kb/1084/sharing)
