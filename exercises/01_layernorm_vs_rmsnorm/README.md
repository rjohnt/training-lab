# LayerNorm vs RMSNorm: training runtime and memory overhead

Status: **planned; not executed**. Added as a future exercise during the data-movement
study. “Overlaying” means displaying both methods on the same chart with shared axes,
not running the two operations concurrently.

Question: how do the two normalization operations differ in runtime, live GPU
allocation, temporary allocation overhead, saved activations, backward-pass costs, and memory traffic?

## Fair comparison

Use the same input shape, dtype, epsilon, device, input values and learned scale.
Disable LayerNorm bias for the primary comparison so both operations have one
scale parameter. Normalize the last dimension; measure both a forward-only inference control and a training pass consisting of
forward, a fixed scalar loss, and backward. Use contiguous inputs and no residual
addition initially. If a residual is added later, give
both methods the same residual-rounding contract.

LayerNorm subtracts the row mean and divides by the square root of the row
variance plus epsilon. RMSNorm divides by the square root of the mean square
plus epsilon, without centering. Their outputs are **not expected to match**.
Validate each against its own FP64 oracle. This exercise measures implementation
costs; it does not establish model-quality equivalence or drop-in interchangeability.

Compare three implementation families separately: native PyTorch normalization,
eager formulas, and compiled formulas. A fused custom-kernel comparison can follow.
Do not attribute a difference between an unfused eager formula and a native fused
operator solely to the choice of normalization mathematics.

## Training scope

Measure forward latency, backward latency and complete forward/loss/backward time
separately. Use the same scalar-loss definition and gradient-buffer policy for both
normalizations. Verify input and scale gradients against each operation's own
high-precision reference. Report gradient storage separately from temporary
activations; optimizer state and optimizer updates are excluded from the first study.

Use saved-tensor hooks only in separate instrumented captures to record saved
activation metadata. Deduplicate shared storage when calculating footprint, avoid
retaining additional tensor references, and do not use instrumented timings as
performance measurements. Report actual peak live allocation as a separate metric.

## Planned measurements

| Measurement | Method / interpretation |
| --- | --- |
| GPU execution and Python wall time | Warm steady-state distributions, compilation excluded |
| Input, scale and output footprint | Tensor byte counts with the output retained |
| Peak allocated memory above baseline | CUDA allocator peak after resetting statistics |
| Temporary allocation estimate | Peak increment minus retained output bytes; allocator-visible estimate |
| Reserved memory | Report separately: the allocator pool is not the same as live tensors |
| Saved activations and gradients | Separate training captures; deduplicated storage and explicit gradient-buffer policy |
| Actual GPU memory traffic | Separate Nsight Compute capture; bytes moved are not allocation footprint |
| Allocation lifetime | Separate instrumented memory trace; never use its timings as benchmark latency |

Measure cold setup/compilation memory separately from warmed execution. Run memory
cases in fresh subprocesses to avoid comparing allocator pools left by different
operators. After warmup and synchronization, establish the live-allocation baseline,
reset peak statistics, run the operation, retain its output, synchronize, and
collect peak/current allocated and reserved bytes. State that external CUDA-library
allocations may not be visible to PyTorch's allocator statistics.

Start with rows 1, 32, 512 and 4096; widths 1024, 4096 and 8192; BF16 and FP32.
Specify epsilon explicitly instead of relying on different defaults. Freeze seeds,
versions, shapes and sampling rules before collecting results. Use long runs and
repeated independent passes, numerical checks and independently audited summaries.

## Planned overlays

- Latency versus row width, with both normalizations on the same axes.
- Peak allocated bytes and temporary overhead versus tensor size, with explicit units.
- Saved activation and gradient memory, with forward-only and training curves clearly labeled.
- Allocation timelines over aligned execution windows, labeled as instrumented captures.
- Nsight kernel timelines where helpful; these are separate from the shared-axis chart overlays.

Use identical axes and distinguish native/eager/compiled implementations. Include
SVG and PNG versions and embed the measured charts here. Keep raw profiler traces,
allocator snapshots and machine-specific logs in ignored local storage.

Dependencies: the shared CUDA/PyTorch environment from
[the inference-lab kernel-fusion exercise](https://github.com/rjohnt/inference-lab/tree/main/exercises/04_kernel_fusion) and the profiler tooling from
[the inference-lab data-movement exercise](https://github.com/rjohnt/inference-lab/tree/main/exercises/01_data_movement).

Reference: [PyTorch CUDA memory statistics](https://docs.pytorch.org/docs/stable/cuda.html) and [normalization implementations](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/normalization.py).
