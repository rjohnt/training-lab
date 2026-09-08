# Training Lab

Reproducible experiments in training performance, activation memory, gradients,
and GPU execution. Each exercise keeps its plan, source, configurations, reviewed
results and embedded charts in one directory. Planned exercises are not measured results.

| Exercise | Status |
| --- | --- |
| [LayerNorm vs RMSNorm](exercises/01_layernorm_vs_rmsnorm/) | Planned: training latency, peak memory, saved activations and shared-axis comparisons; forward-only inference controls |

[Inference Lab](https://github.com/rjohnt/inference-lab) holds the inference,
kernel-fusion and host/device data-movement experiments. This lab focuses on
training-specific costs, while retaining inference controls where useful.

Never commit credentials, private machine configuration, raw profiler traces,
model weights or unreviewed logs. See [SECURITY.md](SECURITY.md).
