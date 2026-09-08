# Training Lab

Reproducible experiments in training performance, activation memory, gradients,
and GPU execution. Each exercise keeps its plan, source, configurations, reviewed
results and embedded charts in one directory. Planned exercises are not measured results.

## Scope of the two labs

| Repository | Primary scope |
| --- | --- |
| [inference-lab](https://github.com/rjohnt/inference-lab) | Forward-only execution, inference kernel optimization, host/device data movement, model serving, decoding, latency and throughput. |
| [training-lab](https://github.com/rjohnt/training-lab) | Forward-plus-backward execution, gradients, saved activations, training memory, recomputation/checkpointing, optimizer costs and training throughput. |

Choose the home by the main question being measured, rather than the operator's
name. GPU tooling and kernels can support both labs. Keep a study in one primary
home and cross-link it; include clearly labeled controls from the other scope
when useful. The LayerNorm vs RMSNorm memory study belongs in training-lab because
it examines saved activations and backward costs, with forward-only inference
controls. The transfer/pipeline study belongs in inference-lab.

| Exercise | Status |
| --- | --- |
| [LayerNorm vs RMSNorm](exercises/01_layernorm_vs_rmsnorm/) | Harness prepared: training latency, peak memory, saved activations and shared-axis comparisons; GPU measurements pending |
| [Volunteer GPU training on Windows](exercises/02_volunteer_distributed_training/) | Planned: opt-in workers for 2–3 friends, outbound HTTPS coordination, LoRA local updates and time-to-quality comparisons |

[Inference Lab](https://github.com/rjohnt/inference-lab) holds the inference,
kernel-fusion and host/device data-movement experiments. This lab focuses on
training-specific costs, while retaining inference controls where useful.

Never commit credentials, private machine configuration, raw profiler traces,
model weights or unreviewed logs. See [SECURITY.md](SECURITY.md).
