"""Independently audit numeric samples and publish compact overlays."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import itertools
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "normalization-training-v1"
import matplotlib.pyplot as plt

MODES = ("inference", "training_forward", "backward_retained", "full_step")


def audit(root):
    assert (root / "COMPLETE").exists(), "Run incomplete"
    manifest = json.loads((root / "manifest.json").read_text())
    cfg = manifest["config"]
    assert hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest() == manifest["config_sha256"]
    expected = set(itertools.product(range(cfg["repeats"]), cfg["rows"], cfg["widths"],
                                     cfg["dtypes"], cfg["families"], cfg["norms"]))
    assert len(manifest["cases"]) == len(expected)
    assert set(map(tuple, manifest["cases"])) == expected
    results, sample_count, call_count, seconds = [], 0, 0, 0.
    for i, case in enumerate(manifest["cases"]):
        directory = root / f"case-{i:03}"
        assert (directory / "COMPLETE").exists()
        row = json.loads((directory / "summary.json").read_text())
        assert row["case"] == case and row["post_timing_finite_gradients"]
        assert row["post_timing_gradient_oracle_passed"]
        assert [v["seed"] for v in row["correctness"]] == cfg["correctness_seeds"]
        for v in row["correctness"]:
            assert set(v["max_abs_errors"]) == {"output", "input_gradient", "weight_gradient"}
            assert all(np.isfinite(x) and x >= 0 for x in v["max_abs_errors"].values())
        raw = (directory / "samples.jsonl").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row["samples_sha256"]
        groups = defaultdict(dict)
        for line in raw.splitlines():
            s = json.loads(line)
            assert s["mode"] in MODES and s["sample"] not in groups[s["mode"]]
            assert isinstance(s["calls"], int) and s["calls"] > 0
            assert all(np.isfinite(s[k]) and s[k] > 0 for k in ("wall_seconds", "gpu_seconds"))
            groups[s["mode"]][s["sample"]] = s
        assert set(groups) == set(MODES)
        for mode, samples in groups.items():
            v = row["timings"][mode]
            assert set(samples) == set(range(v["samples"]))
            assert len(samples) >= cfg["min_samples"]
            assert all(s["calls"] == v["calls_per_sample"] for s in samples.values())
            wall = sum(s["wall_seconds"] for s in samples.values())
            gpu = sum(s["gpu_seconds"] for s in samples.values())
            assert wall >= cfg["min_seconds"]
            assert np.isclose(wall, v["measured_wall_seconds"], rtol=1e-10)
            assert np.isclose(gpu, v["measured_gpu_seconds"], rtol=1e-10)
            for metric, field in (("wall", "wall_seconds"), ("gpu_event", "gpu_seconds")):
                us = [s[field] * 1e6 / s["calls"] for s in samples.values()]
                for label, value in (("p50_us", np.median(us)), ("p95_us", np.percentile(us, 95))):
                    assert np.isclose(v[metric][label], value, rtol=1e-10)
            sample_count += len(samples)
            call_count += len(samples) * v["calls_per_sample"]
            seconds += wall
        for m in row["memory"].values():
            for s in m["stages"]:
                assert s["increment_bytes"] == s["allocated_bytes"] - m["baseline_bytes"]
                assert s["peak_increment_bytes"] >= s["increment_bytes"]
                assert s["reserved_bytes"] >= s["allocated_bytes"]
        saved = row["saved_tensors"]
        assert saved["storages"], "Saved-tensor observation missing"
        itemsize = {"bfloat16": 2, "float32": 4}[case[3]]
        assert row["input_bytes"] == case[1] * case[2] * itemsize
        assert row["weight_bytes"] == case[2] * itemsize
        assert row["memory"]["training"]["gradient_bytes"] == row["input_bytes"] + row["weight_bytes"]
        assert saved["unique_storage_bytes"] == sum(s["storage_bytes"] for s in saved["storages"])
        assert saved["additional_storage_bytes"] == sum(s["storage_bytes"] for s in saved["storages"] if not s["aliases_input_or_weight"])
        results.append(row)
    return manifest, results, {"status": "PASS", "case_processes": len(results),
                               "timed_blocks": sample_count, "operation_calls": call_count,
                               "measured_wall_seconds": seconds,
                               "checks": ["case coverage", "completion", "correctness metadata", "sample hashes",
                                          "unique samples", "fixed batch calls", "duration and sample floors",
                                          "recomputed medians and percentiles", "allocation accounting",
                                          "saved storage accounting"]}


def export(manifest, results, checked, out):
    out.mkdir(parents=True, exist_ok=False)
    for name, obj in (("manifest", manifest), ("summary", results), ("audit", checked)):
        (out / f"{name}.json").write_text(json.dumps(obj, indent=2) + "\n")
    (out / "COMPLETE").write_text("All raw samples independently audited\n")
    flat = []
    for r in results:
        repeat, rows, width, dtype, family, norm = r["case"]
        for mode, v in r["timings"].items():
            for metric in ("wall", "gpu_event"):
                flat.append(dict(repeat=repeat, rows=rows, width=width, dtype=dtype, family=family,
                                 norm=norm, mode=mode, metric=metric, **v[metric], samples=v["samples"],
                                 calls_per_sample=v["calls_per_sample"], measured_wall_seconds=v["measured_wall_seconds"]))
    with (out / "timings.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(flat)
    memory = []
    for r in results:
        repeat, rows, width, dtype, family, norm = r["case"]
        inf, train = r["memory"]["inference"], r["memory"]["training"]
        memory.append(dict(repeat=repeat, rows=rows, width=width, dtype=dtype, family=family, norm=norm,
                           input_bytes=r["input_bytes"], weight_bytes=r["weight_bytes"],
                           inference_baseline_bytes=inf["baseline_bytes"],
                           inference_peak_increment_bytes=inf["stages"][-1]["peak_increment_bytes"],
                           inference_temporary_estimate_bytes=inf["temporary_estimate_bytes"],
                           training_baseline_bytes=train["baseline_bytes"],
                           training_forward_increment_bytes=train["stages"][1]["increment_bytes"],
                           training_peak_increment_bytes=train["stages"][-1]["peak_increment_bytes"],
                           training_absolute_peak_bytes=train["baseline_bytes"] + train["stages"][-1]["peak_increment_bytes"],
                           training_max_reserved_bytes=train["stages"][-1]["max_reserved_bytes"],
                           gradient_bytes=train["gradient_bytes"],
                           saved_unique_storage_bytes=r["saved_tensors"]["unique_storage_bytes"],
                           saved_additional_storage_bytes=r["saved_tensors"]["additional_storage_bytes"]))
    with (out / "memory.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(memory[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(memory)


def plots(manifest, results, out):
    cfg = manifest["config"]
    shapes = list(itertools.product(cfg["rows"], cfg["widths"]))
    definitions = {
        "full-step": ("Forward + loss + backward", "µs per step", lambda r: r["timings"]["full_step"]["wall"]["p50_us"]),
        "inference": ("Forward-only inference", "µs per call", lambda r: r["timings"]["inference"]["wall"]["p50_us"]),
        "backward": ("Backward on a retained graph", "µs per backward", lambda r: r["timings"]["backward_retained"]["wall"]["p50_us"]),
        "training-peak": ("Training peak allocation above warmed baseline", "MiB", lambda r: r["memory"]["training"]["stages"][-1]["peak_increment_bytes"] / 2**20),
        "inference-temporary": ("Inference temporary allocation estimate", "MiB", lambda r: r["memory"]["inference"]["temporary_estimate_bytes"] / 2**20),
        "saved-activations": ("Saved storage excluding input/weight aliases", "MiB", lambda r: r["saved_tensors"]["additional_storage_bytes"] / 2**20),
    }
    for filename, (title, unit, getter) in definitions.items():
        fig, axes = plt.subplots(len(cfg["dtypes"]), len(cfg["families"]), figsize=(17, 9), squeeze=False)
        for i, dtype in enumerate(cfg["dtypes"]):
            for j, family in enumerate(cfg["families"]):
                ax = axes[i, j]
                for norm, color, marker in (("layernorm", "#2563eb", "o"), ("rmsnorm", "#e27519", "s")):
                    values, lows, highs = [], [], []
                    for rows, width in shapes:
                        v = [getter(r) for r in results if r["case"][1:] == [rows, width, dtype, family, norm]]
                        assert len(v) == cfg["repeats"]
                        values.append(np.median(v)); lows.append(min(v)); highs.append(max(v))
                    ax.plot(range(len(shapes)), values, color=color, marker=marker, label=norm)
                    ax.fill_between(range(len(shapes)), lows, highs, color=color, alpha=.15)
                ax.set_title(f"{family} · {dtype}")
                ax.set_xticks(range(len(shapes)), [f"{r}×{w}" for r, w in shapes], rotation=70, fontsize=7)
                if filename in ("full-step", "inference", "backward", "training-peak"):
                    ax.set_yscale("log")
                else:
                    ax.set_ylim(bottom=0)
                ax.set_ylabel(unit); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle(title + "\nMedian of repeat medians; shading spans repeats; synchronized wall timing", fontsize=14)
        if filename not in ("full-step", "inference", "backward"):
            fig.suptitle(title + "\nSeparate instrumented passes; median with repeat range", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .94))
        fig.savefig(out / f"{filename}.png", dpi=150)
        fig.savefig(out / f"{filename}.svg", metadata={"Date": None})
        plt.close(fig)
    # Aligned allocation stages are not a wall-clock allocation-lifetime trace.
    rows, width = max(cfg["rows"]), max(cfg["widths"])
    fig, axes = plt.subplots(len(cfg["dtypes"]), len(cfg["families"]), figsize=(15, 8), squeeze=False)
    for i, dtype in enumerate(cfg["dtypes"]):
        for j, family in enumerate(cfg["families"]):
            ax = axes[i,j]
            for norm, color in (("layernorm", "#2563eb"), ("rmsnorm", "#e27519")):
                selected = [r for r in results if r["case"][1:] == [rows, width, dtype, family, norm]]
                values = np.median([[s["increment_bytes"] / 2**20 for s in r["memory"]["training"]["stages"]] for r in selected], axis=0)
                ax.plot(range(4), values, "o-", color=color, label=norm)
            ax.set_title(f"{family} · {dtype}"); ax.set_ylabel("Live allocation increment (MiB)")
            ax.set_xticks(range(4), ["inputs", "forward", "loss", "backward"], rotation=15)
            ax.grid(alpha=.2); ax.legend()
    fig.suptitle(f"Allocation at synchronized training stages · {rows}×{width}\nOutput retained; optimizer excluded; horizontal spacing does not represent time")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(out / "allocation-stages.png", dpi=150)
    fig.savefig(out / "allocation-stages.svg", metadata={"Date": None})
    plt.close(fig)
    for p in out.glob("*.svg"):
        p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")


def report(manifest, results, checked, out):
    cfg = manifest["config"]
    elapsed = (datetime.fromisoformat(manifest["finished_utc"]) - datetime.fromisoformat(manifest["started_utc"])).total_seconds()
    text = ["# LayerNorm versus RMSNorm: measured training baseline", "",
            f"{checked['case_processes']} fresh case processes; {cfg['repeats']} repeats per configuration; "
            f"{checked['timed_blocks']:,} timed blocks; {checked['operation_calls']:,} calls. "
            f"Measured wall batches: {checked['measured_wall_seconds']/60:.2f} minutes; total elapsed: {elapsed/60:.2f} minutes.", "",
            f"Every mode has at least {cfg['min_samples']:,} samples AND {cfg['min_seconds']} wall seconds per process. "
            "Each sample is a fixed calibrated batch; medians and p95 describe batch averages, not individual-request tails.", "",
            "Bias-free, learned scale, fixed epsilon, no residual. Native kernels and FP32 eager/compiled formulas are separate families. "
            "Forward values and input/weight gradients pass each norm's own FP64 oracle. Saved-tensor hooks and allocator snapshots run after timing.", "",
            "| Family | Dtype | Full-step RMSNorm speedup over LayerNorm, min–max across shapes | RMSNorm faster shapes |", "|---|---|---:|---:|"]
    for family in cfg["families"]:
        for dtype in cfg["dtypes"]:
            ratios = []
            for rows, width in itertools.product(cfg["rows"], cfg["widths"]):
                med = {n: np.median([r["timings"]["full_step"]["wall"]["p50_us"] for r in results
                                   if r["case"][1:] == [rows, width, dtype, family, n]]) for n in cfg["norms"]}
                ratios.append(med["layernorm"] / med["rmsnorm"])
            text.append(f"| {family} | {dtype} | {min(ratios):.2f}–{max(ratios):.2f}× | {sum(v>1 for v in ratios)}/{len(ratios)} |")
    for name in ("full-step", "inference", "backward", "training-peak", "inference-temporary", "saved-activations", "allocation-stages"):
        text += ["", f"![{name}]({name}.png)", "", f"[SVG]({name}.svg)"]
    text += ["", "[Timing CSV](timings.csv) · [Memory CSV](memory.csv) · [Audited summaries](summary.json)", ""]
    text += ["", "## Interpretation limits", "",
             "CUDA-event intervals include launch starvation during ordinary Python calls; they are not graph-replay kernel latency. "
             "The primary wall interval includes event recording and end synchronization, amortized over each calibrated batch. "
             "Backward-only repeats reuse a retained graph; full-step builds a fresh graph and clears gradients to None on every call. "
             "Full-step includes the fixed weighted-sum loss and excludes optimizer updates. "
             f"Compiler buffer donation is set to {cfg['compiled_donated_buffer']} to support graph reuse; this also constrains "
             "compiled full-step and memory results relative to default compiler optimization.", "",
             "Allocator-visible bytes are not DRAM traffic or process-wide GPU usage. The warmed baseline includes inputs, weight, upstream gradients "
             "and any live runtime buffers; absolute baseline, current and peak values can be recovered from the summaries. Saved-storage metadata deduplicates aliases and excludes "
             "input/weight storage from the additional-storage figure; it excludes the loss's saved tensors. Stage snapshots include the loss, "
             "output and gradient allocations. Stage positions are not elapsed-time measurements. External CUDA allocations may be invisible.", "",
             f"The {cfg['repeats']} process repeat(s) per case characterize this session, not cross-day stability. Compilation and checks are excluded "
             "from warmed timings; first-call records may use persistent compiler caches. Reused buffers, unlocked GPU clocks, display activity "
             "and other host activity limit generalization. Numerical comparisons do not establish equal model quality or drop-in interchangeability.", "",
             "Actual DRAM-traffic counters and detailed allocation-lifetime traces remain separate future profiling work. "
             "This result covers timing, allocator peaks, saved storage and synchronized allocation stages.", ""]
    (out / "report.md").write_text("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    m, r, a = audit(args.run)
    export(m, r, a, args.out)
    plots(m, r, args.out)
    report(m, r, a, args.out)
    print(json.dumps(a))
