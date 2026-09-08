"""Fresh-process normalization timing and allocator/saved-tensor measurements."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = ("inference", "training_forward", "backward_retained", "full_step")


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(cfg, case, out):
    import numpy as np
    import torch
    import triton
    import torch._functorch.config
    from implementations import build, formula
    repeat, rows, width, dtype_name, family, norm = case
    dtype = getattr(torch, dtype_name)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._functorch.config.donated_buffer = cfg["compiled_donated_buffer"]
    torch.manual_seed(cfg["seed"])
    fn = build(norm, family, cfg["epsilon"])
    def data(seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(rows, width, device="cuda", dtype=dtype, generator=g).requires_grad_()
        w = torch.randn(width, device="cuda", dtype=dtype, generator=g).requires_grad_()
        dy = torch.randn(rows, width, device="cuda", dtype=dtype, generator=g) / math.sqrt(rows * width)
        return x, w, dy
    def loss(y, dy):
        return (y.float() * dy.float()).sum()
    x, w, dy = data(cfg["seed"])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    cold_base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    with torch.no_grad():
        y = fn(x, w)
    loss(fn(x, w), dy).backward()
    torch.cuda.synchronize()
    cold = {"first_inference_and_training_seconds": time.perf_counter() - start,
            "peak_increment_bytes": torch.cuda.max_memory_allocated() - cold_base,
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
            "cache_policy": "persistent compiler cache; not a cold-cache compilation benchmark"}
    del y
    x.grad = w.grad = None
    checks = []
    for seed in cfg["correctness_seeds"]:
        a, b, grad = data(seed)
        a0, b0 = a.detach().clone(), b.detach().clone()
        ah = a.detach().double().requires_grad_()
        bh = b.detach().double().requires_grad_()
        expected = formula(ah, bh, cfg["epsilon"], norm, high_precision=True)
        expected_grads = torch.autograd.grad((expected * grad.double()).sum(), (ah, bh))
        actual = fn(a, b)
        actual_grads = torch.autograd.grad(loss(actual, grad), (a, b))
        errs = {}
        for label, got, want in zip(("output", "input_gradient", "weight_gradient"),
                                    (actual, *actual_grads), (expected, *expected_grads)):
            want = want.to(dtype)
            torch.testing.assert_close(got, want, **cfg["tolerances"][dtype_name])
            assert torch.isfinite(got).all().item()
            errs[label] = (got.float() - want.float()).abs().max().item()
        torch.testing.assert_close(a, a0, rtol=0, atol=0)
        torch.testing.assert_close(b, b0, rtol=0, atol=0)
        checks.append({"seed": seed, "max_abs_errors": errs})
        del a, b, grad, a0, b0, ah, bh, expected, expected_grads, actual, actual_grads, got, want
    gc.collect()
    torch.cuda.empty_cache()

    # Four independent modes: backward reuses a fixed graph; full_step does not.
    # Both timing modes include dispatch; CUDA events can include GPU idle gaps.
    def inference():
        with torch.no_grad():
            return fn(x, w)
    def training_forward():
        return fn(x, w)
    def full_step():
        x.grad = w.grad = None
        loss(fn(x, w), dy).backward()
    timings = {}
    mode_order = list(MODES)
    random.Random(cfg["seed"] + repeat).shuffle(mode_order)
    with (out / "samples.jsonl").open("w") as raw:
        for mode in mode_order:
            retained_loss = loss(fn(x, w), dy) if mode == "backward_retained" else None
            def backward():
                x.grad = w.grad = None
                retained_loss.backward(retain_graph=True)
            run = {"inference": inference, "training_forward": training_forward,
                   "backward_retained": backward, "full_step": full_step}[mode]
            for _ in range(cfg["warmup_calls"]):
                run()
            torch.cuda.synchronize()
            begin_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            def block(calls):
                torch.cuda.synchronize()
                t = time.perf_counter_ns()
                begin_event.record()
                for _ in range(calls):
                    run()
                end_event.record()
                end_event.synchronize()
                wall = (time.perf_counter_ns() - t) / 1e9
                gpu = begin_event.elapsed_time(end_event) / 1000
                return wall, gpu
            estimate = np.median([block(5)[0] / 5 for _ in range(3)])
            calls = max(1, math.ceil(cfg["target_sample_seconds"] / estimate))
            walls, gpus = [], []
            seconds = 0.
            while len(walls) < cfg["min_samples"] or seconds < cfg["min_seconds"]:
                if len(walls) >= cfg["max_samples"]:
                    raise RuntimeError("Sample limit reached before duration floor")
                wall, gpu = block(calls)
                raw.write(json.dumps({"mode": mode, "sample": len(walls), "calls": calls,
                                      "wall_seconds": wall, "gpu_seconds": gpu}) + "\n")
                walls.append(wall * 1e6 / calls)
                gpus.append(gpu * 1e6 / calls)
                seconds += wall
            def summary(v):
                return {"p50_us": float(np.median(v)), "p95_us": float(np.percentile(v, 95))}
            timings[mode] = {"wall": summary(walls), "gpu_event": summary(gpus),
                             "samples": len(walls), "calls_per_sample": calls,
                             "measured_wall_seconds": seconds,
                             "measured_gpu_seconds": sum(gpus) * calls / 1e6}
            if retained_loss is not None:
                x.grad = w.grad = None
                retained_loss.backward()  # release the retained autograd buffers
            del retained_loss, run
            x.grad = w.grad = None
    # Ensure the timed full-step path still produces the expected gradients.
    full_step()
    hx = x.detach().double().requires_grad_()
    hw = w.detach().double().requires_grad_()
    hy = formula(hx, hw, cfg["epsilon"], norm, high_precision=True)
    hg = torch.autograd.grad((hy * dy.double()).sum(), (hx, hw))
    for got, want in zip((x.grad, w.grad), hg):
        torch.testing.assert_close(got, want.to(dtype), **cfg["tolerances"][dtype_name])
        assert torch.isfinite(got).all().item()
    del hx, hw, hy, hg, got, want
    x.grad = w.grad = None
    gc.collect()
    torch.cuda.synchronize()

    def snapshot(stage, base):
        torch.cuda.synchronize()
        return {"stage": stage, "allocated_bytes": torch.cuda.memory_allocated(),
                "increment_bytes": torch.cuda.memory_allocated() - base,
                "peak_increment_bytes": torch.cuda.max_memory_allocated() - base,
                "reserved_bytes": torch.cuda.memory_reserved(),
                "max_reserved_bytes": torch.cuda.max_memory_reserved()}
    memory = {}
    # Warmed allocator; each norm/family/shape/repeat is in a fresh process.
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    stages = [snapshot("baseline", base)]
    y = inference()
    stages.append(snapshot("output_retained", base))
    output_bytes = y.numel() * y.element_size()
    memory["inference"] = {"baseline_bytes": base, "stages": stages,
                           "output_bytes": output_bytes,
                           "temporary_estimate_bytes": max(0, stages[-1]["peak_increment_bytes"] - output_bytes)}
    del y
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    stages = [snapshot("baseline", base)]
    y = fn(x, w)
    stages.append(snapshot("forward_output_retained", base))
    scalar = loss(y, dy)
    stages.append(snapshot("loss_retained", base))
    scalar.backward()
    stages.append(snapshot("backward_complete", base))
    memory["training"] = {"baseline_bytes": base, "stages": stages,
                          "gradient_bytes": sum(t.grad.numel() * t.grad.element_size() for t in (x, w)),
                          "output_bytes": output_bytes}
    del scalar, y
    x.grad = w.grad = None

    # Instrumented pass, outside timing. Store metadata only; never extra tensors.
    known = {t.untyped_storage().data_ptr() for t in (x, w)}
    seen, records = {}, []
    def pack(t):
        s = t.untyped_storage()
        key = s.data_ptr()
        if key not in seen:
            seen[key] = len(seen)
            records.append({"storage_id": seen[key], "storage_bytes": s.nbytes(),
                            "aliases_input_or_weight": key in known,
                            "shape": list(t.shape), "dtype": str(t.dtype)})
        return t.detach()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        y = fn(x, w)  # normalization only; exclude the loss's own saved tensors
    assert records, "Saved-tensor hooks observed no storage; instrumentation is incomplete"
    saved = {"unique_storage_bytes": sum(r["storage_bytes"] for r in records),
             "additional_storage_bytes": sum(r["storage_bytes"] for r in records if not r["aliases_input_or_weight"]),
             "storages": records}
    loss(y, dy).backward()
    info = torch.cuda.get_device_properties(0)
    result = {"case": case, "correctness": checks, "post_timing_finite_gradients": True,
              "post_timing_gradient_oracle_passed": True,
              "timings": timings, "cold": cold, "memory": memory, "saved_tensors": saved,
              "input_bytes": x.numel() * x.element_size(), "weight_bytes": w.numel() * w.element_size(),
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "triton": triton.__version__, "cuda": torch.version.cuda,
                              "gpu": info.name, "vram_bytes": info.total_memory},
              "samples_sha256": digest(out / "samples.jsonl")}
    save(out / "summary.json", result)
    (out / "COMPLETE").write_text("Correctness and all duration/sample floors passed\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/standard.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker-case", type=int)
    args = parser.parse_args()
    if args.worker_case is not None:
        manifest = json.loads((args.out / "manifest.json").read_text())
        directory = args.out / f"case-{args.worker_case:03}"
        directory.mkdir(exist_ok=False)
        worker(manifest["config"], manifest["cases"][args.worker_case], directory)
        return
    cfg = json.loads(args.config.read_text())
    assert cfg["min_samples"] >= 10 and cfg["min_seconds"] > 0
    assert cfg["max_samples"] >= cfg["min_samples"] and cfg["repeats"] >= 1
    smi = "/usr/lib/wsl/lib/nvidia-smi" if Path("/usr/lib/wsl/lib/nvidia-smi").exists() else "nvidia-smi"
    def gpu_state():
        fields = ["driver_version", "memory.used", "utilization.gpu", "temperature.gpu", "clocks.sm", "power.draw"]
        result = subprocess.run([smi, "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits", "-i", "0"],
                                text=True, capture_output=True, check=True)
        return dict(zip(fields, map(str.strip, result.stdout.strip().split(","))))
    cases = list(itertools.product(range(cfg["repeats"]), cfg["rows"], cfg["widths"],
                                   cfg["dtypes"], cfg["families"], cfg["norms"]))
    random.Random(cfg["seed"]).shuffle(cases)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "config": cfg,
                "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
                "source_sha256": {p.name: digest(p) for p in sorted((ROOT / "src").glob("*.py"))},
                "cases": cases, "gpu_before": gpu_state()}
    save(args.out / "manifest.json", manifest)
    for i, case in enumerate(cases):
        with (args.out / f"case-{i:03}.log").open("w") as log:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--out", str(args.out.resolve()),
                            "--worker-case", str(i)], stdout=log, stderr=subprocess.STDOUT, check=True)
        save(args.out / "progress.json", {"completed": i + 1, "total": len(cases), "case": case})
        print(json.dumps({"completed": i + 1, "total": len(cases), "case": case}), flush=True)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["gpu_after"] = gpu_state()
    save(args.out / "manifest.json", manifest)
    (args.out / "COMPLETE").write_text(f"{len(cases)} case processes complete\n")


if __name__ == "__main__":
    main()
