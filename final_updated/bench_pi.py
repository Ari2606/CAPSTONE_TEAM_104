"""
Measure per-beat latency, memory and accuracy of the exported ONNX models on the machine it runs on
(intended: Raspberry Pi).  Needs:  pip install onnxruntime numpy psutil

  python bench_pi.py --model_dir deploy/seed0 --threads 1 --warmup 50 --repeats 3

Reads  model_fp32.onnx, model_int8.onnx, eval_beats.npz  (from export_onnx.py).  Writes  bench_<host>.json
and prints a markdown table.  Latency = wall-clock of one session.run() on one beat (batch 1), after warm-up.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

import numpy as np


def cpu_info() -> dict:
    info = {"machine": platform.machine(), "system": platform.platform(), "python": platform.python_version()}
    try:
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "hardware", "model\t")):
                info.setdefault("cpu", line.split(":", 1)[1].strip())
        info["cores"] = sum(1 for l in open("/proc/cpuinfo") if l.startswith("processor"))
    except Exception:
        pass
    try:
        info["rpi_model"] = open("/proc/device-tree/model").read().strip("\x00 \n")
    except Exception:
        pass
    return info


def cpu_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except Exception:
        return None


def cpu_freq_mhz():
    try:
        return int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read()) / 1000.0
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3, help="passes over the evaluation beats")
    a = ap.parse_args()

    import onnxruntime as ort
    import psutil

    d = Path(a.model_dir)
    data = np.load(d / "eval_beats.npz")
    x, y = data["x"], data["y"]
    aux = data["aux"] if "aux" in data.files else None
    proc = psutil.Process()
    result = {"host": socket.gethostname(), "info": cpu_info(), "threads": a.threads,
              "n_beats": int(len(x)), "temp_start_C": cpu_temp(), "freq_MHz_start": cpu_freq_mhz(), "models": {}}

    for name in ("fp32", "int8"):
        path = d / f"model_{name}.onnx"
        if not path.exists():
            continue
        rss0 = proc.memory_info().rss
        so = ort.SessionOptions()
        so.intra_op_num_threads = a.threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        t_load = time.perf_counter()
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        t_load = time.perf_counter() - t_load

        def feed(i):
            f = {"beat": x[i:i + 1]}
            if aux is not None:
                f["aux"] = aux[i:i + 1]
            return f

        for i in range(min(a.warmup, len(x))):
            sess.run(None, feed(i))
        lat, pred = [], np.zeros(len(x), dtype=int)
        for r in range(a.repeats):
            for i in range(len(x)):
                f = feed(i)
                t0 = time.perf_counter()
                out = sess.run(None, f)[0]
                lat.append((time.perf_counter() - t0) * 1e3)
                if r == 0:
                    pred[i] = int(out[0].argmax())
        lat = np.array(lat)
        rec = [float((pred[y == c] == c).mean()) for c in (0, 1, 2) if (y == c).any()]
        result["models"][name] = {
            "file_KB": path.stat().st_size / 1024.0,
            "load_s": t_load,
            "rss_delta_MB": (proc.memory_info().rss - rss0) / 1e6,
            "latency_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                           "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99)),
                           "max": float(lat.max())},
            "balanced_recall": float(np.mean(rec)),
            "recall_NSV": rec,
        }
    result["temp_end_C"] = cpu_temp()
    result["freq_MHz_end"] = cpu_freq_mhz()
    out = d / f"bench_{socket.gethostname()}_t{a.threads}.json"
    out.write_text(json.dumps(result, indent=2))

    print(f"\nHost: {result['info'].get('rpi_model') or result['info'].get('cpu')}  threads={a.threads}  "
          f"temp {result['temp_start_C']}->{result['temp_end_C']} C  freq {result['freq_MHz_start']} MHz")
    print("| model | file KB | RSS +MB | p50 ms | p95 ms | p99 ms | max ms | bal. recall |")
    print("|---|---|---|---|---|---|---|---|")
    for n, m in result["models"].items():
        L = m["latency_ms"]
        print(f"| {n} | {m['file_KB']:.0f} | {m['rss_delta_MB']:.1f} | {L['p50']:.3f} | {L['p95']:.3f} | "
              f"{L['p99']:.3f} | {L['max']:.3f} | {100*m['balanced_recall']:.1f}% |")
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
