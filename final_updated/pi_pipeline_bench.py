"""pi_pipeline_bench.py -- time the WHOLE processing chain per 30 s ECG segment on the machine it runs on (intended: Raspberry Pi).
Chain per segment (everything causal / forward-only, as an embedded monitor would run it):
  1 preprocess   Butterworth 0.5-40 Hz forward filter + trailing-60 s z-score
  2 QRS detect   causal Pan-Tompkins (pan_tompkins.py; DS1-tuned thr_k=0.5, shift=4)
  3 build        256-sample window per beat + 2 template channels (median of previous 16 beats) + 7 RR features
  4 infer        ONNX Runtime, one beat at a time (batch 1)
Needs: numpy scipy onnxruntime  (pip install numpy scipy onnxruntime psutil).  Files: segments.npz, pan_tompkins.py, <model_dir>/model_*.onnx
  OMP_NUM_THREADS=1 python pi_pipeline_bench.py --model_dir tmplaug_seed0 --segments segments.npz --repeats 5
Reports per-stage and total wall-clock per 30 s segment (median / p95 over segments x repeats). Timing only: no accuracy is computed here."""
import argparse, json, platform, socket, time
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfilt
from pan_tompkins import pan_tompkins

FS, PRE, POST = 360, 96, 160
SOS = butter(4, [0.5, 40.0], btype="band", fs=FS, output="sos")


def preprocess(sig):
    y = sosfilt(SOS, sig)
    c1, c2 = np.cumsum(np.concatenate([[0.0], y])), np.cumsum(np.concatenate([[0.0], y * y]))
    n = np.arange(1, len(y) + 1); w = np.minimum(n, 60 * FS); lo = n - w
    mu = (c1[n] - c1[lo]) / w; var = np.maximum((c2[n] - c2[lo]) / w - mu * mu, 0.0)
    return (y - mu) / (np.sqrt(var) + 1e-8)


def add_template_channels(x, k):
    n, c, L = x.shape
    tmpl = np.empty((n, L), dtype=np.float32)
    for i in range(n):
        tmpl[i] = np.median(x[max(0, i - k):i, 0, :], axis=0) if i > 0 else x[0, 0, :]
    return np.concatenate([x, tmpl[:, None, :], (x[:, 0, :] - tmpl)[:, None, :]], axis=1).astype(np.float32)


def rr_features(samples, fs, local_window=9):
    n = len(samples); pre_rr = np.full(n, np.nan); post_rr = np.full(n, np.nan); local_rr = np.full(n, np.nan)
    s = samples.astype(np.float64)
    for i in range(n):
        if i > 0: pre_rr[i] = (s[i] - s[i - 1]) / fs
        if i < n - 1: post_rr[i] = (s[i + 1] - s[i]) / fs
        lo = max(0, i - local_window)
        if i > lo: local_rr[i] = np.mean(np.diff(s[lo:i + 1])) / fs
    valid = np.concatenate([pre_rr[~np.isnan(pre_rr)], post_rr[~np.isnan(post_rr)]])
    fb = float(np.median(valid)) if valid.size else 0.8
    pre_rr = np.where(np.isnan(pre_rr), np.where(~np.isnan(post_rr), post_rr, fb), pre_rr)
    post_rr = np.where(np.isnan(post_rr), np.where(~np.isnan(pre_rr), pre_rr, fb), post_rr)
    local_rr = np.where(np.isnan(local_rr), pre_rr, local_rr)
    z = lambda a: np.clip((a - 0.8) / 0.3, -3.0, 3.0).astype(np.float32)
    cols = [z(pre_rr), z(post_rr), z(local_rr)]
    prev_avg = pre_rr.copy()
    for i in range(2, n):
        lo = max(0, i - 1 - local_window); prev_avg[i] = np.mean(np.diff(s[lo:i])) / fs
    eps = 1e-3
    cols += [np.clip(np.log((pre_rr + eps) / (prev_avg + eps)) / 0.25, -3, 3).astype(np.float32),
             np.clip(np.log((post_rr + eps) / (pre_rr + eps)) / 0.25, -3, 3).astype(np.float32)]
    d = np.diff(s) / fs; cs = np.concatenate([[0.0], np.cumsum(d)])
    for W in (80, 400):
        r = np.zeros(n)
        for i in range(2, n):
            lo = max(0, i - 1 - W); r[i] = np.log((pre_rr[i] + eps) / ((cs[i - 1] - cs[lo]) / (i - 1 - lo) + eps))
        cols.append(np.clip(r / 0.25, -3, 3).astype(np.float32))
    return np.stack(cols, axis=1)


def build(proc, peaks):
    T = len(proc); keep = np.asarray([p for p in peaks if p - PRE >= 0 and p + POST <= T], dtype=np.int64)
    if len(keep) == 0:
        return np.zeros((0, 3, PRE + POST), np.float32), np.zeros((0, 7), np.float32), keep
    x = np.stack([proc[p - PRE:p + POST] for p in keep])[:, None, :].astype(np.float32)
    return add_template_channels(x, 16), rr_features(keep, FS).astype(np.float32), keep


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model_dir", required=True); ap.add_argument("--segments", default="segments.npz")
    ap.add_argument("--repeats", type=int, default=5); ap.add_argument("--warmup", type=int, default=3); a = ap.parse_args()
    import onnxruntime as ort
    seg = np.load(a.segments)["sig"]; res = {"host": socket.gethostname(), "machine": platform.machine(), "n_segments": int(len(seg)), "models": {}}
    try: res["rpi_model"] = open("/proc/device-tree/model").read().strip("\x00 \n")
    except Exception: pass
    for name in ("fp32", "int8"):
        path = Path(a.model_dir) / f"model_{name}.onnx"
        if not path.exists(): continue
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        def one(sig):
            t0 = time.perf_counter(); proc = preprocess(sig); t1 = time.perf_counter()
            peaks = np.unique(pan_tompkins(proc, FS, thr_k=0.5, shift=4)); t2 = time.perf_counter()
            x, rr, keep = build(proc, peaks); t3 = time.perf_counter()
            preds = [int(sess.run(None, {"beat": x[i:i + 1], "aux": rr[i:i + 1]})[0][0].argmax()) for i in range(len(x))]; t4 = time.perf_counter()
            return [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0], len(keep)
        for i in range(a.warmup): one(seg[i % len(seg)])
        T, NB = [], []
        for _ in range(a.repeats):
            for s in seg:
                t, nb = one(s); T.append(t); NB.append(nb)
        T = np.array(T) * 1e3; names = ["preprocess", "qrs_detect", "build_inputs", "inference", "total"]
        res["models"][name] = {"beats_per_segment_mean": float(np.mean(NB)),
            **{n: {"p50_ms": float(np.percentile(T[:, i], 50)), "p95_ms": float(np.percentile(T[:, i], 95)), "max_ms": float(T[:, i].max())} for i, n in enumerate(names)},
            "total_as_fraction_of_30s": float(np.percentile(T[:, 4], 50) / 30000.0)}
    out = Path(a.model_dir) / f"pipeline_{socket.gethostname()}.json"; out.write_text(json.dumps(res, indent=2))
    print(f"\nHost: {res.get('rpi_model', res['machine'])}  segments={res['n_segments']}  (30 s each)")
    print("| model | beats/seg | preprocess ms | QRS ms | build ms | infer ms | TOTAL p50 ms | TOTAL p95 ms | total / 30 s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for n, m in res["models"].items():
        print(f"| {n} | {m['beats_per_segment_mean']:.1f} | {m['preprocess']['p50_ms']:.1f} | {m['qrs_detect']['p50_ms']:.1f} | {m['build_inputs']['p50_ms']:.1f} | "
              f"{m['inference']['p50_ms']:.1f} | {m['total']['p50_ms']:.1f} | {m['total']['p95_ms']:.1f} | {100*m['total_as_fraction_of_30s']:.2f}% |")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
