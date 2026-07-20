"""
Real power / energy measurement for DeepLOB, Axial-LOB and CFLOB on Jetson Orin Nano.

Uses the on-board INA3221 rails via tegrastats (no external meter needed).
For each model reports per-inference energy on three rails:
  - VDD_IN          : total module power (system-level)
  - VDD_CPU_GPU_CV  : CPU + GPU + CV engines (inference-attributable)
  - VDD_SOC         : rest of SoC

Methodology:
  1. Idle baseline power (device idle).
  2. Long inference loop; sample power continuously.
  3. energy_per_inf        = P_active_avg * T / N
     dynamic_energy_per_inf = (P_active_avg - P_idle_avg) * T / N

Results are written to power_results.log (and echoed to stdout).

Run:  sudo python3 measure_power_all.py
"""
import re
import time
import threading
import subprocess
import datetime
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# (display name, engine path) -- adjust paths/extensions if needed
MODELS = [
    ("DeepLOB",   "other_model/deeplob_fp16.engin"),
    ("Axial-LOB", "other_model/axiallob_fp16.engin"),
    ("CFLOB",     "cflob/cflob_fp16.engine"),
]

LOG_PATH = "power_results.log"
RUN_SECONDS = 30          # inference loop duration per model
IDLE_SECONDS = 10         # idle baseline duration per model
TEGRA_INTERVAL_MS = 50    # tegrastats sample interval
WARMUP = 50

RAILS = ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC")
RAIL_RE = re.compile(r"(VDD_IN|VDD_CPU_GPU_CV|VDD_SOC)\s+(\d+)mW")

log_lines = []
def log(msg=""):
    print(msg)
    log_lines.append(msg)

def flush_log():
    with open(LOG_PATH, "a") as f:
        f.write("\n".join(log_lines) + "\n")


class PowerSampler(threading.Thread):
    """Background thread parsing tegrastats; accumulates rail power (mW) samples."""
    def __init__(self, interval_ms=TEGRA_INTERVAL_MS):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.samples = {r: [] for r in RAILS}
        self._stop = threading.Event()
        self._proc = None

    def run(self):
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, text=True,
        )
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            for rail, mw in RAIL_RE.findall(line):
                self.samples[rail].append(int(mw))

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()

    def averages_w(self):
        return {r: (np.mean(v) / 1000.0 if v else float("nan"))
                for r, v in self.samples.items()}

    def reset(self):
        for v in self.samples.values():
            v.clear()


def build_runner(engine_path):
    """Load engine, read I/O shapes from the engine itself, return an infer closure."""
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    in_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    in_shape = tuple(engine.get_tensor_shape(in_name))
    out_shape = tuple(engine.get_tensor_shape(out_name))

    d_input = cuda.mem_alloc(int(np.prod(in_shape) * 2))   # FP16 = 2 bytes
    d_output = cuda.mem_alloc(int(np.prod(out_shape) * 2))
    h_input = np.random.randn(*in_shape).astype(np.float16)
    h_output = np.empty(out_shape, dtype=np.float16)
    stream = cuda.Stream()

    context.set_tensor_address(in_name, int(d_input))
    context.set_tensor_address(out_name, int(d_output))

    def infer_once():
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    return infer_once, in_shape, out_shape


def measure_model(name, engine_path):
    log("\n" + "=" * 70)
    log(f"MODEL: {name}   ({engine_path})")
    log("=" * 70)

    infer_once, in_shape, out_shape = build_runner(engine_path)
    log(f"Input shape {in_shape}  ->  Output shape {out_shape}")

    for _ in range(WARMUP):
        infer_once()

    sampler = PowerSampler()
    sampler.start()
    time.sleep(1.0)  # let tegrastats spin up

    # 1. idle baseline
    log(f"Measuring idle baseline for {IDLE_SECONDS}s ...")
    sampler.reset()
    time.sleep(IDLE_SECONDS)
    idle = sampler.averages_w()

    # 2. active inference loop
    log(f"Running inference loop for {RUN_SECONDS}s ...")
    sampler.reset()
    n = 0
    t0 = time.time()
    while time.time() - t0 < RUN_SECONDS:
        infer_once()
        n += 1
    elapsed = time.time() - t0
    active = sampler.averages_w()
    sampler.stop()

    # 3. report
    lat_ms = elapsed / n * 1000
    thpt = n / elapsed
    log(f"Inferences: {n}   Wall: {elapsed:.2f}s   "
        f"Latency: {lat_ms:.4f} ms   Throughput: {thpt:.1f} qps")
    log(f"{'Rail':<18}{'Idle(W)':>10}{'Active(W)':>11}"
        f"{'E/inf(uJ)':>12}{'dynE/inf(uJ)':>14}")
    results = {}
    for rail in RAILS:
        p_idle, p_act = idle[rail], active[rail]
        e_inf = p_act * elapsed / n * 1e6        # uJ
        e_dyn = (p_act - p_idle) * elapsed / n * 1e6
        results[rail] = (p_idle, p_act, e_inf, e_dyn)
        log(f"{rail:<18}{p_idle:>10.3f}{p_act:>11.3f}{e_inf:>12.3f}{e_dyn:>14.3f}")
    return name, lat_ms, thpt, results


def main():
    log("#" * 70)
    log(f"# Jetson Orin Nano power measurement   {datetime.datetime.now().isoformat()}")
    log(f"# RUN_SECONDS={RUN_SECONDS}  IDLE_SECONDS={IDLE_SECONDS}  "
        f"tegra_interval={TEGRA_INTERVAL_MS}ms")
    log("#" * 70)

    summary = []
    for name, path in MODELS:
        try:
            summary.append(measure_model(name, path))
        except Exception as e:
            log(f"[ERROR] {name} ({path}): {e}")
        time.sleep(2.0)  # let device settle between models

    # summary table (CPU_GPU_CV rail = inference power)
    log("\n" + "=" * 70)
    log("SUMMARY  (VDD_CPU_GPU_CV rail = inference-attributable)")
    log("=" * 70)
    log(f"{'Model':<12}{'Lat(ms)':>10}{'Thpt(qps)':>11}"
        f"{'P_act(W)':>10}{'E/inf(uJ)':>12}{'dynE/inf(uJ)':>14}")
    for name, lat, thpt, res in summary:
        _, p_act, e_inf, e_dyn = res["VDD_CPU_GPU_CV"]
        log(f"{name:<12}{lat:>10.4f}{thpt:>11.1f}"
            f"{p_act:>10.3f}{e_inf:>12.3f}{e_dyn:>14.3f}")

    flush_log()
    print(f"\nResults written to {LOG_PATH}")


if __name__ == "__main__":
    main()
