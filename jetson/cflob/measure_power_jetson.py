"""
Real power / energy measurement for CFLOB on Jetson Orin Nano.

Uses the on-board INA3221 rails via tegrastats (no external meter needed).
Reports per-inference energy on two rails:
  - VDD_IN          : total module power (system-level)
  - VDD_CPU_GPU_CV  : CPU + GPU + CV engines (inference-attributable)

Methodology (addresses reviewer comments):
  1. Measure idle baseline power.
  2. Run a long inference loop; sample power continuously.
  3. energy_per_inf      = P_active_avg * T / N
     dynamic_energy_per_inf = (P_active_avg - P_idle_avg) * T / N

Run:  sudo python3 measure_power_jetson.py
(sudo helps tegrastats sample reliably; otherwise plain python3 is usually fine)
"""
import re
import time
import threading
import subprocess
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "cflob_fp16.engine"
INPUT_SHAPE = (1, 1, 10, 40)
OUTPUT_SHAPE = (1, 3)
RUN_SECONDS = 30          # inference loop duration
IDLE_SECONDS = 10         # idle baseline duration
TEGRA_INTERVAL_MS = 50    # tegrastats sample interval

# tegrastats prints e.g.:  VDD_IN 3456mW/3456mW VDD_CPU_GPU_CV 1234mW/1234mW VDD_SOC 800mW/800mW
RAIL_RE = re.compile(r"(VDD_IN|VDD_CPU_GPU_CV|VDD_SOC)\s+(\d+)mW")


class PowerSampler(threading.Thread):
    """Background thread that parses tegrastats and accumulates rail power samples."""
    def __init__(self, interval_ms=TEGRA_INTERVAL_MS):
        super().__init__(daemon=True)
        self.interval_ms = interval_ms
        self.samples = {"VDD_IN": [], "VDD_CPU_GPU_CV": [], "VDD_SOC": []}
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


def build_runner():
    logger = trt.Logger(trt.Logger.WARNING)
    with open(ENGINE_PATH, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    d_input = cuda.mem_alloc(int(np.prod(INPUT_SHAPE) * 2))
    d_output = cuda.mem_alloc(int(np.prod(OUTPUT_SHAPE) * 2))
    h_input = np.random.randn(*INPUT_SHAPE).astype(np.float16)
    h_output = np.empty(OUTPUT_SHAPE, dtype=np.float16)
    stream = cuda.Stream()

    context.set_tensor_address(engine.get_tensor_name(0), int(d_input))
    context.set_tensor_address(engine.get_tensor_name(1), int(d_output))

    def infer_once():
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()

    return infer_once


def main():
    infer_once = build_runner()

    # warmup
    for _ in range(50):
        infer_once()

    sampler = PowerSampler()
    sampler.start()
    time.sleep(1.0)  # let tegrastats spin up

    # ---- 1. idle baseline ----
    print(f"Measuring idle baseline for {IDLE_SECONDS}s ...")
    sampler.reset()
    time.sleep(IDLE_SECONDS)
    idle = sampler.averages_w()

    # ---- 2. active inference loop ----
    print(f"Running inference loop for {RUN_SECONDS}s ...")
    sampler.reset()
    n = 0
    t0 = time.time()
    while time.time() - t0 < RUN_SECONDS:
        infer_once()
        n += 1
    elapsed = time.time() - t0
    active = sampler.averages_w()
    sampler.stop()

    # ---- 3. report ----
    lat_ms = elapsed / n * 1000
    print("\n========== RESULTS ==========")
    print(f"Inferences: {n}   Wall time: {elapsed:.2f}s   Latency: {lat_ms:.4f} ms")
    print(f"{'Rail':<18}{'Idle(W)':>10}{'Active(W)':>11}"
          f"{'E/inf(uJ)':>12}{'dynE/inf(uJ)':>14}")
    for rail in ("VDD_IN", "VDD_CPU_GPU_CV", "VDD_SOC"):
        p_idle, p_act = idle[rail], active[rail]
        e_inf = p_act * elapsed / n * 1e6        # uJ
        e_dyn = (p_act - p_idle) * elapsed / n * 1e6
        print(f"{rail:<18}{p_idle:>10.3f}{p_act:>11.3f}{e_inf:>12.3f}{e_dyn:>14.3f}")
    print("\nReport VDD_CPU_GPU_CV as inference power; VDD_IN as system-level.")


if __name__ == "__main__":
    main()
