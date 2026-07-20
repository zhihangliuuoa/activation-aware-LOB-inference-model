# Activation-Aware Neural Architecture for Sustainable Large-Scale Limit-Order-Book Inference on Embedded GPU Fleets

Code and pretrained models for the paper *"Activation-Aware Neural Architecture for Sustainable Large-Scale Limit-Order-Book Inference on Embedded GPU Fleets"* (submitted to IEEE Transactions on Sustainable Computing).

ConvFormerLOB (CFLOB) is a power-efficient neural architecture for limit order book (LOB) price trend prediction, co-designed with the memory hierarchy of embedded GPUs. On an NVIDIA Jetson Orin Nano, CFLOB achieves **5.6x / 12.9x higher throughput**, **5.6x / 12.9x lower latency**, and **6.6x / 29.3x lower energy per inference** compared with DeepLOB and Axial-LOB respectively, while improving the F1 score by up to **6.75 percentage points** on FI-2010.

## Repository Structure

```
.
├── models/                       # PyTorch model definitions, pretrained weights, ONNX export
│   ├── CFLOB.py                  # CFLOB (ours)
│   ├── deeplob.py                # DeepLOB baseline
│   ├── axiallob.py               # Axial-LOB baseline
│   ├── vitlob.py                 # ViT-LOB baseline
│   ├── cflob_model               # pretrained weights (torch.save format)
│   ├── deeplob_model / axiallob_model / Vit-LOB_model.pt
│   ├── export_pth.py             # convert saved models to .pth state dicts
│   ├── export_onnx_cflob.py      # export CFLOB to ONNX
│   └── export_onnx_other.py      # export baselines to ONNX
├── jetson/                       # deployment & measurement on Jetson Orin Nano
│   ├── measure_power_all.py      # per-inference energy for all models (INA3221 via tegrastats)
│   ├── cflob/                    # CFLOB ONNX, prebuilt FP16 TensorRT engine, benchmark scripts
│   │   ├── test_latency_jetson.py     # latency / throughput
│   │   ├── test_memory_jetson.py      # GPU memory & EMC utilization
│   │   ├── measure_power_jetson.py    # power / energy (CFLOB only)
│   │   └── layer_info.json            # TensorRT engine layer inspection
│   └── other_model/              # baseline ONNX models and engines
└── results/                      # raw measurement logs reported in the paper
```

## Environment

**Training / export (workstation):**

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.5.1, CUDA 12.x on an NVIDIA RTX 4080.

**Deployment (NVIDIA Jetson Orin Nano, 15 W mode):**
JetPack 6.x ships TensorRT and PyCUDA. No extra installation is required beyond `numpy`.

## Dataset

We use the public **FI-2010** benchmark (Ntakaris et al., *"Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods"*, Journal of Forecasting, 2018):

- Download: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

We follow the standard setup: z-score normalized data, the first 7 days for training and the last 3 days for testing, prediction horizons k = 10, 20, 50, 100. CFLOB uses a sliding window of 10 most recent LOB snapshots (input shape `1x1x10x40`).

Training hyperparameters are listed in the paper (Lion optimizer, lr 5e-5, weight decay 2e-3, 300 epochs, cosine schedule with k-decay, label smoothing 0.1, dropout 0.1, stochastic depth).

## Usage

### 1. Export ONNX

Run from the `models/` directory so relative paths resolve:

```bash
cd models
python export_pth.py            # convert saved models to .pth state dicts
python export_onnx_cflob.py     # -> cflob.onnx (+ cflob.onnx.data)
python export_onnx_other.py     # -> deeplob.onnx / axiallob.onnx / vitlob.onnx
```

### 2. Build TensorRT engines (on the Jetson)

Prebuilt FP16 engines for Jetson Orin Nano (JetPack 6 / TensorRT 10) are included. TensorRT engines are **device- and version-specific**; rebuild them on your own device with:

```bash
trtexec --onnx=cflob.onnx --fp16 --saveEngine=cflob_fp16.engine
```

### 3. Benchmarks (on the Jetson)

```bash
cd jetson/cflob
python3 test_latency_jetson.py        # latency & throughput
python3 test_memory_jetson.py         # GPU memory / scratch / EMC utilization
sudo python3 measure_power_jetson.py  # power & energy per inference (tegrastats)

cd ..
sudo python3 measure_power_all.py     # energy comparison across all models
```

Power is sampled from the on-board INA3221 sensor (`VDD_CPU_GPU_CV` rail) at 50 ms intervals during a sustained inference loop; per-inference energy is computed as mean active power x mean latency, with a 10 s idle baseline subtracted for dynamic energy. See `results/` for the raw logs behind the tables in the paper.

## Results (Jetson Orin Nano, TensorRT FP16, batch = 1)

| Model | Latency (ms) | Throughput (qps) | Power (W) | Energy (uJ) | Peak EMC (%) |
|---|---|---|---|---|---|
| DeepLOB | 1.786 | 560 | 1.81 | 3,236.8 | 4 |
| Axial-LOB | 4.094 | 244 | 3.51 | 14,370.5 | 28 |
| **CFLOB (ours)** | **0.318** | **3,145** | **1.54** | **489.7** | **2** |

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2026cflob,
  title   = {Activation-Aware Neural Architecture for Sustainable Large-Scale Limit-Order-Book Inference on Embedded GPU Fleets},
  author  = {Liu, Zhihang and Chi, Cheng-Lung and Li, Jiale and Ma, Sean Longyu and Ueng, Yeong-Luh and Sham, Chiu-Wing},
  journal = {Under review},
  year    = {2026}
}
```
