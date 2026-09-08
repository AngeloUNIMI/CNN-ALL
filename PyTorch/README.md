# CNN-ALL-PyTorch

GPU-oriented Python/PyTorch refactor of the MATLAB **CNN-ALL** pipeline for adaptive unsharpening and white-blood-cell classification.

The default configuration targets the five-class RAABIN-WBC directory layout:

```text
imgs/orig/RAABIN-WBC/
├── Basophil/
├── Eosinophil/
├── Lymphocyte/
├── Monocyte/
└── Neutrophil/
```

A compatibility configuration for ALL-IDB2 is also included.

For the shortest server setup path, see [`QUICKSTART_UBUNTU.md`](QUICKSTART_UBUNTU.md). A file-by-file responsibility map is in [`MATLAB_TO_PYTORCH_MAP.md`](MATLAB_TO_PYTORCH_MAP.md).

## What is implemented

The project ports the complete active research path used by the supplied MATLAB code:

- folder-based RAABIN-WBC import and class mapping;
- optional ALL-IDB2 filename-label parsing and legacy ROI preparation;
- FQPath focus-quality assessment using the supplied `FQPath_kernel.mat` coefficients;
- adaptive unsharp masking with radii `1, 3, ..., 17`;
- threshold search based on focus/label dependence;
- shared train/test splits and PCANet threshold tuning;
- GPU-batched PCANet patch extraction, covariance accumulation, eigendecomposition, convolution, binary hashing, and local histograms;
- chunked 1-NN classification, CMC, binary metrics, and multiclass macro metrics;
- pretrained feature extraction, fine-tuning, and Grad-CAM for:
  - AlexNet
  - VGG-16
  - VGG-19
  - ResNet-18
  - ResNet-50
  - ResNet-101
  - DenseNet-201
- standalone export of the adaptively unsharpened database;
- checkpoints, resumable phases, structured logs, CSV/JSON summaries, and headless execution.

The sharpening rule is preserved deliberately: **each candidate radius is applied to the original image, never cumulatively to the result of the previous radius**.

## Main performance changes

The Python pipeline avoids the largest redundancy in the MATLAB loop. Focus estimation, split generation, and PCANet threshold tuning are performed once per experimental iteration and reused by all seven CNNs.

GPU-oriented operations include:

- staged model placement so only one large fine-tuned CNN occupies GPU memory at a time;
- batched FQPath filtering;
- batched Lab-luminance unsharp masking;
- `torch.nn.functional.unfold` for PCANet patches;
- accelerator covariance accumulation and `torch.linalg.eigh`;
- batched PCANet responses and histogram construction;
- automatic mixed precision for CNN inference/training;
- channels-last CNN tensors on CUDA;
- pinned-memory/persistent-worker data loading;
- chunked accelerator distance computation to bound memory use.

Original and unsharp fine-tuning runs use the same initialization, data order, and augmentation random seed for a controlled comparison.

## Installation on Ubuntu

Python 3.10 or newer is required. A virtual environment is recommended.

```bash
cd CNN-ALL-PyTorch
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install a CUDA-enabled PyTorch/torchvision build appropriate for the server driver using the official selector at `https://pytorch.org/get-started/locally/`. Then install this project:

```bash
python -m pip install -e .
```

Verify the environment and all seven architectures:

```bash
cnn-all doctor
nvidia-smi
```

The first pretrained run downloads torchvision weights into the Torch Hub cache. To require already-cached weights instead, set:

```yaml
cnn:
  download_weights: false
```

## RAABIN-WBC run

Place the database in the default structure shown above, or edit `dataset.root` in `configs/raabin_wbc.yaml`.

Run the whole pipeline:

```bash
cnn-all run --config configs/raabin_wbc.yaml
```

Run individual phases:

```bash
cnn-all prepare     --config configs/raabin_wbc.yaml
cnn-all focus-cache --config configs/raabin_wbc.yaml
cnn-all shared      --config configs/raabin_wbc.yaml
cnn-all networks    --config configs/raabin_wbc.yaml
cnn-all export      --config configs/raabin_wbc.yaml
```

Restrict an experiment while debugging:

```bash
cnn-all networks \
  --config configs/raabin_wbc.yaml \
  --backbones resnet18 resnet50 \
  --iterations 1 2
```

Iteration numbers on the command line are 1-based.

## Detached SSH execution

Using `nohup`:

```bash
./scripts/run_raabin_nohup.sh configs/raabin_wbc.yaml raabin_pytorch.log
```

Monitor it later:

```bash
tail -f raabin_pytorch.log
cat raabin_pytorch.log.pid
```

Using `tmux`:

```bash
./scripts/run_raabin_tmux.sh configs/raabin_wbc.yaml raabin-pytorch raabin_pytorch.log
tmux attach -t raabin-pytorch
```

## Output structure

```text
Results_PyTorch/RAABIN_WBC/
├── run.log
├── resolved_config.yaml
├── shared/
│   ├── split_01.json
│   ├── dependence_01.json
│   ├── focus_tuning_01.csv
│   └── summary.json
├── alexnet/
│   ├── iteration_01/
│   │   ├── pretrained_features.json
│   │   ├── features_original.pt
│   │   ├── features_unsharp.pt
│   │   ├── finetune_original/
│   │   ├── finetune_unsharp/
│   │   ├── gradcam/
│   │   └── summary.json
│   └── summary.json
└── ...
```

The standalone enhanced database is exported by default to:

```text
imgs/unsharp/RAABIN-WBC/
├── Basophil/
├── Eosinophil/
├── Lymphocyte/
├── Monocyte/
├── Neutrophil/
├── export_info.csv
├── export_info.json
└── README.txt
```

## Configuration notes

### GPU memory

Start conservatively and then increase:

```yaml
focus:
  batch_size: 32
pcanet:
  fit_batch_size: 4
  feature_batch_size: 16
cnn:
  batch_size: 20
  feature_batch_size: 64
```

For an out-of-memory error, reduce the relevant batch size. PCANet covariance memory is mostly controlled by patch dimensionality, while `covariance_patch_chunk` bounds temporary matrix-multiplication work.

### Numerical mode

`pcanet.dtype: float32` is faster on GPU. Set `float64` for closer numerical comparison with MATLAB at a substantial performance cost.

### CNN preprocessing

The default is ImageNet normalization, appropriate for torchvision pretrained weights:

```yaml
cnn:
  preprocess_mode: imagenet
```

A compatibility option is available:

```yaml
cnn:
  preprocess_mode: legacy_per_image_center
```

### Reproducibility

For maximum repeatability:

```yaml
runtime:
  deterministic: true
  cudnn_benchmark: false
  allow_tf32: false
```

This is slower than the default throughput-oriented configuration.

## Testing

```bash
python -m pip install -e '.[dev]'
pytest
```

A tiny synthetic integration dataset can be generated with:

```bash
cnn-all make-smoke-data --output smoke_data --images-per-class 4 --size 64
cnn-all run --config configs/smoke_test.yaml
```

## Scope and parity

The RAABIN-WBC path is the primary, fully implemented path. The supplied MATLAB repository also bundles older research/demo libraries that are not invoked by the active RAABIN launcher. The ALL-IDB2 compatibility module reproduces their intent but the SCD-MA/FastICA wavelet stain path and legacy active-contour stack are approximations rather than bit-for-bit ports. See [`PORTING_NOTES.md`](PORTING_NOTES.md).

GPU and CPU floating-point results need not be bit-identical to MATLAB because eigensolver signs, interpolation, histogram edge conventions, and pretrained model implementations can differ. The test suite checks structural and algorithmic behavior; scientific equivalence should be established by running both pipelines on the same fixed splits.

## Citation

This refactor accompanies the method introduced in:

> A. Genovese, M. S. Hosseini, V. Piuri, K. N. Plataniotis, and F. Scotti, “Acute Lymphoblastic Leukemia detection based on adaptive unsharpening and Deep Learning,” Proc. IEEE ICASSP, 2021, pp. 1205–1209. DOI: 10.1109/ICASSP39728.2021.9414362.

It also incorporates the algorithmic ideas and supplied coefficients for PCANet and FQPath. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## License

GNU General Public License v3.0 or later, matching the supplied CNN-ALL repository.
