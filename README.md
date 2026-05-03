# ParallelDerm — Hair Removal Preprocessing Pipeline

A parallelised dermoscopic image preprocessing pipeline benchmarking **Serial**, **OpenMP**, **MPI**, and **OpenCL** backends against each other. Built as a PDC assignment and extended into a portfolio-grade tool for medical imaging pipelines.

---

## Project Structure

```
Project/
├── backend/                        # C++ backend binaries (compiled)
│   ├── hair_removal_serial
│   ├── hair_removal_omp
│   ├── hair_removal_mpi
│   └── hair_removal_ocl
│
├── gui/                            # C++ Qt6 frontend (compiled binary)
│
├── images/                         # Input dermoscopic images (ISIC dataset)
│
├── processed_images/               # Output organised by method
│   ├── serial/
│   ├── omp/
│   ├── mpi/
│   └── ocl/
│
├── gui.h                           # C++ Qt6 GUI header
├── main.cpp                        # C++ Qt6 GUI entry point
├── gui.py                          # Python PyQt6 GUI (drop-in alternative)
├── main.py                         # Python entry point (calls gui.py)
├── imageprocessing.code-workspace  # VS Code workspace
└── README.md
```

---

## Algorithm

Each backend implements the same morphological hair removal pipeline:

1. **Grayscale conversion** — reduce input to single channel
2. **Black-hat transform** — isolate dark hair strands against lighter skin using a large elliptical structuring element
3. **Binary thresholding** — produce a hair mask from the transform result
4. **Inpainting** — reconstruct skin texture under the mask using the Telea algorithm

---

## Backends

| Method | Parallelism | Notes |
|--------|-------------|-------|
| Serial | None | Baseline reference |
| OMP | CPU threads via OpenMP | Configurable thread count |
| MPI | Multi-process via MPI | `mpirun -np N` |
| OCL | GPU via OpenCL | Requires compatible GPU + driver |

### Building the backends

```bash
# Dependencies: OpenCV, OpenMP, OpenMPI, OpenCL
mkdir build && cd build
cmake ../backend
make -j$(nproc)
```

### CLI interface

All backends accept the same arguments:

```bash
./hair_removal_serial  <input_dir> <output_dir>
./hair_removal_omp     <input_dir> <output_dir> <num_threads>
mpirun -np <N> ./hair_removal_mpi <input_dir> <output_dir>
./hair_removal_ocl     <input_dir> <output_dir>
```

Output files are written as `hair_removed_<original_filename>` into the target directory.

---

## Frontends

Two frontends are provided. Both call the same backend binaries.

### C++ Qt6 GUI (`main.cpp` + `gui.h`)

```bash
# Dependencies: Qt6
mkdir build-gui && cd build-gui
cmake ..
make
./gui
```

### Python PyQt6 GUI (`gui.py`)

A richer, portfolio-grade interface with a **draggable split-view** for before/after comparison, live scan-line progress, per-method benchmark table with speedup ratios, and folder pickers.

```bash
pip install PyQt6
python main.py
# or directly:
python gui.py
```

#### GUI Features

| Feature | Description |
|---------|-------------|
| Split-view viewer | Drag the divider to reveal original vs processed on the same image |
| Live scan-line | Animated progress indicator that tracks output folder in real time |
| Elapsed clock | mm:ss.ms precision timer per run |
| Metrics sidebar | Images done, img/sec throughput, progress percentage |
| Benchmark table | Records runtime for each method and shows speedup vs Serial |
| Folder pickers | Browse for any input/output directory at runtime |
| Method switcher | Instantly compare outputs from different backends |

---

## Dataset

Images from the [ISIC Archive](https://www.isic-archive.com/). Place `.jpg` / `.png` files into `images/` before running.

---

## Sample Benchmark

Results on Intel Core i7 / NVIDIA GTX 1650, 200 ISIC images:

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Serial | ~42.0 | 1.0× |
| OMP (8 threads) | ~7.1 | ~5.9× |
| MPI (8 procs) | ~8.4 | ~5.0× |
| OCL (GPU) | ~3.2 | ~13.1× |

> Run your own benchmarks — results populate the sidebar table automatically after each method completes.

---

## Context

Built for the Parallel & Distributed Computing course (CSC334) and extended as a preprocessing module for an FYP on melanoma detection with skin-tone bias mitigation. Clean, hair-free images improve synthetic image generation quality across the Fitzpatrick scale.

---

## License

MIT
