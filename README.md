# NMF-IVF

Approximate nearest neighbour search using an inverted index constructed via Non-negative Matrix Factorization (NMF). Designed for sparse high-dimensional vectors (e.g. BM25/SPLADE representations).

## Requirements

- CMake ≥ 3.20
- A C++20 compiler (GCC 12+ or Clang 15+)
- [vcpkg](https://github.com/microsoft/vcpkg)
- OpenMP

## Setup

### 1. Clone and configure vcpkg (if needed)

```bash
git clone https://github.com/microsoft/vcpkg ~/vcpkg
~/vcpkg/bootstrap-vcpkg.sh -disableMetrics
```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env variables as needed
```

### 3. Configure and build

```bash
chmod +x configure.sh
./configure.sh
cmake --build build --parallel
```

## Running

### Directly

```bash
./build/main --dataset nq
./build/main --dataset fiqa-dev
```

Override any default parameter:

```bash
./build/main --dataset nq --nprobe 50 --n-components 2048 --threads 16
```

All available flags:

| Flag | Default (nq) | Default (fiqa-dev) | Description |
|---|---|---|---|
| `--dataset` | `nq` | — | Dataset preset |
| `--n-components` | 3000 | 512 | Number of NMF components |
| `--batch-size` | 20000 | 5000 | Mini-batch size |
| `--max-iter` | 30 | 50 | Maximum NMF iterations |
| `--forget-factor` | 0.7 | 0.7 | Mini-batch forget factor |
| `--m` | 5000 | 500 | Number of index lists |
| `--nprobe` | 20 | 16 | Lists probed at query time |
| `--threads` | 8 | 8 | OpenBLAS / OMP threads |

