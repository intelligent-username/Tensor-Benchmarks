# Tensor Benchmarking

Which framework (implementation) is most efficient for doing tensor arithmetic? This project explores that question through unit testing. Many features have been implemented to "even-out the playing field" (i.e. the C program would be faster since C is a compiled language, so that is controlled for, or Vanilla Python would be faster than the libraries for smaller operations due to overhead, which is also (mostly) controlled for). The goal is to specifically test the speed of mathematical operations.

1. **C** - Custom tensor library with manual memory management
2. **NumPy** - Python with NumPy
3. **PyTorch** - Python with PyTorch
4. **Vanilla Python** - Tensor operations using Python Lists.

## Operations Benchmarked

- **ADD**: Element-wise tensor addition
- **MM**: Matrix multiplication (2D)
- **BMM**: Batched matrix multiplication (3D)
- **SCAL**: Scalar multiplication
- **DOT**: Vector dot product (1D)

## Quick Start

### Setup Virtual Environment

```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```

### Run Individual Implementations

**Demo modes (show results):**
For each test, make sure you first navigate to the `src` folder.

```bash
    cd src
```

#### C

```bash
    gcc tensors.c -std=c99 -O2 -Wall -Wextra -o tensors -lm
    ./tensors
```

#### NumPy

```bash
    python nptensors.py
```

#### PyTorch

```bash
    python pttensors.py
```

#### Vanilla Python

```bash
    python tensors.py
```

**Benchmark modes (JSON timing output):**

```bash
    # Create a simple test case
    echo "OP ADD
    A SHAPE 2 3 3 DATA 1 2 3 4 5 6 7 8 9
    B SHAPE 2 3 3 DATA 9 8 7 6 5 4 3 2 1" > simple_test.txt

    # Run benchmarks
    cd src
    ./tensors --bench ../benchmarks/simple_test.txt
    python3 nptensors.py --bench ../benchmarks/simple_test.txt
    python3 pttensors.py --bench ../benchmarks/simple_test.txt  
    python3 tensors.py --bench ../benchmarks/simple_test.txt
```

### Run Centralized Benchmark Suite

```bash
    # Quick test with reduced warmup (runs now default to 200 for stability)
    BENCH_WARMUPS=1 python3 src/testrunner.py

    # Override runs if needed (minimum enforced to 200)
    BENCH_RUNS=100 BENCH_WARMUPS=3 python3 src/testrunner.py
```

Results are saved to `benchmarks/bench_report.json`. Plots use log-scale with decade ticks for clean, comparable visuals.

## File Structure

- `src/` - All source code files
  - `tensors.c` + `test_cases.h` - C implementation with test data
  - `nptensors.py` - NumPy implementation
  - `pttensors.py` - PyTorch implementation
  - `tensors.py` - Vanilla Python implementation
  - `testrunner.py` - Centralized benchmark coordinator
  - `results.py` - Visualization and reporting script
- `benchmarks/` - Benchmark data and results
  - `bench_tests.txt` - Large test suite (15 test cases, 300+ lines)
  - `bench_report.json` - Generated benchmark results
- `results/` - Generated plots and tables (bar/, box/, table/)
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Benchmark Format

Test cases use a simple tagged format:

```md
    OP <operation_name>
    A SHAPE <ndim> <dim1> <dim2> ... DATA <val1> <val2> ...
    B SHAPE <ndim> <dim1> <dim2> ... DATA <val1> <val2> ...
    # or for scalar ops:
    SCALAR <value>
```

## Results Summary

From quick testing (3 runs each):

- **C**: Fastest (1-2μs), but limited precision in timing
- **Vanilla Python**: Often competitive for small operations
- **NumPy**: 2-7μs, good balance of speed and features
- **PyTorch**: 5-25μs, higher overhead but GPU-capable

## Environment Controls

- `BENCH_RUNS=N` - Number of timing iterations per test. Higher values improve accuracy but increase runtime.
- `BENCH_WARMUPS=N` - Number of warmup runs before timing. Warmups help stabilize results by allowing caches and JIT optimizations to settle.
- `VANILLA_SKIP_OVERLARGE=1` - Skips overly large test cases for the Vanilla Python implementation to avoid excessive runtime.
- `C_SKIP_EXPENSIVE=1` - Skips computationally expensive test cases for the C implementation to prevent long execution times.
- Most Importantly, a virtual environment for consistent Python dependencies

Feel free to change these controls in order to modify the parameters of the tests (i.e. higher `BENCH_RUNS`, for example, would lead to more accurate results).

## Notes

- All implementations use `time.perf_counter()` for consistent timing
- C uses `clock()` for portability
- PyTorch includes CUDA synchronization barriers when using GPU
- Results include median, mean, standard deviation, min/max times
- Some tests are skipped due to sheer size (C & Vanilla Python implementations aren't optimized for them, no point spending hours waiting)

## Setup

### Prerequisites

1. **Python**: Ensure Python 3.8 or higher is installed on your system. You can check your Python version with:

    ```bash
        python3 --version
    ```

2. **C Compiler**: Install a C compiler such as GCC. On Linux, you can install GCC with:

    ```bash
    sudo apt update && sudo apt install build-essential
    ```

3. **Python Libraries**: The required Python libraries are listed in `requirements.txt`.

### Steps to Set Up

1. **Create a Virtual Environment**:

   ```bash
        python3 -m venv venv
   ```

2. **Activate the Virtual Environment**:

   ```bash
        source venv/bin/activate
   ```

3. **Install Dependencies**:

   ```bash
        pip install -r requirements.txt
   ```

4. **Build the C Implementation**:

   ```bash
        cd src
        gcc tensors.c -std=c99 -O2 -Wall -Wextra -o tensors -lm
   ```

Once these steps are complete, you are ready to run the benchmarks and tests as described in the sections below.

## Building C Implementation

```bash
    cd src
    gcc tensors.c -std=c99 -O2 -Wall -Wextra -o tensors -lm
```
