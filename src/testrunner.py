#!/usr/bin/env python3
"""Benchmark orchestrator.

Compiles the C implementation, runs all implementations in batched mode over
the bench tests, writes a JSON report, and generates plots/tables.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

ROOT = Path(__file__).parent.parent
BENCH_FILE = ROOT / "benchmarks" / "bench_tests.txt"


def ensure_c_built() -> Path:
    """
    Ensure the C implementation is compiled and up-to-date.

    Returns:
        Path: Path to the compiled C executable.
    """
    exe = ROOT / "src" / "tensors"
    src = ROOT / "src" / "tensors.c"
    if exe.exists() and exe.stat().st_mtime > src.stat().st_mtime:
        return exe
    print("[build] compiling C implementation...")
    cmd = [
        "gcc", str(src), "-std=c99", "-O2", "-Wall", "-Wextra", "-o", str(exe), "-lm",
    ]
    subprocess.run(cmd, check=True)
    return exe


def find_python_interpreter() -> str:
    """
    Find the Python interpreter to use for running benchmarks.

    Returns:
        str: Path to the Python interpreter.
    """
    # Priority: BENCH_PYTHON env -> venv/bin/python -> vevn/bin/python -> .venv/bin/python -> sys.executable
    env_py = os.getenv("BENCH_PYTHON")
    if env_py and Path(env_py).exists():
        return env_py
    candidates = [
        # POSIX venvs
        ROOT / "venv" / "bin" / "python",
        ROOT / "vevn" / "bin" / "python",  # in case folder is named 'vevn'
        ROOT / ".venv" / "bin" / "python",
        # Windows venvs
        ROOT / "venv" / "Scripts" / "python.exe",
        ROOT / "vevn" / "Scripts" / "python.exe",
        ROOT / ".venv" / "Scripts" / "python.exe",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return sys.executable


def split_tests(path: Path) -> List[str]:
    """
    Split a benchmark file into individual test blocks.

    Args:
        path (Path): Path to the benchmark file.

    Returns:
        List[str]: List of test blocks as strings.
    """
    # Return list of test blocks; keep original formatting for correctness
    tests = []
    current = []
    with open(path, "r") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            if line.strip() == "":
                if current:
                    tests.append("".join(current))
                    current = []
                continue
            if line.startswith("OP ") and current:
                tests.append("".join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            tests.append("".join(current))
    return [t for t in tests if t.strip()]


def _parse_shapes_from_block(block: str):
    """
    Parse the operation and shapes from a benchmark block.

    Args:
        block (str): A single benchmark block.

    Returns:
        Tuple[str, List[int], List[int]]: Operation, shape of tensor A, and shape of tensor B (if applicable).
    """
    # Return (op, shapeA, shapeB or None)
    toks = block.strip().split()
    op = toks[1] if len(toks) > 1 else "?"
    # find 'A SHAPE d ...'
    def read_shape(idx):
        # idx points at 'SHAPE'
        d = int(toks[idx + 1])
        dims = list(map(int, toks[idx + 2: idx + 2 + d]))
        return dims, idx + 2 + d
    shapeA = shapeB = None
    i = 0
    while i < len(toks):
        if toks[i] == 'A' and i + 1 < len(toks) and toks[i+1] == 'SHAPE':
            shapeA, i = read_shape(i + 1)
        elif toks[i] == 'B' and i + 1 < len(toks) and toks[i+1] == 'SHAPE':
            shapeB, i = read_shape(i + 1)
        else:
            i += 1
    return op, shapeA, shapeB


def _too_large_for_c(op: str, shapeA, shapeB, max_ops: int = 1_500_000_000) -> bool:
    """
    Determine if a benchmark case is too large for the C implementation.

    Args:
        op (str): Operation type.
        shapeA (List[int]): Shape of tensor A.
        shapeB (List[int]): Shape of tensor B.
        max_ops (int): Maximum allowed operations. Defaults to 1,500,000,000.

    Returns:
        bool: True if the case is too large, False otherwise.
    """
    # Skip computationally expensive cases for naive C implementation
    def prod(arr):
        p = 1
        for x in arr or []:
            p *= x
        return p
    
    if op in ("MM", "BMM") and shapeA and shapeB:
        # Matrix mult work: batch * M * K * N (cubic complexity)
        batch = 1
        a_nd = len(shapeA)
        b_nd = len(shapeB)
        bd = max(a_nd - 2, b_nd - 2)
        for d in range(bd):
            aval = shapeA[d] if d < a_nd - 2 else 1
            bval = shapeB[d] if d < b_nd - 2 else 1
            batch *= max(aval, bval)
        M = shapeA[-2] if a_nd >= 2 else 1
        K = shapeA[-1] if a_nd >= 2 else 1
        N = shapeB[-1] if b_nd >= 2 else 1
        work = batch * M * K * N
        if work > max_ops:  # Skip if > 1.5B ops per call
            return True
    return False


def _too_large_for_vanilla(op: str, shapeA, shapeB, max_elems: int = 200_000) -> bool:
    """
    Determine if a benchmark case is too large for the vanilla Python implementation.

    Args:
        op (str): Operation type.
        shapeA (List[int]): Shape of tensor A.
        shapeB (List[int]): Shape of tensor B.
        max_elems (int): Maximum allowed elements. Defaults to 200,000.

    Returns:
        bool: True if the case is too large, False otherwise.
    """

    # Conservative filter: skip if total elems exceed threshold, otherwise matmul cubic work is HUGE
    def prod(arr):
        p = 1
        for x in arr or []:
            p *= x
        return p
    total = prod(shapeA) + prod(shapeB)
    if total > max_elems:
        return True
    if op in ("MM", "BMM") and shapeA and shapeB:
        # Work estimate: batch * M*K*N for last-two dims
        batch = 1
        a_nd = len(shapeA)
        b_nd = len(shapeB)
        bd = max(a_nd - 2, b_nd - 2)
        for d in range(bd):
            aval = shapeA[d] if d < a_nd - 2 else 1
            bval = shapeB[d] if d < b_nd - 2 else 1
            batch *= max(aval, bval)
        M = shapeA[-2] if a_nd >= 2 else 1
        K = shapeA[-1] if a_nd >= 2 else 1
        N = shapeB[-1] if b_nd >= 2 else 1
        work = batch * M * K * N
        if work > 20_000_000:  # ~20M mults 6 for vanilla
            return True
    return False


def run_impl_batch(label: str, cmd: Sequence[str], bench_texts: Sequence[str], runs: int = 1000, warmups: int = 3, timeout: int = 1000):
    """
    Run a batch of benchmark tests for a specific implementation.

    Args:
        label (str): Label for the implementation (e.g., 'C', 'NumPy').
        cmd (Sequence[str]): Command to execute the implementation.
        bench_texts (Sequence[str]): List of benchmark test blocks.
        runs (int): Number of benchmark runs. Defaults to 1000.
        warmups (int): Number of warmup runs. Defaults to 3.
        timeout (int): Timeout in seconds. Defaults to 1000.

    Returns:
        List[Dict]: Results of the benchmark tests.
    """

    # Write all tests into a single file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".bench.txt") as tf:
        tf.write("\n\n".join(bench_texts))
        tf.flush()
        tmp_path = tf.name
    env = os.environ.copy()
    env["BENCH_RUNS"] = str(runs)
    env["BENCH_WARMUPS"] = str(warmups)
    try:
        result = subprocess.run(
            cmd + ["--bench", tmp_path],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=timeout,
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("{" )]
        data = []
        for ln in lines:
            try:
                obj = json.loads(ln)
                obj["impl"] = label
                data.append(obj)
            except json.JSONDecodeError:
                continue
        if result.returncode != 0 and not data:
            return [{"impl": label, "error": result.stderr or result.stdout}]
        return data
    except subprocess.TimeoutExpired:
        return [{"impl": label, "error": f"timeout after {timeout}s"}]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def main() -> int:
    """
    Main function to orchestrate the benchmark process.

    Returns:
        int: Exit code.
    """

    if not BENCH_FILE.exists():
        print(f"Bench file not found: {BENCH_FILE}")
        return 1
    exe = ensure_c_built()
    tests = split_tests(BENCH_FILE)
    print(f"Discovered {len(tests)} tests from {BENCH_FILE.name}")

    pybin = find_python_interpreter()
    if pybin != sys.executable:
        print(f"[env] using interpreter: {pybin}")
    impls = [
        ("C", [str(exe)]),
        ("NumPy", [pybin, str(ROOT / "src" / "nptensors.py")]),
        ("PyTorch", [pybin, str(ROOT / "src" / "pttensors.py")]),
        ("Vanilla", [pybin, str(ROOT / "src" / "tensors.py")]),
    ]

    runs = int(os.getenv("BENCH_RUNS", "100"))
    warmups = int(os.getenv("BENCH_WARMUPS", "3"))
    timeout = int(os.getenv("BENCH_TIMEOUT", "600"))
    print(f"[env] runs={runs} warmups={warmups} timeout={timeout}s")

    # Batch: run each implementation once over all test blocks
    results = []
    for label, cmd in impls:
        bench_blocks = tests
        if label == "Vanilla":
            skip_flag = os.getenv("VANILLA_SKIP_OVERLARGE", "1") not in ("0", "false", "False")
            if skip_flag:
                filtered = []
                skipped = 0
                for blk in tests:
                    op, sA, sB = _parse_shapes_from_block(blk)
                    if _too_large_for_vanilla(op, sA, sB):
                        skipped += 1
                    else:
                        filtered.append(blk)
                bench_blocks = filtered
                if skipped:
                    print(f"[Vanilla] skipping {skipped} oversized tests; processing {len(filtered)}")
        elif label == "C":
            skip_flag = os.getenv("C_SKIP_EXPENSIVE", "1") not in ("0", "false", "False")
            if skip_flag:
                filtered = []
                skipped = 0
                for blk in tests:
                    op, sA, sB = _parse_shapes_from_block(blk)
                    if _too_large_for_c(op, sA, sB):
                        skipped += 1
                    else:
                        filtered.append(blk)
                bench_blocks = filtered
                if skipped:
                    print(f"[C] skipping {skipped} computationally expensive test(s); processing {len(filtered)}")
        print(f"\n== Running {label} over {len(bench_blocks)} tests in one batch ==")
        batch_res = run_impl_batch(label, cmd, bench_blocks, runs=runs, warmups=warmups, timeout=timeout)
        for obj in batch_res:
            if "error" in obj:
                print(f"{label}: ERROR: {obj['error'][:200]}")
            else:
                print(f"{label} {obj['op']}: median={obj['median']:.6g}s mean={obj['mean']:.6g}s stdev={obj['stdev']:.3g}s")
        results.extend(batch_res)

    # Optionally, write a JSON report
    report_path = ROOT / "benchmarks" / "bench_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote report to {report_path}")

    # Display results using results.py
    print("\nVisualizing Results...")
    try:
        subprocess.run([find_python_interpreter(), str(ROOT / "src" / "results.py")], check=False)
    except Exception as e:
        print(f"[error] Could not display results: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
