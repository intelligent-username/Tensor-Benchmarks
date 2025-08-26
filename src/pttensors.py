"""
Tensor Operations Implemented in Pytorch
Includes function definitions for mass benchmarking (PT takes a long time to startup otherwise)
Example tests run when file is executed as main
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Iterable, List, Sequence, Tuple

import torch


def _is_number(s: str) -> bool:
	"""
	Check if a string can be converted to a number.

	Args:
		s (str): The string to check.

	Returns:
		bool: True if the string can be converted to a number, False otherwise.
	"""
	try:
		float(s)
		return True
	except Exception:
		return False


def _parse_bench_file_multi(path: str):
	"""
	Parse a benchmark file containing multiple test cases.

	Args:
		path (str): Path to the benchmark file.

	Returns:
		list: A list of tuples representing test cases. Each tuple contains the operation, tensors, and scalar (if applicable).
	"""

	# Accept multiple test blocks and support scalar-fill after DATA
	tokens = []
	with open(path, "r") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith('#'):
				continue
			tokens.extend(line.split())

	cases = []
	i = 0

	def read_tensor():
		nonlocal i
		tag = tokens[i]; i += 1  # 'A' or 'B'
		if tokens[i] != "SHAPE":
			raise ValueError("Expected SHAPE after tensor tag")
		i += 1
		d = int(tokens[i]); i += 1
		shape = tuple(int(tokens[i + k]) for k in range(d))
		i += d
		if tokens[i] != "DATA":
			raise ValueError("Expected DATA after SHAPE")
		i += 1
		total = 1
		for s in shape:
			total *= s
		if i >= len(tokens) or not _is_number(tokens[i]):
			raise ValueError("Expected numeric after DATA")
		first = float(tokens[i]); i += 1
		vals = [first]
		start_i = i
		while len(vals) < total and i < len(tokens) and _is_number(tokens[i]):
			vals.append(float(tokens[i])); i += 1
		if len(vals) < total:
			# scalar fill; consume any trailing numeric tokens to align to next tag
			while i < len(tokens) and _is_number(tokens[i]):
				i += 1
			arr = torch.full(shape, first, dtype=torch.float32)
		else:
			arr = torch.tensor(vals, dtype=torch.float32).reshape(shape)
		return arr

	def resync():
		nonlocal i
		while i < len(tokens) and tokens[i] != "OP":
			i += 1

	while i < len(tokens):
		if tokens[i] != "OP":
			i += 1
			continue
		try:
			i += 1
			if i >= len(tokens):
				break
			op = tokens[i]; i += 1
			scalar = None
			A = B = None
			if op in ("ADD", "MM", "DOT", "BMM"):
				A = read_tensor()
				B = read_tensor()
			elif op == "SCAL":
				A = read_tensor()
				if tokens[i] != "SCALAR":
					raise ValueError("Expected SCALAR for SCAL op")
				i += 1
				if not _is_number(tokens[i]):
					raise ValueError("Expected numeric scalar")
				scalar = float(tokens[i]); i += 1
			else:
				raise ValueError(f"Unknown op {op}")
			cases.append((op, A, B, scalar))
		except Exception:
			resync()
	return cases


def _maybe_barrier(*tensors) -> None:
	"""
	Ensure previous CUDA kernel's complete execution.

	Args:
		*tensors: Variable length argument list of tensors to check for CUDA synchronization.
	"""

	# Ensure previous kernels are complete
	try:
		for t in tensors:
			if isinstance(t, torch.Tensor) and t.is_cuda:
				torch.cuda.synchronize()
				break
	except Exception:
		# Ignore CUDA sync errors in case CUDA is unavailable
		pass


def _bench(op: str, A: torch.Tensor, B: torch.Tensor | None = None, scalar: float | None = None, runs: int | None = None, warmups: int | None = None) -> int:
	"""
	Benchmark a tensor operation and print the results.

	Args:
		op (str): The operation to benchmark (e.g., 'ADD', 'MM', 'SCAL', 'DOT', 'BMM').
		A (torch.Tensor): The first tensor operand.
		B (torch.Tensor, optional): The second tensor operand (if applicable). Defaults to None.
		scalar (float, optional): A scalar value (if applicable). Defaults to None.
		runs (int, optional): Number of benchmark runs. Defaults to the value of the BENCH_RUNS environment variable or 10.
		warmups (int, optional): Number of warmup runs. Defaults to the value of the BENCH_WARMUPS environment variable or 3.

	Returns:
		int: Always returns 0.
	"""

	runs = runs or int(os.getenv("BENCH_RUNS", "10"))
	warmups = warmups or int(os.getenv("BENCH_WARMUPS", "3"))

	# Warmup
	for _ in range(warmups):
		if op == "ADD":
			_ = A + B
		elif op == "MM":
			_ = torch.matmul(A, B)
		elif op == "SCAL":
			_ = A * scalar
		elif op == "DOT":
			_ = torch.dot(A.reshape(-1), B.reshape(-1))
		elif op == "BMM":
			_ = torch.matmul(A, B)
	_maybe_barrier(A, B)

	times = []
	for _ in range(runs):
		_maybe_barrier(A, B)
		t0 = time.perf_counter()
		if op == "ADD":
			res = A + B
		elif op == "MM":
			res = torch.matmul(A, B)
		elif op == "SCAL":
			res = A * scalar
		elif op == "DOT":
			res = torch.dot(A.reshape(-1), B.reshape(-1))
		elif op == "BMM":
			res = torch.matmul(A, B)
		_maybe_barrier(res)
		t1 = time.perf_counter()
		times.append(t1 - t0)

	mean = statistics.fmean(times)
	median = statistics.median(times)
	stdev = statistics.pstdev(times) if len(times) > 1 else 0.0
	result = {
		"op": op,
		"shape_A": list(A.shape),
		"shape_B": list(B.shape) if B is not None else None,
		"runs": runs,
		"warmups": warmups,
		"median": median,
		"mean": mean,
		"stdev": stdev,
		"min": min(times),
		"max": max(times),
		"times": times,
	}
	print(json.dumps(result))
	return 0

# MAIN
if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--bench", type=str, default=None)
	args, _unknown = parser.parse_known_args()

	if args.bench:
		cases = _parse_bench_file_multi(args.bench)
		rc = 0
		for (op, A, B, scalar) in cases:
			rc |= _bench(op, A, B, scalar)
		sys.exit(rc)

	print("Testing PyTorch version.")

	test_values1 = torch.tensor([
		[1, 2, 3],
		[4, 5, 6],
		[7, 8, 9]
	], dtype=torch.float32)

	test_values2 = torch.tensor([
		[9, 8, 7],
		[6, 5, 4],
		[3, 2, 1]
	], dtype=torch.float32)

	# Matrix addition test
	print("MATRIX ADDITION")
	print("------------------------\n")

	print("Matrix 1:")
	print(test_values1)
	print("Matrix 2:")
	print(test_values2)

	matrix3 = test_values1 + test_values2

	print("Result (Matrix 1 + Matrix 2):")
	print(matrix3)

	print("MATRIX MULTIPLICATION")
	print("------------------------\n")

	print("Matrix 1:")
	print(test_values1)
	print("Matrix 2:")
	print(test_values2)

	matrix4 = torch.matmul(test_values1, test_values2)

	print(matrix4)

	print("\nSCALAR MULTIPLICATION")
	print("------------------------\n")

	# same as test_scalar_multiplier
	constant = 3.1

	print("Matrix 1:")
	print(test_values1)

	print(f"\nAfter multiplying by {constant}:")

	scalar_multiple = constant * test_values1
	print(scalar_multiple, "\n")

	print("VECTORS")
	print("------------------------\n")

	vector1 = torch.tensor([1, 2, 3], dtype=torch.float32)
	vector2 = torch.tensor([4, 5, 6], dtype=torch.float32)

	print("Vector 1: ")
	print(vector1)
	print()
	print("Vector 2: ")
	print(vector2)

	result = torch.dot(vector1, vector2)

	print("Result (Dot Product): ")
	print(result.item())
	print()

	print("TESTING n-DIMENSIONAL TENSORS")
	print("------------------------\n")

	tensor1 = torch.tensor([
		1, 2, 3, 4,    5, 6, 7, 8,    9, 10, 11, 12,
		13, 14, 15, 16,  17, 18, 19, 20,  21, 22, 23, 24
	], dtype=torch.float32).reshape(2, 3, 4)

	print("2x3x4 Tensor: ")
	print(tensor1)

	scaled_tens = tensor1 * constant

	print(f"After scaling by {constant}: ")
	print(scaled_tens)
	print()

	print("Testing 4D Tensor")
	print("------------------------\n")

	tensor2 = torch.tensor([
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
	], dtype=torch.float32).reshape(2, 2, 2, 3)

	print("Created:")
	print(tensor2)

	print("\nTesting Batched Matrix Multiplication")
	print("------------------------\n")

	batch_a = torch.tensor([
		[[1, 2, 3], [4, 5, 6]],
		[[7, 8, 9], [10, 11, 12]]
	], dtype=torch.float32)

	batch_b = torch.tensor([
		[[1, 2], [3, 4], [5, 6]],
		[[7, 8], [9, 10], [11, 12]]
	], dtype=torch.float32)

	print("Batch A (2 matrices of 2x3):")
	print(batch_a)
	print("\nBatch B (2 matrices of 3x2):")
	print(batch_b)

	batch_result = torch.matmul(batch_a, batch_b)

	print("\nResult (Batched matrix multiplication, 2 matrices of 2x2):")
	print(batch_result)

	print("\nTesting Tensor Addition with Same Shapes")
	print("------------------------\n")

	tensor3d_copy = torch.tensor([
		1, 2, 3, 4,    5, 6, 7, 8,    9, 10, 11, 12,
		13, 14, 15, 16,  17, 18, 19, 20,  21, 22, 23, 24
	], dtype=torch.float32).reshape(2, 3, 4)

	added_3d = tensor1 + tensor3d_copy

	print("Result (3D Tensor + 3D Tensor, element-wise):")
	print(added_3d)

