"""Vanilla Python tensor operations and benchmarking harness.

Implements basic n-D operations (add, scalar mul, dot, 2D matmul, batched matmul),
plus a CLI to run microbenchmarks from a bench file.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Iterable, List, Sequence, Tuple


def shape_of(a: object) -> Tuple[int, ...]:
	s = []
	while isinstance(a, list):
		s.append(len(a))
		a = a[0] if a else []
	return tuple(s)


def zeros(shape: Tuple[int, ...]):
	if not shape:
		return 0.0
	n = shape[0]
	return [zeros(shape[1:]) for _ in range(n)]


def elementwise_add(a, b):
	if isinstance(a, list) and isinstance(b, list):
		return [elementwise_add(x, y) for x, y in zip(a, b)]
	return a + b


def scalar_mul(a, c: float):
	if isinstance(a, list):
		return [scalar_mul(x, c) for x in a]
	return a * c


def dot(v1: Sequence[float], v2: Sequence[float]) -> float:
	return sum(x * y for x, y in zip(v1, v2))


def matmul2d(a: List[List[float]], b: List[List[float]]):
	# shapes: (m x k) @ (k x n)
	m, k1 = len(a), len(a[0])
	k2, n = len(b), len(b[0])
	if k1 != k2:
		raise ValueError("Incompatible shapes for matmul2d")
	out = [[0.0 for _ in range(n)] for _ in range(m)]
	for i in range(m):
		for j in range(n):
			s = 0.0
			for k in range(k1):
				s += a[i][k] * b[k][j]
			out[i][j] = s
	return out


def reshape(flat: List[float], shape: Tuple[int, ...]):
	# simple recursive reshape
	if not shape:
		return flat[0]
	step = 1
	for d in shape[1:]:
		step *= d
	return [reshape(flat[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]


def batched_matmul3d(a: List[List[List[float]]], b: List[List[List[float]]]):
	# Expect shapes (B, M, K) and (B, K, N)
	B = len(a)
	out = []
	for bidx in range(B):
		out.append(matmul2d(a[bidx], b[bidx]))
	return out


def pretty_print(arr, indent: int = 0) -> None:
	"""" 
	If printing, ensure the matrices look the same as NumPy.
	"""
	if not isinstance(arr, list):
		print(arr, end="")
		return
	print("[", end="")
	for i, x in enumerate(arr):
		if isinstance(x, list):
			if i > 0:
				print(",")
				print(" " * (indent + 1), end="")
			else:
				print("", end="")
			pretty_print(x, indent + 1)
		else:
			if i > 0:
				print(", ", end="")
			print(x, end="")
	print("]", end="")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--bench", type=str, default=None)
	args, _unknown = parser.parse_known_args()

	def _is_number(s: str) -> bool:
		try:
			float(s)
			return True
		except Exception:
			return False

	def _parse_bench_file_multi(path):
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
			tag = tokens[i]; i += 1
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
				tensor = reshape([first]*total, shape)
			else:
				tensor = reshape(vals, shape)
			return tensor

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

	def _bench(op, A, B=None, scalar=None, runs=None, warmups=None):
		runs = runs or int(os.getenv("BENCH_RUNS", "10"))
		warmups = warmups or int(os.getenv("BENCH_WARMUPS", "3"))

		# Warmup
		for _ in range(warmups):
			if op == "ADD":
				_ = elementwise_add(A, B)
			elif op == "MM":
				_ = matmul2d(A, B)
			elif op == "SCAL":
				_ = scalar_mul(A, scalar)
			elif op == "DOT":
				_ = dot(sum(A, []), sum(B, [])) if isinstance(A[0], list) else dot(A, B)
			elif op == "BMM":
				_ = batched_matmul3d(A, B)

		times = []
		for _ in range(runs):
			t0 = time.perf_counter()
			if op == "ADD":
				res = elementwise_add(A, B)
			elif op == "MM":
				res = matmul2d(A, B)
			elif op == "SCAL":
				res = scalar_mul(A, scalar)
			elif op == "DOT":
				res = dot(sum(A, []), sum(B, [])) if isinstance(A[0], list) else dot(A, B)
			elif op == "BMM":
				res = batched_matmul3d(A, B)
			t1 = time.perf_counter()
			times.append(t1 - t0)

		mean = statistics.fmean(times)
		median = statistics.median(times)
		stdev = statistics.pstdev(times) if len(times) > 1 else 0.0
		result = {
			"op": op,
			"shape_A": list(shape_of(A)),
			"shape_B": list(shape_of(B)) if B is not None else None,
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

	if args.bench:
		cases = _parse_bench_file_multi(args.bench)
		rc = 0
		for (op, A, B, scalar) in cases:
			rc |= _bench(op, A, B, scalar)
		sys.exit(rc)

	print("Testing vanilla Python version.")

	test_values1 = [
		[1.0, 2.0, 3.0],
		[4.0, 5.0, 6.0],
		[7.0, 8.0, 9.0],
	]

	test_values2 = [
		[9.0, 8.0, 7.0],
		[6.0, 5.0, 4.0],
		[3.0, 2.0, 1.0],
	]

	# Matrix addition test
	print("MATRIX ADDITION")
	print("------------------------\n")

	print("Matrix 1:")
	pretty_print(test_values1); print()
	print("Matrix 2:")
	pretty_print(test_values2); print()

	matrix3 = elementwise_add(test_values1, test_values2)

	print("Result (Matrix 1 + Matrix 2):")
	pretty_print(matrix3); print()

	print("MATRIX MULTIPLICATION")
	print("------------------------\n")

	print("Matrix 1:")
	pretty_print(test_values1); print()
	print("Matrix 2:")
	pretty_print(test_values2); print()

	matrix4 = matmul2d(test_values1, test_values2)

	pretty_print(matrix4); print()

	print("\nSCALAR MULTIPLICATION")
	print("------------------------\n")

	constant = 3.1

	print("Matrix 1:")
	pretty_print(test_values1); print()

	print(f"\nAfter multiplying by {constant}:")

	scalar_multiple = scalar_mul(test_values1, constant)
	pretty_print(scalar_multiple); print("\n")

	print("VECTORS")
	print("------------------------\n")

	vector1 = [1.0, 2.0, 3.0]
	vector2 = [4.0, 5.0, 6.0]

	print("Vector 1: ")
	pretty_print(vector1); print("\n")
	print("Vector 2: ")
	pretty_print(vector2); print()

	result = dot(vector1, vector2)

	print("Result (Dot Product): ")
	print(result)
	print()

	print("TESTING n-DIMENSIONAL TENSORS")
	print("------------------------\n")

	flat = [
		1, 2, 3, 4,    5, 6, 7, 8,    9, 10, 11, 12,
		13, 14, 15, 16,  17, 18, 19, 20,  21, 22, 23, 24
	]
	tensor1 = reshape([float(x) for x in flat], (2, 3, 4))

	print("2x3x4 Tensor: ")
	pretty_print(tensor1); print()

	scaled_tens = scalar_mul(tensor1, constant)

	print(f"After scaling by {constant}: ")
	pretty_print(scaled_tens); print()
	print()

	print("Testing 4D Tensor")
	print("------------------------\n")

	flat2 = [
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
	]
	tensor2 = reshape([float(x) for x in flat2], (2, 2, 2, 3))

	print("Created:")
	pretty_print(tensor2); print()

	print("\nTesting Batched Matrix Multiplication")
	print("------------------------\n")

	batch_a = [
		[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
		[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
	]

	batch_b = [
		[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
		[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
	]

	print("Batch A (2 matrices of 2x3):")
	pretty_print(batch_a); print()
	print("\nBatch B (2 matrices of 3x2):")
	pretty_print(batch_b); print()

	batch_result = batched_matmul3d(batch_a, batch_b)

	print("\nResult (Batched matrix multiplication, 2 matrices of 2x2):")
	pretty_print(batch_result); print()

	print("\nTesting Tensor Addition with Same Shapes")
	print("------------------------\n")

	tensor3d_copy = reshape([float(x) for x in flat], (2, 3, 4))
	added_3d = elementwise_add(tensor1, tensor3d_copy)

	print("Result (3D Tensor + 3D Tensor, element-wise):")
	pretty_print(added_3d); print()

