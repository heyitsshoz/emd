#!/usr/bin/env python3
"""Integrity checks for grid_sinkhorn_fft.py.

These checks are intentionally small and dense. They verify the matrix-free FFT
kernel against explicit dense matrices on cases where dense construction is
safe, including periodic, non-periodic, mixed-boundary, and squared-cost grids.
"""

from __future__ import annotations

import math

import numpy as np

from grid_sinkhorn_fft import (
    ArrayBackend,
    FFTKernel,
    GridGeometry,
    delta_measure,
    dense_cost_matrix,
    normalize_density,
    random_grid_mixture,
    sinkhorn_fft,
    validate_against_dense,
)


def assert_close(name: str, actual: np.ndarray | float, expected: np.ndarray | float, tol: float) -> None:
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    err = float(np.max(np.abs(actual_arr - expected_arr)))
    if not np.isfinite(err) or err > tol:
        raise AssertionError(f"{name}: max error {err:.3e} exceeds tolerance {tol:.3e}")
    print(f"ok: {name} max error {err:.3e}")


def dense_kernel_products(geometry: GridGeometry, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cost = dense_cost_matrix(geometry)
    kernel = np.exp(-cost / geometry.epsilon)
    flat = x.reshape(-1)
    return (kernel @ flat).reshape(geometry.shape), ((cost * kernel) @ flat).reshape(geometry.shape)


def check_kernel_products() -> None:
    backend = ArrayBackend("numpy")
    geometries = [
        GridGeometry((4, 5), (0.7, 1.3), (0, 1), 0.4, "euclidean"),
        GridGeometry((4, 5), (0.7, 1.3), (), 0.4, "euclidean"),
        GridGeometry((3, 4, 5), (0.5, 1.0, 1.5), (1,), 0.7, "euclidean"),
        GridGeometry((4, 5), (0.7, 1.3), (1,), 0.9, "sqeuclidean"),
    ]
    rng = np.random.default_rng(123)
    for geometry in geometries:
        x = rng.random(geometry.shape)
        fft_kernel = FFTKernel(geometry, backend)
        dense_kx, dense_ckx = dense_kernel_products(geometry, x)
        assert_close(f"kernel product {geometry}", fft_kernel.apply_kernel(x), dense_kx, 1e-11)
        assert_close(f"cost-kernel product {geometry}", fft_kernel.apply_cost_kernel(x), dense_ckx, 1e-11)


def check_dense_cost_properties() -> None:
    geometry = GridGeometry((5, 6), (0.25, 0.75), (1,), 0.3, "euclidean")
    cost = dense_cost_matrix(geometry)
    assert_close("dense cost symmetry", cost, cost.T, 0.0)
    assert_close("dense cost diagonal", np.diag(cost), np.zeros(geometry.n_points), 0.0)
    if np.any(cost < 0):
        raise AssertionError("dense cost contains negative entries")
    print("ok: dense cost non-negative")


def check_delta_transport() -> None:
    backend = ArrayBackend("numpy")
    geometry = GridGeometry((8, 8), (2 * math.pi / 8, 2 * math.pi / 8), (0, 1), 0.3, "euclidean")
    a_index = (1, 2)
    b_index = (7, 5)
    a = delta_measure(geometry.shape, a_index, floor=0.0)
    b = delta_measure(geometry.shape, b_index, floor=0.0)
    kernel = FFTKernel(geometry, backend)
    result = sinkhorn_fft(a, b, kernel, max_iter=100, tol=1e-12, check_every=1, tiny=1e-300)
    dq = min(abs(a_index[0] - b_index[0]), geometry.shape[0] - abs(a_index[0] - b_index[0]))
    dp = min(abs(a_index[1] - b_index[1]), geometry.shape[1] - abs(a_index[1] - b_index[1]))
    expected = math.sqrt((dq * geometry.spacings[0]) ** 2 + (dp * geometry.spacings[1]) ** 2)
    assert_close("delta-pair transport cost", result["cost"], expected, 1e-12)
    assert result["converged"] is True
    print("ok: delta-pair converged")


def check_sinkhorn_validation() -> None:
    backend = ArrayBackend("numpy")
    geometry = GridGeometry((7, 7), (2 * math.pi / 7, 2 * math.pi / 7), (0, 1), 0.35, "euclidean")
    rng = np.random.default_rng(456)
    a = random_grid_mixture(geometry.shape, geometry.spacings, geometry.periodic_axes, rng, 3, 0.5, 0.0)
    b = random_grid_mixture(geometry.shape, geometry.spacings, geometry.periodic_axes, rng, 3, 0.5, 0.0)
    kernel = FFTKernel(geometry, backend)
    result = sinkhorn_fft(a, b, kernel, max_iter=2000, tol=1e-10, check_every=10, tiny=1e-300)
    validation = validate_against_dense(
        normalize_density(a),
        normalize_density(b),
        geometry,
        fft_cost=float(result["cost"]),
        max_dense_n=500,
        exact_emd=True,
        max_emd_n=500,
    )
    if validation["status"] != "ok":
        raise AssertionError(f"dense validation skipped/failed: {validation}")
    assert_close(
        "FFT Sinkhorn vs dense Sinkhorn",
        validation["fft_cost"],
        validation["dense_entropic_sinkhorn_cost"],
        1e-7,
    )
    if validation.get("exact_emd_status") != "ok":
        raise AssertionError(f"exact EMD validation skipped/failed: {validation}")
    print("ok: exact EMD comparison available")


def main() -> None:
    check_dense_cost_properties()
    check_kernel_products()
    check_delta_transport()
    check_sinkhorn_validation()
    print("all grid Sinkhorn integrity checks passed")


if __name__ == "__main__":
    main()
