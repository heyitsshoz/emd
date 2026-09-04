#!/usr/bin/env python3
"""Matrix-free Sinkhorn experiments for grid-based physical distance.

This script targets the Wasserstein bottleneck in Wang, Wang, and Wu,
Phys. Rev. E 103, 042209 (2021), when the basis states form a regular
phase-space grid and the basis metric depends only on grid displacement.

It computes the entropically regularized OT plan implicitly:

    P_ij = u_i exp(-C_ij / epsilon) v_j

The dense C and P matrices are never formed. Kernel-vector products are
performed as FFT convolutions on periodic and/or non-periodic grid axes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.fft import fftn, ifftn


Array = np.ndarray


class ArrayBackend:
    """Small adapter for NumPy/SciPy FFT on CPU or CuPy FFT on GPU."""

    def __init__(self, name: str, device_id: int | None = None):
        if name == "auto":
            name = "cupy" if self._cupy_available() else "numpy"
        if name == "numpy":
            self.name = "numpy"
            self.xp = np
            self._fftn = fftn
            self._ifftn = ifftn
            self.device = "cpu"
        elif name == "cupy":
            try:
                import cupy as cp
            except ImportError as exc:
                raise RuntimeError(
                    "CuPy is not installed. Use --backend numpy, or install a CuPy "
                    "wheel matching the CUDA version on the target machine."
                ) from exc
            if device_id is not None:
                cp.cuda.Device(device_id).use()
            self.name = "cupy"
            self.xp = cp
            self._fftn = cp.fft.fftn
            self._ifftn = cp.fft.ifftn
            device = cp.cuda.runtime.getDevice()
            props = cp.cuda.runtime.getDeviceProperties(device)
            device_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
            self.device = f"cuda:{device} ({device_name})"
        else:
            raise ValueError(f"unknown backend: {name}")

    @staticmethod
    def _cupy_available() -> bool:
        try:
            import cupy as cp
            return cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False

    def asarray(self, x: Array):
        return self.xp.asarray(x, dtype=self.xp.float64)

    def to_numpy(self, x) -> Array:
        if self.name == "cupy":
            return self.xp.asnumpy(x)
        return np.asarray(x)

    def scalar(self, x) -> float:
        if self.name == "cupy":
            return float(self.xp.asnumpy(x))
        return float(x)

    def fftn(self, x):
        return self._fftn(x)

    def ifftn(self, x):
        return self._ifftn(x)

    def synchronize(self) -> None:
        if self.name == "cupy":
            self.xp.cuda.Stream.null.synchronize()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_axes(values: Iterable[int] | None, ndim: int) -> tuple[int, ...]:
    if values is None:
        return ()
    axes = tuple(sorted(set(values)))
    bad = [axis for axis in axes if axis < 0 or axis >= ndim]
    if bad:
        raise ValueError(f"axis/axes out of range for ndim={ndim}: {bad}")
    return axes


def normalize_density(x: Array, floor: float = 0.0) -> Array:
    """Return a non-negative array normalized to unit mass."""
    y = np.asarray(x, dtype=np.float64)
    if floor < 0:
        raise ValueError("floor must be non-negative")
    if floor:
        y = np.maximum(y, floor)
    if np.any(y < 0):
        raise ValueError("density contains negative entries")
    total = float(np.sum(y))
    if not np.isfinite(total) or total <= 0:
        raise ValueError("density must have positive finite mass")
    return y / total


def finite_or_none(value: float) -> float | None:
    """Return a finite JSON-safe float, or None for inf/nan."""
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def random_grid_mixture(
    shape: tuple[int, ...],
    spacings: tuple[float, ...],
    periodic_axes: tuple[int, ...],
    rng: np.random.Generator,
    components: int,
    sigma: float,
    floor: float,
) -> Array:
    """Generate a smooth test probability distribution on a mixed-boundary grid."""
    if components <= 0:
        raise ValueError("components must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    ndim = len(shape)
    coords = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    density = np.zeros(shape, dtype=np.float64)
    centers = [rng.uniform(0.0, n, size=components) for n in shape]
    weights = rng.random(components)
    weights /= np.sum(weights)

    for comp in range(components):
        squared = np.zeros(shape, dtype=np.float64)
        for axis in range(ndim):
            delta_index = np.abs(coords[axis] - centers[axis][comp])
            if axis in periodic_axes:
                delta_index = np.minimum(delta_index, shape[axis] - delta_index)
            squared += (delta_index * spacings[axis]) ** 2
        density += weights[comp] * np.exp(-0.5 * squared / sigma**2)

    return normalize_density(density, floor=floor)


def single_grid_gaussian(
    shape: tuple[int, ...],
    spacings: tuple[float, ...],
    periodic_axes: tuple[int, ...],
    center: tuple[float, ...],
    sigma: float,
    floor: float,
) -> Array:
    """Generate one Gaussian-like bump on the configured grid."""
    if len(center) != len(shape):
        raise ValueError("--center-a/--center-b length must match grid dimension")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    coords = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    squared = np.zeros(shape, dtype=np.float64)
    for axis, center_axis in enumerate(center):
        delta_index = np.abs(coords[axis] - center_axis)
        if axis in periodic_axes:
            delta_index = np.minimum(delta_index, shape[axis] - delta_index)
        squared += (delta_index * spacings[axis]) ** 2
    return normalize_density(np.exp(-0.5 * squared / sigma**2), floor=floor)


def delta_measure(shape: tuple[int, ...], index: tuple[int, ...], floor: float) -> Array:
    """Generate a point mass at a grid coordinate."""
    if len(index) != len(shape):
        raise ValueError("--index-a/--index-b length must match grid dimension")
    bad = [axis for axis, value in enumerate(index) if value < 0 or value >= shape[axis]]
    if bad:
        raise ValueError(f"delta index is out of bounds on axes {bad}")
    density = np.zeros(shape, dtype=np.float64)
    density[index] = 1.0
    return normalize_density(density, floor=floor)


def load_measure_file(path: str, key: str | None) -> Array:
    """Load a measure from .npy, .npz, .csv, or whitespace-delimited text."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        data = np.load(source)
    elif suffix == ".npz":
        archive = np.load(source)
        if key is None:
            if len(archive.files) != 1:
                raise ValueError(f"{path} contains multiple arrays; pass --a-key/--b-key")
            key = archive.files[0]
        data = archive[key]
    elif suffix == ".csv":
        data = np.loadtxt(source, delimiter=",")
    else:
        data = np.loadtxt(source)
    return np.asarray(data, dtype=np.float64)


def _tuple_or_none(values: Iterable[float] | None) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(values)


def build_measures(
    args: argparse.Namespace,
    geometry: "GridGeometry",
) -> tuple[Array, Array, dict[str, object]]:
    """Build or load the two probability measures used in a run."""
    if args.a_file or args.b_file:
        if not (args.a_file and args.b_file):
            raise ValueError("--a-file and --b-file must be provided together")
        a = load_measure_file(args.a_file, args.a_key)
        b = load_measure_file(args.b_file, args.b_key)
        metadata: dict[str, object] = {
            "kind": "file",
            "source_role": "user-provided probability array",
            "a_file": args.a_file,
            "b_file": args.b_file,
            "a_key": args.a_key,
            "b_key": args.b_key,
            "floor": args.floor,
        }
        a = normalize_density(a, floor=args.floor)
        b = normalize_density(b, floor=args.floor)
    elif args.measure == "random-mixture":
        rng = np.random.default_rng(args.seed)
        a = random_grid_mixture(
            geometry.shape,
            geometry.spacings,
            geometry.periodic_axes,
            rng,
            components=args.components,
            sigma=args.sigma,
            floor=args.floor,
        )
        b = random_grid_mixture(
            geometry.shape,
            geometry.spacings,
            geometry.periodic_axes,
            rng,
            components=args.components,
            sigma=args.sigma,
            floor=args.floor,
        )
        metadata = {
            "kind": "random-mixture",
            "source_role": "synthetic benchmark distribution",
            "seed": args.seed,
            "components": args.components,
            "sigma": args.sigma,
            "floor": args.floor,
        }
    elif args.measure == "translated-mixture":
        if args.shift is None:
            raise ValueError("--shift is required for --measure translated-mixture")
        shift = tuple(args.shift)
        if len(shift) != geometry.ndim:
            raise ValueError("--shift length must match grid dimension")
        rng = np.random.default_rng(args.seed)
        a = random_grid_mixture(
            geometry.shape,
            geometry.spacings,
            geometry.periodic_axes,
            rng,
            components=args.components,
            sigma=args.sigma,
            floor=args.floor,
        )
        b = np.roll(a, shift=shift, axis=tuple(range(geometry.ndim)))
        metadata = {
            "kind": "translated-mixture",
            "source_role": "synthetic benchmark distribution",
            "seed": args.seed,
            "components": args.components,
            "sigma": args.sigma,
            "shift": shift,
            "floor": args.floor,
        }
    elif args.measure == "gaussian-pair":
        center_a = _tuple_or_none(args.center_a)
        center_b = _tuple_or_none(args.center_b)
        if center_a is None or center_b is None:
            raise ValueError("--center-a and --center-b are required for --measure gaussian-pair")
        a = single_grid_gaussian(
            geometry.shape,
            geometry.spacings,
            geometry.periodic_axes,
            center=center_a,
            sigma=args.sigma,
            floor=args.floor,
        )
        b = single_grid_gaussian(
            geometry.shape,
            geometry.spacings,
            geometry.periodic_axes,
            center=center_b,
            sigma=args.sigma,
            floor=args.floor,
        )
        metadata = {
            "kind": "gaussian-pair",
            "source_role": "synthetic benchmark distribution",
            "center_a": center_a,
            "center_b": center_b,
            "sigma": args.sigma,
            "floor": args.floor,
        }
    elif args.measure == "delta-pair":
        if args.index_a is None or args.index_b is None:
            raise ValueError("--index-a and --index-b are required for --measure delta-pair")
        index_a = tuple(args.index_a)
        index_b = tuple(args.index_b)
        a = delta_measure(geometry.shape, index_a, floor=args.floor)
        b = delta_measure(geometry.shape, index_b, floor=args.floor)
        metadata = {
            "kind": "delta-pair",
            "source_role": "synthetic benchmark distribution",
            "index_a": index_a,
            "index_b": index_b,
            "floor": args.floor,
        }
    else:
        raise ValueError(f"unknown measure mode: {args.measure}")

    if a.shape != geometry.shape or b.shape != geometry.shape:
        raise ValueError(f"measure shapes must both equal {geometry.shape}; got {a.shape} and {b.shape}")
    return normalize_density(a), normalize_density(b), metadata


@dataclass(frozen=True)
class GridGeometry:
    shape: tuple[int, ...]
    spacings: tuple[float, ...]
    periodic_axes: tuple[int, ...]
    epsilon: float
    cost: str

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def n_points(self) -> int:
        return math.prod(self.shape)


class FFTKernel:
    """Apply exp(-C/epsilon) and C exp(-C/epsilon) without a dense matrix."""

    def __init__(self, geometry: GridGeometry, backend: ArrayBackend):
        if geometry.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if geometry.cost not in {"euclidean", "sqeuclidean"}:
            raise ValueError("cost must be 'euclidean' or 'sqeuclidean'")
        self.geometry = geometry
        self.backend = backend
        self.xp = backend.xp
        self.periodic_axes = set(geometry.periodic_axes)
        self.fft_shape = tuple(
            n if axis in self.periodic_axes else 2 * n - 1
            for axis, n in enumerate(geometry.shape)
        )
        kernel, cost_kernel = self._build_kernels()
        self._kernel_hat = self.backend.fftn(kernel)
        self._cost_kernel_hat = self.backend.fftn(cost_kernel)

    def _axis_displacements(self, axis: int) -> Array:
        n = self.geometry.shape[axis]
        spacing = self.geometry.spacings[axis]
        if axis in self.periodic_axes:
            raw = self.xp.arange(n, dtype=self.xp.float64)
            signed = self.xp.where(raw <= n / 2, raw, raw - n)
        else:
            size = 2 * n - 1
            raw = self.xp.arange(size, dtype=self.xp.float64)
            signed = self.xp.where(raw <= n - 1, raw, raw - size)
        return signed * spacing

    def _build_kernels(self) -> tuple[Array, Array]:
        grids = self.xp.meshgrid(
            *[self._axis_displacements(axis) for axis in range(self.geometry.ndim)],
            indexing="ij",
            sparse=True,
        )
        squared = self.xp.zeros(self.fft_shape, dtype=self.xp.float64)
        for grid in grids:
            squared = squared + grid**2
        if self.geometry.cost == "sqeuclidean":
            cost = squared
        else:
            cost = self.xp.sqrt(squared)
        kernel = self.xp.exp(-cost / self.geometry.epsilon)
        return kernel, cost * kernel

    def _pad(self, x: Array) -> Array:
        if x.shape != self.geometry.shape:
            raise ValueError(f"expected shape {self.geometry.shape}, got {x.shape}")
        if self.fft_shape == self.geometry.shape:
            return x
        padded = self.xp.zeros(self.fft_shape, dtype=self.xp.float64)
        slices = tuple(slice(0, n) for n in self.geometry.shape)
        padded[slices] = x
        return padded

    def _crop(self, x: Array) -> Array:
        if self.fft_shape == self.geometry.shape:
            return x
        slices = tuple(slice(0, n) for n in self.geometry.shape)
        return x[slices]

    def apply_kernel(self, x: Array) -> Array:
        transformed = self.backend.ifftn(self._kernel_hat * self.backend.fftn(self._pad(x))).real
        return self._crop(transformed)

    def apply_cost_kernel(self, x: Array) -> Array:
        transformed = self.backend.ifftn(self._cost_kernel_hat * self.backend.fftn(self._pad(x))).real
        return self._crop(transformed)


def sinkhorn_fft(
    a: Array,
    b: Array,
    kernel: FFTKernel,
    max_iter: int,
    tol: float,
    check_every: int,
    tiny: float,
) -> dict[str, object]:
    """Compute the entropic OT transport cost with implicit Sinkhorn scalings."""
    backend = kernel.backend
    xp = backend.xp
    a = backend.asarray(normalize_density(a))
    b = backend.asarray(normalize_density(b))
    if a.shape != kernel.geometry.shape or b.shape != kernel.geometry.shape:
        raise ValueError("a, b, and geometry shape must match")

    u = xp.ones_like(a)
    v = xp.ones_like(b)
    err = math.inf
    converged = False
    backend.synchronize()
    start = time.perf_counter()

    for iteration in range(1, max_iter + 1):
        kv = xp.maximum(kernel.apply_kernel(v), tiny)
        u = a / kv
        ku = xp.maximum(kernel.apply_kernel(u), tiny)
        v = b / ku

        if iteration % check_every == 0 or iteration == max_iter:
            row = u * kernel.apply_kernel(v)
            col = v * kernel.apply_kernel(u)
            err = max(
                backend.scalar(xp.sum(xp.abs(row - a))),
                backend.scalar(xp.sum(xp.abs(col - b))),
            )
            if err <= tol:
                converged = True
                break

    backend.synchronize()
    elapsed = time.perf_counter() - start
    raw_cost = backend.scalar(xp.sum(u * kernel.apply_cost_kernel(v)))
    raw_mass = backend.scalar(xp.sum(u * kernel.apply_kernel(v)))
    raw_values = {
        "cost": raw_cost,
        "mass": raw_mass,
        "marginal_l1_error": err,
    }
    nonfinite_fields = [key for key, value in raw_values.items() if not math.isfinite(float(value))]
    return {
        "cost": finite_or_none(raw_cost),
        "mass": finite_or_none(raw_mass),
        "iterations": iteration,
        "marginal_l1_error": finite_or_none(err),
        "converged": converged,
        "seconds": elapsed,
        "finite": not nonfinite_fields,
        "nonfinite_fields": nonfinite_fields,
        "u": u,
        "v": v,
    }


def dense_cost_matrix(geometry: GridGeometry) -> Array:
    """Build a dense cost matrix for small validation cases only."""
    axes = [np.arange(n, dtype=np.float64) for n in geometry.shape]
    coords = np.stack([x.reshape(-1) for x in np.meshgrid(*axes, indexing="ij")], axis=1)
    deltas = np.abs(coords[:, None, :] - coords[None, :, :])
    for axis in geometry.periodic_axes:
        deltas[:, :, axis] = np.minimum(deltas[:, :, axis], geometry.shape[axis] - deltas[:, :, axis])
    deltas *= np.asarray(geometry.spacings, dtype=np.float64)
    squared = np.sum(deltas**2, axis=2)
    if geometry.cost == "sqeuclidean":
        return squared
    return np.sqrt(squared)


def validate_against_dense(
    a: Array,
    b: Array,
    geometry: GridGeometry,
    fft_cost: float,
    max_dense_n: int,
    exact_emd: bool,
    max_emd_n: int,
) -> dict[str, float | int | str | bool]:
    """Compare FFT Sinkhorn to dense Sinkhorn and optionally exact EMD."""
    if geometry.n_points > max_dense_n:
        return {
            "status": "skipped",
            "reason": f"N={geometry.n_points} exceeds --max-dense-n={max_dense_n}",
        }

    try:
        import ot
    except ImportError:
        return {"status": "skipped", "reason": "POT package 'ot' is not installed"}

    cost_matrix = dense_cost_matrix(geometry)
    start = time.perf_counter()
    dense_cost = float(
        ot.sinkhorn2(
            a.reshape(-1),
            b.reshape(-1),
            cost_matrix,
            reg=geometry.epsilon,
            numItermax=20_000,
            stopThr=1e-12,
            method="sinkhorn",
        )
    )
    elapsed = time.perf_counter() - start
    validation: dict[str, float | int | str | bool] = {
        "status": "ok",
        "dense_entropic_sinkhorn_cost": dense_cost,
        "fft_cost": fft_cost,
        "dense_vs_fft_abs_difference": abs(dense_cost - fft_cost),
        "dense_vs_fft_relative_difference": abs(dense_cost - fft_cost) / max(abs(dense_cost), 1e-300),
        "dense_entropic_sinkhorn_seconds": elapsed,
        "dense_matrix_megabytes": cost_matrix.nbytes / 1_000_000,
    }
    if exact_emd:
        if geometry.n_points > max_emd_n:
            validation["exact_emd_status"] = "skipped"
            validation["exact_emd_reason"] = f"N={geometry.n_points} exceeds --max-emd-n={max_emd_n}"
        else:
            emd_start = time.perf_counter()
            exact_cost = float(ot.emd2(a.reshape(-1), b.reshape(-1), cost_matrix))
            emd_elapsed = time.perf_counter() - emd_start
            validation.update(
                {
                    "exact_emd_status": "ok",
                    "exact_emd_cost": exact_cost,
                    "exact_emd_seconds": emd_elapsed,
                    "entropic_bias_vs_exact": dense_cost - exact_cost,
                    "fft_bias_vs_exact": fft_cost - exact_cost,
                    "fft_relative_bias_vs_exact": (fft_cost - exact_cost) / max(abs(exact_cost), 1e-300),
                }
            )
    return validation


def geometry_from_args(args: argparse.Namespace) -> GridGeometry:
    if args.preset == "kicked-rotor":
        shape = (args.m, args.m)
        spacings = (2.0 * math.pi / args.m, 2.0 * math.pi / args.m)
        periodic_axes = (0, 1)
    elif args.preset == "bose-hubbard":
        shape = (args.L, args.L, args.L, args.L)
        spacings = tuple(1.0 / args.L for _ in shape)
        periodic_axes = (1, 3)
    else:
        if args.shape is None or args.spacings is None:
            raise ValueError("--shape and --spacings are required for --preset custom")
        shape = tuple(args.shape)
        spacings = tuple(args.spacings)
        if len(shape) != len(spacings):
            raise ValueError("--shape and --spacings must have the same length")
        periodic_axes = _parse_axes(args.periodic_axes, len(shape))

    periodic_axes = _parse_axes(periodic_axes, len(shape))
    return GridGeometry(
        shape=shape,
        spacings=spacings,
        periodic_axes=periodic_axes,
        epsilon=args.epsilon,
        cost=args.cost,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("numpy", "cupy", "auto"),
        default="numpy",
        help="numpy uses CPU; cupy uses CUDA GPU if CuPy is installed; auto prefers CuPy when available",
    )
    parser.add_argument("--device-id", type=int, help="CUDA device id for --backend cupy/auto")
    parser.add_argument(
        "--preset",
        choices=("kicked-rotor", "bose-hubbard", "custom"),
        default="kicked-rotor",
        help="geometry preset; use custom with --shape/--spacings/--periodic-axes",
    )
    parser.add_argument("--m", type=_positive_int, default=96, help="kicked-rotor grid side length")
    parser.add_argument("--L", type=_positive_int, default=10, help="Bose-Hubbard grid side length")
    parser.add_argument("--shape", nargs="+", type=_positive_int, help="custom grid shape")
    parser.add_argument("--spacings", nargs="+", type=_positive_float, help="custom positive spacing per grid axis")
    parser.add_argument("--periodic-axes", nargs="*", type=int, help="custom periodic axis indices")
    parser.add_argument("--cost", choices=("euclidean", "sqeuclidean"), default="euclidean")
    parser.add_argument("--epsilon", type=_positive_float, default=0.15, help="entropic regularization strength")
    parser.add_argument(
        "--epsilon-values",
        nargs="+",
        type=_positive_float,
        help="run an epsilon sweep; overrides --epsilon when provided",
    )
    parser.add_argument("--max-iter", type=_positive_int, default=1000)
    parser.add_argument("--tol", type=_positive_float, default=1e-9)
    parser.add_argument("--check-every", type=_positive_int, default=10)
    parser.add_argument("--tiny", type=_positive_float, default=1e-300, help="division floor for kernel products")
    parser.add_argument(
        "--measure",
        choices=("random-mixture", "translated-mixture", "gaussian-pair", "delta-pair"),
        default="random-mixture",
        help="synthetic measure type; ignored when --a-file/--b-file are provided",
    )
    parser.add_argument("--a-file", help="load first probability array from .npy, .npz, .csv, or .txt")
    parser.add_argument("--b-file", help="load second probability array from .npy, .npz, .csv, or .txt")
    parser.add_argument("--a-key", help="array key when --a-file is .npz")
    parser.add_argument("--b-key", help="array key when --b-file is .npz")
    parser.add_argument("--components", type=_positive_int, default=4, help="test mixture components")
    parser.add_argument("--sigma", type=_positive_float, default=0.35, help="test mixture width in physical units")
    parser.add_argument("--shift", nargs="+", type=int, help="integer grid shift for translated-mixture")
    parser.add_argument("--center-a", nargs="+", type=float, help="grid-index center for gaussian-pair")
    parser.add_argument("--center-b", nargs="+", type=float, help="grid-index center for gaussian-pair")
    parser.add_argument("--index-a", nargs="+", type=int, help="grid coordinate for delta-pair")
    parser.add_argument("--index-b", nargs="+", type=int, help="grid coordinate for delta-pair")
    parser.add_argument(
        "--floor",
        type=_nonnegative_float,
        default=0.0,
        help="optional non-negative density floor; default 0 preserves input measures exactly before normalization",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-dense", action="store_true")
    parser.add_argument("--max-dense-n", type=_positive_int, default=900)
    parser.add_argument(
        "--validate-exact-emd",
        action="store_true",
        help="also run exact POT EMD during dense validation, capped by --max-emd-n",
    )
    parser.add_argument("--max-emd-n", type=_positive_int, default=625)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--output-jsonl", help="append one JSON record per epsilon to this file")
    return parser


def run_single(
    args: argparse.Namespace,
    geometry: GridGeometry,
    backend: ArrayBackend,
    a: Array,
    b: Array,
    measure_metadata: dict[str, object],
) -> dict[str, object]:
    setup_start = time.perf_counter()
    kernel = FFTKernel(geometry, backend=backend)
    backend.synchronize()
    setup_seconds = time.perf_counter() - setup_start
    result = sinkhorn_fft(
        a,
        b,
        kernel,
        max_iter=args.max_iter,
        tol=args.tol,
        check_every=args.check_every,
        tiny=args.tiny,
    )

    summary: dict[str, object] = {
        "backend": backend.name,
        "device": backend.device,
        "preset": args.preset,
        "shape": geometry.shape,
        "N": geometry.n_points,
        "fft_shape": kernel.fft_shape,
        "periodic_axes": geometry.periodic_axes,
        "spacings": geometry.spacings,
        "cost": geometry.cost,
        "epsilon": geometry.epsilon,
        "distance_object": "transport cost of entropically regularized OT plan",
        "is_exact_unregularized_emd": False,
        "measure": measure_metadata,
        "setup_seconds": setup_seconds,
        "kernel_storage_megabytes": (
            kernel._kernel_hat.nbytes + kernel._cost_kernel_hat.nbytes
        )
        / 1_000_000,
        "dense_cost_matrix_megabytes_estimate_float64": geometry.n_points**2 * 8 / 1_000_000,
        "sinkhorn": {
            key: value
            for key, value in result.items()
            if key not in {"u", "v"}
        },
    }

    if args.validate_dense:
        if result["cost"] is None:
            summary["dense_validation"] = {
                "status": "skipped",
                "reason": "FFT Sinkhorn cost was non-finite",
            }
        else:
            summary["dense_validation"] = validate_against_dense(
                a,
                b,
                geometry,
                fft_cost=float(result["cost"]),
                max_dense_n=args.max_dense_n,
                exact_emd=args.validate_exact_emd,
                max_emd_n=args.max_emd_n,
            )

    return summary


def print_human_summary(summary: dict[str, object]) -> None:
    def fmt(value: object, precision: str = ".12g") -> str:
        if value is None:
            return "non-finite"
        if isinstance(value, float):
            return format(value, precision)
        return str(value)

    print(f"backend: {summary['backend']}  device: {summary['device']}")
    print(f"preset: {summary['preset']}")
    print(f"shape: {summary['shape']}  N={summary['N']}  fft_shape={summary['fft_shape']}")
    print(f"periodic_axes: {summary['periodic_axes']}")
    print(f"cost: {summary['cost']}  epsilon={summary['epsilon']}")
    print(f"measure: {summary['measure']}")
    print(f"setup_seconds: {summary['setup_seconds']:.6g}")
    print(f"kernel_storage_megabytes: {summary['kernel_storage_megabytes']:.6g}")
    print(
        "dense_cost_matrix_megabytes_estimate_float64: "
        f"{summary['dense_cost_matrix_megabytes_estimate_float64']:.6g}"
    )
    sinkhorn = summary["sinkhorn"]
    assert isinstance(sinkhorn, dict)
    print(
        "sinkhorn: "
        f"cost={fmt(sinkhorn['cost'])} "
        f"mass={fmt(sinkhorn['mass'])} "
        f"iterations={sinkhorn['iterations']} "
        f"error={fmt(sinkhorn['marginal_l1_error'], '.3g')} "
        f"seconds={fmt(sinkhorn['seconds'], '.6g')} "
        f"converged={sinkhorn['converged']}"
    )
    if "dense_validation" in summary:
        print(f"dense_validation: {summary['dense_validation']}")


def main() -> None:
    args = build_parser().parse_args()
    backend = ArrayBackend(args.backend, device_id=args.device_id)
    epsilons = tuple(args.epsilon_values) if args.epsilon_values else (args.epsilon,)
    bad_epsilons = [epsilon for epsilon in epsilons if epsilon <= 0]
    if bad_epsilons:
        raise ValueError(f"epsilon values must be positive; got {bad_epsilons}")

    base_geometry = geometry_from_args(args)
    a, b, measure_metadata = build_measures(args, base_geometry)

    summaries: list[dict[str, object]] = []
    for epsilon in epsilons:
        args.epsilon = epsilon
        geometry = geometry_from_args(args)
        summaries.append(run_single(args, geometry, backend, a, b, measure_metadata))

    if args.output_jsonl:
        with open(args.output_jsonl, "a", encoding="utf-8") as handle:
            for summary in summaries:
                handle.write(json.dumps(summary, sort_keys=True, allow_nan=False) + "\n")

    if args.json:
        payload: object = summaries[0] if len(summaries) == 1 else summaries
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        for index, summary in enumerate(summaries):
            if index:
                print()
            print_human_summary(summary)


if __name__ == "__main__":
    main()
