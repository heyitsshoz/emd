#!/usr/bin/env python3
"""Run a labeled, sequential OT benchmark suite.

The suite calls grid_sinkhorn_fft.py as a subprocess for each experiment. Every
run gets its own folder containing command metadata, stdout, stderr, timing,
and parsed JSON when available. A suite-level JSONL file records one row per
run label so results can be sent back without manual copying.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_MAIN = HERE / "grid_sinkhorn_fft.py"
DEFAULT_RESULTS = HERE / "results"


@dataclass(frozen=True)
class RunSpec:
    name: str
    description: str
    args: tuple[str, ...]
    expected_heavy: bool = False


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def command_output(command: list[str], timeout: int = 10) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "command": command}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "command": command,
    }


def package_status() -> dict[str, Any]:
    script = (
        "import importlib.util, json; "
        "mods=['numpy','scipy','ot','cupy']; "
        "print(json.dumps({m: importlib.util.find_spec(m) is not None for m in mods}))"
    )
    result = command_output([sys.executable, "-c", script])
    if not result.get("ok"):
        return {"error": result}
    try:
        return json.loads(result["stdout"])
    except json.JSONDecodeError:
        return {"raw": result["stdout"], "stderr": result["stderr"]}


def system_info() -> dict[str, Any]:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "package_status": package_status(),
        "nvidia_smi": command_output(["nvidia-smi"], timeout=10)
        if shutil.which("nvidia-smi")
        else {"ok": False, "reason": "nvidia-smi not found"},
    }


def base_args(
    backend: str,
    preset: str,
    epsilon_values: list[float] | None,
    epsilon: float | None,
    max_iter: int,
    tol: float,
    device_id: int | None,
) -> list[str]:
    args = ["--backend", backend, "--preset", preset, "--max-iter", str(max_iter), "--tol", str(tol), "--json"]
    if device_id is not None:
        args.extend(["--device-id", str(device_id)])
    if epsilon_values is not None:
        args.extend(["--epsilon-values", *[str(value) for value in epsilon_values]])
    elif epsilon is not None:
        args.extend(["--epsilon", str(epsilon)])
    return args


def kicked_rotor_args(m: int) -> list[str]:
    return ["--m", str(m)]


def bh_args(level: int) -> list[str]:
    return ["--L", str(level)]


def measure_args(kind: str, args: argparse.Namespace, *, m: int | None = None) -> list[str]:
    if kind == "random":
        return [
            "--measure",
            "random-mixture",
            "--components",
            str(args.components),
            "--sigma",
            str(args.random_sigma),
            "--seed",
            str(args.seed),
        ]
    if kind == "translated":
        return [
            "--measure",
            "translated-mixture",
            "--shift",
            "1",
            "1",
            "--components",
            str(args.components),
            "--sigma",
            str(args.random_sigma),
            "--seed",
            str(args.seed),
        ]
    if kind == "gaussian_near":
        if m is None:
            raise ValueError("m is required for gaussian measures")
        return [
            "--measure",
            "gaussian-pair",
            "--center-a",
            str(0.30 * m),
            str(0.35 * m),
            "--center-b",
            str(0.30 * m + 1),
            str(0.35 * m + 1),
            "--sigma",
            str(args.gaussian_sigma),
        ]
    if kind == "gaussian_far":
        if m is None:
            raise ValueError("m is required for gaussian measures")
        return [
            "--measure",
            "gaussian-pair",
            "--center-a",
            str(0.20 * m),
            str(0.30 * m),
            "--center-b",
            str(0.55 * m),
            str(0.75 * m),
            "--sigma",
            str(args.gaussian_sigma),
        ]
    if kind == "delta":
        if m is None:
            raise ValueError("m is required for delta measures")
        return [
            "--measure",
            "delta-pair",
            "--index-a",
            str(m // 4),
            str(m // 3),
            "--index-b",
            str(m // 4 + 1),
            str(m // 3 + 1),
            "--floor",
            "0",
        ]
    raise ValueError(f"unknown measure kind: {kind}")


def build_specs(args: argparse.Namespace, backend: str) -> list[RunSpec]:
    specs: list[RunSpec] = []

    default_small_ms = {
        "smoke": [8, 12],
        "standard": [8, 12, 16, 20],
        "aggressive": [8, 12, 16, 20, 24],
    }[args.suite]
    default_scale_ms_cpu = {
        "smoke": [96],
        "standard": [96, 128, 192],
        "aggressive": [96, 128, 192, 256, 384],
    }[args.suite]
    default_scale_ms_gpu = {
        "smoke": [96],
        "standard": [96, 128, 192, 256, 384, 512],
        "aggressive": [96, 128, 192, 256, 384, 512, 768, 1024],
    }[args.suite]
    default_bh_levels_cpu = {
        "smoke": [8],
        "standard": [8, 10, 12],
        "aggressive": [8, 10, 12, 16],
    }[args.suite]
    default_bh_levels_gpu = {
        "smoke": [10],
        "standard": [10, 12, 16, 20],
        "aggressive": [10, 12, 16, 20, 24, 28, 32],
    }[args.suite]

    small_ms = args.small_ms or default_small_ms
    scale_ms_cpu = args.cpu_scale_ms or default_scale_ms_cpu
    scale_ms_gpu = args.gpu_scale_ms or default_scale_ms_gpu
    bh_levels_cpu = args.cpu_bh_levels or default_bh_levels_cpu
    bh_levels_gpu = args.gpu_bh_levels or default_bh_levels_gpu
    eps_small = args.epsilon_values
    eps_large = args.epsilon_values
    max_dense_n = args.max_dense_n
    max_emd_n = args.max_emd_n

    for m in small_ms:
        specs.append(
            RunSpec(
                name=f"{backend}_01_small_m{m}_exact_dense_fft_random",
                description="Small kicked-rotor random-mixture comparison: exact EMD if feasible, dense entropic Sinkhorn, FFT entropic Sinkhorn, epsilon sweep.",
                args=tuple(
                    base_args(backend, "kicked-rotor", eps_small, None, args.max_iter, args.tol, args.device_id)
                    + kicked_rotor_args(m)
                    + measure_args("random", args)
                    + [
                        "--validate-dense",
                        "--validate-exact-emd",
                        "--max-dense-n",
                        str(max_dense_n),
                        "--max-emd-n",
                        str(max_emd_n),
                    ]
                ),
            )
        )

    measure_ms = args.measure_ms or [96]
    for m in measure_ms:
        for measure in ["random", "translated", "gaussian_near", "gaussian_far", "delta"]:
            specs.append(
                RunSpec(
                    name=f"{backend}_02_m{m}_eps_{measure}",
                    description=f"Kicked-rotor m={m} epsilon-bias/stability sweep for {measure}.",
                    args=tuple(
                        base_args(backend, "kicked-rotor", eps_large, None, args.max_iter, args.tol, args.device_id)
                        + kicked_rotor_args(m)
                        + measure_args(measure, args, m=m)
                    ),
                )
            )

    scale_ms = scale_ms_gpu if backend == "cupy" else scale_ms_cpu
    for m in scale_ms:
        specs.append(
            RunSpec(
                name=f"{backend}_03_scale_kicked_rotor_m{m}",
                description="Progressive kicked-rotor FFT scaling at fixed epsilon.",
                args=tuple(
                    base_args(backend, "kicked-rotor", None, args.scale_epsilon, args.max_iter, args.tol, args.device_id)
                    + kicked_rotor_args(m)
                    + measure_args("random", args)
                ),
                expected_heavy=m >= 512,
            )
        )

    bh_levels = bh_levels_gpu if backend == "cupy" else bh_levels_cpu
    for level in bh_levels:
        specs.append(
            RunSpec(
                name=f"{backend}_04_scale_bose_hubbard_L{level}",
                description="Progressive Bose-Hubbard grid FFT scaling at fixed epsilon.",
                args=tuple(
                    base_args(backend, "bose-hubbard", None, args.bh_epsilon, args.max_iter, args.tol, args.device_id)
                    + bh_args(level)
                    + [
                        "--measure",
                        "random-mixture",
                        "--components",
                        str(args.components),
                        "--sigma",
                        str(args.bh_sigma),
                        "--seed",
                        str(args.seed),
                    ]
                ),
                expected_heavy=level >= 24,
            )
        )

    if args.a_file and args.b_file:
        real_preset = args.real_preset
        geometry_args = kicked_rotor_args(args.real_m) if real_preset == "kicked-rotor" else bh_args(args.real_L)
        specs.append(
            RunSpec(
                name=f"{backend}_05_real_input_eps_sweep",
                description="Real probability arrays from quantum states; this is the physics-relevant path.",
                args=tuple(
                    base_args(backend, real_preset, eps_large, None, args.max_iter, args.tol, args.device_id)
                    + geometry_args
                    + ["--a-file", args.a_file, "--b-file", args.b_file]
                    + (["--a-key", args.a_key] if args.a_key else [])
                    + (["--b-key", args.b_key] if args.b_key else [])
                ),
            )
        )

    return specs


def parse_json_payload(stdout: str) -> Any:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return None


def run_spec(
    spec: RunSpec,
    index: int,
    total: int,
    main_script: Path,
    python: str,
    results_dir: Path,
    summary_handle,
    dry_run: bool,
) -> bool:
    run_dir = results_dir / f"{index:03d}_{spec.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [python, str(main_script), *spec.args]
    metadata = {
        "index": index,
        "total": total,
        "name": spec.name,
        "description": spec.description,
        "expected_heavy": spec.expected_heavy,
        "command": command,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "command.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"[{index:03d}/{total:03d}] {spec.name}")
    print(f"  {spec.description}")
    if dry_run:
        print("  dry-run: skipped execution")
        row = {**metadata, "status": "dry-run"}
        summary_handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary_handle.flush()
        return True

    start = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - start
    stdout = completed.stdout
    stderr = completed.stderr
    payload = parse_json_payload(stdout)

    (run_dir / "stdout.json" if payload is not None else run_dir / "stdout.txt").write_text(
        json.dumps(payload, indent=2, sort_keys=True) if payload is not None else stdout,
        encoding="utf-8",
    )
    (run_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
    result = {
        **metadata,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "returncode": completed.returncode,
        "elapsed_seconds_subprocess": elapsed,
        "parsed_json": payload is not None,
        "stdout_path": str(run_dir / ("stdout.json" if payload is not None else "stdout.txt")),
        "stderr_path": str(run_dir / "stderr.txt"),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    row = {**result, "payload": payload}
    summary_handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary_handle.flush()

    if completed.returncode == 0:
        print(f"  ok in {elapsed:.3f}s")
    else:
        print(f"  FAILED rc={completed.returncode} in {elapsed:.3f}s")
        if stderr.strip():
            print(f"  stderr: {stderr.strip()[:500]}")
    return completed.returncode == 0


def compact_rows(summary_jsonl: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in summary_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        suite_row = json.loads(line)
        payload = suite_row.get("payload")
        records = payload if isinstance(payload, list) else [payload]
        for rec_index, record in enumerate(records):
            if not isinstance(record, dict):
                rows.append(
                    {
                        "suite_run_name": suite_row.get("name"),
                        "record_index": rec_index,
                        "returncode": suite_row.get("returncode"),
                        "parsed_json": suite_row.get("parsed_json"),
                    }
                )
                continue
            sinkhorn = record.get("sinkhorn", {})
            validation = record.get("dense_validation", {})
            measure = record.get("measure", {})
            rows.append(
                {
                    "suite_run_name": suite_row.get("name"),
                    "record_index": rec_index,
                    "returncode": suite_row.get("returncode"),
                    "backend": record.get("backend"),
                    "device": record.get("device"),
                    "preset": record.get("preset"),
                    "shape": "x".join(str(x) for x in record.get("shape", [])),
                    "N": record.get("N"),
                    "epsilon": record.get("epsilon"),
                    "measure_kind": measure.get("kind") if isinstance(measure, dict) else None,
                    "cost": sinkhorn.get("cost") if isinstance(sinkhorn, dict) else None,
                    "mass": sinkhorn.get("mass") if isinstance(sinkhorn, dict) else None,
                    "iterations": sinkhorn.get("iterations") if isinstance(sinkhorn, dict) else None,
                    "marginal_l1_error": sinkhorn.get("marginal_l1_error") if isinstance(sinkhorn, dict) else None,
                    "converged": sinkhorn.get("converged") if isinstance(sinkhorn, dict) else None,
                    "sinkhorn_seconds": sinkhorn.get("seconds") if isinstance(sinkhorn, dict) else None,
                    "kernel_storage_megabytes": record.get("kernel_storage_megabytes"),
                    "dense_cost_matrix_megabytes_estimate_float64": record.get(
                        "dense_cost_matrix_megabytes_estimate_float64"
                    ),
                    "dense_validation_status": validation.get("status") if isinstance(validation, dict) else None,
                    "dense_entropic_sinkhorn_cost": validation.get("dense_entropic_sinkhorn_cost")
                    if isinstance(validation, dict)
                    else None,
                    "dense_vs_fft_abs_difference": validation.get("dense_vs_fft_abs_difference")
                    if isinstance(validation, dict)
                    else None,
                    "exact_emd_status": validation.get("exact_emd_status") if isinstance(validation, dict) else None,
                    "exact_emd_cost": validation.get("exact_emd_cost") if isinstance(validation, dict) else None,
                    "fft_bias_vs_exact": validation.get("fft_bias_vs_exact") if isinstance(validation, dict) else None,
                    "fft_relative_bias_vs_exact": validation.get("fft_relative_bias_vs_exact")
                    if isinstance(validation, dict)
                    else None,
                }
            )
    return rows


def write_compact_csv(summary_jsonl: Path, csv_path: Path) -> None:
    rows = compact_rows(summary_jsonl)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_backends(args: argparse.Namespace) -> list[str]:
    if args.backends:
        return args.backends
    if args.include_gpu:
        return ["numpy", "cupy"]
    return ["numpy"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "standard", "aggressive"), default="standard")
    parser.add_argument("--backends", nargs="+", choices=("numpy", "cupy"), help="default: numpy; use --include-gpu for numpy+cupy")
    parser.add_argument("--include-gpu", action="store_true", help="run both numpy and cupy suites")
    parser.add_argument("--device-id", type=int, help="CUDA device id passed to grid_sinkhorn_fft.py")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--main-script", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-iter", type=positive_int, default=5000)
    parser.add_argument("--tol", type=positive_float, default=1e-8)
    parser.add_argument(
        "--epsilon-values",
        nargs="+",
        type=positive_float,
        default=[0.5, 0.3, 0.2, 0.15, 0.1],
        help="epsilon sweep used for validation and m=96 measure tests",
    )
    parser.add_argument("--scale-epsilon", type=positive_float, default=0.2)
    parser.add_argument("--bh-epsilon", type=positive_float, default=0.08)
    parser.add_argument("--max-dense-n", type=positive_int, default=900)
    parser.add_argument("--max-emd-n", type=positive_int, default=625)
    parser.add_argument("--small-ms", nargs="+", type=positive_int, help="override small validation m values")
    parser.add_argument("--measure-ms", nargs="+", type=positive_int, help="override kicked-rotor m values for measure epsilon sweeps")
    parser.add_argument("--cpu-scale-ms", nargs="+", type=positive_int, help="override CPU kicked-rotor scaling m values")
    parser.add_argument("--gpu-scale-ms", nargs="+", type=positive_int, help="override GPU kicked-rotor scaling m values")
    parser.add_argument("--cpu-bh-levels", nargs="+", type=positive_int, help="override CPU Bose-Hubbard L values")
    parser.add_argument("--gpu-bh-levels", nargs="+", type=positive_int, help="override GPU Bose-Hubbard L values")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--components", type=positive_int, default=8)
    parser.add_argument("--random-sigma", type=positive_float, default=0.25)
    parser.add_argument("--gaussian-sigma", type=positive_float, default=0.20)
    parser.add_argument("--bh-sigma", type=positive_float, default=0.12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--a-file", help="optional real first probability array")
    parser.add_argument("--b-file", help="optional real second probability array")
    parser.add_argument("--a-key", help="array key for .npz a-file")
    parser.add_argument("--b-key", help="array key for .npz b-file")
    parser.add_argument("--real-preset", choices=("kicked-rotor", "bose-hubbard"), default="kicked-rotor")
    parser.add_argument("--real-m", type=positive_int, default=96)
    parser.add_argument("--real-L", type=positive_int, default=10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = args.results_root / f"{timestamp}_{args.suite}"
    results_dir.mkdir(parents=True, exist_ok=False)

    info = {
        "suite": args.suite,
        "backends": choose_backends(args),
        "argv": sys.argv,
        "runner_parameters": {
            "epsilon_values": args.epsilon_values,
            "scale_epsilon": args.scale_epsilon,
            "bh_epsilon": args.bh_epsilon,
            "max_dense_n": args.max_dense_n,
            "max_emd_n": args.max_emd_n,
            "small_ms": args.small_ms,
            "measure_ms": args.measure_ms,
            "cpu_scale_ms": args.cpu_scale_ms,
            "gpu_scale_ms": args.gpu_scale_ms,
            "cpu_bh_levels": args.cpu_bh_levels,
            "gpu_bh_levels": args.gpu_bh_levels,
            "seed": args.seed,
            "components": args.components,
            "random_sigma": args.random_sigma,
            "gaussian_sigma": args.gaussian_sigma,
            "bh_sigma": args.bh_sigma,
        },
        "system": system_info(),
    }
    (results_dir / "suite_info.json").write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    specs: list[RunSpec] = []
    for backend in choose_backends(args):
        specs.extend(build_specs(args, backend))

    (results_dir / "planned_runs.json").write_text(
        json.dumps([spec.__dict__ for spec in specs], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Results folder: {results_dir}")
    print(f"Planned runs: {len(specs)}")
    print("Sequential execution starts now.\n")

    failures = 0
    summary_jsonl = results_dir / "summary.jsonl"
    with summary_jsonl.open("w", encoding="utf-8") as summary_handle:
        for index, spec in enumerate(specs, start=1):
            ok = run_spec(
                spec,
                index=index,
                total=len(specs),
                main_script=args.main_script,
                python=args.python,
                results_dir=results_dir,
                summary_handle=summary_handle,
                dry_run=args.dry_run,
            )
            if not ok:
                failures += 1
                if args.stop_on_failure:
                    break

    compact_csv = results_dir / "summary_compact.csv"
    write_compact_csv(summary_jsonl, compact_csv)

    final = {
        "results_dir": str(results_dir),
        "summary_jsonl": str(summary_jsonl),
        "summary_compact_csv": str(compact_csv),
        "planned_runs": len(specs),
        "failures": failures,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    (results_dir / "final_status.json").write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    print("\nDone.")
    print(json.dumps(final, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
