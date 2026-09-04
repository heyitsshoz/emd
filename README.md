# Matrix-Free OT Bottleneck Notes

This folder contains a first computational probe for replacing the paper's
dense Earth Mover's Distance calculation with matrix-free entropic OT.

## What the paper computes

Wang, Wang, and Wu define the physical distance between quantum states by:

1. choosing an orthonormal physical basis `B = {|xi_i>}`;
2. converting each quantum state to a probability distribution
   `p_i(psi) = |<xi_i|psi>|^2`;
3. computing a Wasserstein distance between those two distributions using a
   basis metric `d(xi_i, xi_j)`.

For the kicked rotor, `B` is an `m x m` Wannier phase-space grid with a
periodic 2D metric. For the three-site Bose-Hubbard model, `B` is an
`L x L x L x L` quantum phase-space grid, with phase axes periodic and
particle-number axes non-periodic. These two cases have enough grid structure
to avoid storing the full pairwise cost matrix.

## What The Script Does

`grid_sinkhorn_fft.py` solves the entropically regularized OT problem

```text
min_P <C, P> + epsilon * sum_ij P_ij (log P_ij - 1)
```

with Sinkhorn scalings

```text
P_ij = u_i exp(-C_ij / epsilon) v_j.
```

The operations `K @ v` and `(C * K) @ v`, where
`K_ij = exp(-C_ij / epsilon)`, are performed as FFT convolutions. This avoids
both the dense cost matrix and the dense transport plan.

This is an approximation to exact EMD/Wasserstein-1 unless `epsilon` is taken
toward zero with adequate numerical stabilization. Every run reports
`epsilon`, convergence status, marginal error, backend, timing, and estimated
dense-matrix memory so the result cannot be confused with exact EMD.

## Install Notes

CPU runs only need NumPy and SciPy. This workspace already has them.

For GPU runs, use CuPy on the target NVIDIA machine. As of the CuPy stable
installation docs, CUDA 12.x wheels use:

```bash
pip install cupy-cuda12x
```

CUDA 13.x wheels use:

```bash
pip install cupy-cuda13x
```

CuPy warns not to install multiple CuPy packages in the same environment. Check
the official install page if the machine has an unusual CUDA/driver setup:
https://docs.cupy.dev/en/stable/install.html

## Basic CPU/GPU Commands

The easiest way to run the full progressive suite is:

```bash
python3 ot_bottleneck/run_ot_experiment_suite.py --suite standard
```

On the GPU machine:

```bash
python3 ot_bottleneck/run_ot_experiment_suite.py \
  --suite standard \
  --include-gpu
```

For the powerful desktop, after the standard suite works, use:

```bash
python3 ot_bottleneck/run_ot_experiment_suite.py \
  --suite aggressive \
  --include-gpu
```

Each suite creates a timestamped folder under `ot_bottleneck/results/` with:

- `suite_info.json`: machine/package/GPU metadata;
- `planned_runs.json`: all run labels and commands;
- `summary.jsonl`: full parsed results, one suite run per line;
- `summary_compact.csv`: one row per epsilon/result for quick inspection;
- one subfolder per labeled run with stdout, stderr, command, and timing.

CPU:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --backend numpy \
  --preset kicked-rotor \
  --m 96 \
  --epsilon 0.2 \
  --sigma 0.35
```

GPU:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --backend cupy \
  --preset kicked-rotor \
  --m 96 \
  --epsilon 0.2 \
  --sigma 0.35
```

Use `--backend auto` if you want the script to choose CuPy when available and
NumPy otherwise.

## Measure Inputs

Synthetic measures are for benchmarking only. They are not a substitute for
quantum-evolved distributions.

Random mixture:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset kicked-rotor \
  --m 96 \
  --measure random-mixture \
  --components 8 \
  --sigma 0.25 \
  --seed 123 \
  --epsilon 0.2
```

Translated mixture:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset kicked-rotor \
  --m 96 \
  --measure translated-mixture \
  --shift 1 1 \
  --components 8 \
  --sigma 0.25 \
  --seed 123 \
  --epsilon 0.2
```

Gaussian pair:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset kicked-rotor \
  --m 96 \
  --measure gaussian-pair \
  --center-a 20 30 \
  --center-b 21 31 \
  --sigma 0.20 \
  --epsilon 0.2
```

Delta pair:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset kicked-rotor \
  --m 96 \
  --measure delta-pair \
  --index-a 20 30 \
  --index-b 21 31 \
  --floor 0 \
  --epsilon 0.2
```

File input from quantum simulations:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset kicked-rotor \
  --m 96 \
  --a-file state_a_probs.npy \
  --b-file state_b_probs.npy \
  --epsilon 0.2 \
  --json
```

Supported file formats are `.npy`, `.npz`, `.csv`, and whitespace-delimited
text. For `.npz` files with multiple arrays, pass `--a-key` and `--b-key`.
Arrays must already be in the paper's chosen basis-grid shape, e.g. `(m, m)`
for kicked rotor or `(L, L, L, L)` for the BH preset. The script normalizes
non-negative arrays to unit mass and refuses negative entries.

By default `--floor` is `0`, so real input arrays are not artificially
smoothed. If you intentionally add a positive floor for numerical experiments,
report it as a change to the measure.

## Epsilon Sweeps

Use epsilon sweeps to estimate entropic bias and numerical stability:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --backend numpy \
  --preset kicked-rotor \
  --m 96 \
  --measure random-mixture \
  --seed 123 \
  --epsilon-values 0.5 0.3 0.2 0.15 0.1 \
  --max-iter 5000 \
  --tol 1e-8 \
  --json \
  --output-jsonl kicked_rotor_eps_sweep.jsonl
```

If smaller epsilon values fail to converge, that is a real result, not a bug to
hide. Record it.

## Local Validation

Small-grid runs were checked against POT's dense Sinkhorn implementation:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset kicked-rotor --m 12 --epsilon 0.2 \
  --sigma 0.4 --validate-dense --json
```

This agreed with dense Sinkhorn to about `1e-11` in the tested case.

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset bose-hubbard --L 5 --epsilon 0.12 \
  --sigma 0.15 --validate-dense --json
```

This agreed with dense Sinkhorn to about `1e-14` in the tested case.

At the user-proposed kicked-rotor scale:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset kicked-rotor --m 96 --epsilon 0.2 \
  --sigma 0.35 --max-iter 5000 --tol 1e-9 --json
```

This ran in about `0.03 s` on the local CPU, using about `0.3 MB` for FFT
kernel storage. A denser exact/dense Sinkhorn approach would need the
`9216 x 9216` cost matrix, which is about `679 MB` in float64 for the cost
matrix alone, before transport plans and temporaries.

## Honest Limitations

- This computes entropic OT, not exact unregularized Wasserstein distance.
- As `epsilon -> 0`, the result approaches exact OT in principle, but plain
  Sinkhorn can become slow or numerically unstable. Stabilized or log-domain
  methods are needed for very small epsilon.
- FFT convolution applies when the basis metric is displacement-invariant on a
  regular product grid, with periodic and/or non-periodic axes. This covers the
  kicked rotor metric and the paper's Bose-Hubbard grid metric.
- The XXZ spin-chain basis is not a simple low-dimensional rectangular grid.
  Matrix-free GPU kernels or low-rank/sparse approximations may help there,
  but the exponential Hilbert-space dimension remains the fundamental issue.

## Model Notes

Kicked rotor: strong candidate for this FFT method. The paper's metric is
translation-invariant on a periodic `m x m` grid, so the Sinkhorn matrix-vector
operation is exactly a circular convolution.

Three-site Bose-Hubbard: also a strong candidate for this implementation,
because the paper's `L^4` phase-space grid has product structure. The phase
axes are periodic and the particle-number axes are non-periodic, which the
script handles by zero-padding the non-periodic axes.

XXZ spin chain: not solved by this FFT grid trick. The paper's distance between
spin configurations is the ordered-particle transport distance on a combinatorial
subspace. We should separately test specialized combinatorial OT, sparse/local
Sinkhorn, low-rank kernels, sliced/embedded Wasserstein, and physically
motivated cheaper distances.

## Example Commands

Kicked rotor:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset kicked-rotor --m 96 --epsilon 0.2 --sigma 0.35
```

Bose-Hubbard grid:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset bose-hubbard --L 16 --epsilon 0.08 --sigma 0.12
```

Custom grid:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --preset custom --shape 64 64 --spacings 0.1 0.1 \
  --periodic-axes 0 1 --epsilon 0.2
```

## References To Read

- Wang, Wang, and Wu, "Quantum chaos and physical distance between quantum
  states", Phys. Rev. E 103, 042209 (2021).
- Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport",
  NeurIPS 2013.
- Solomon et al., "Convolutional Wasserstein distances: Efficient optimal
  transportation on geometric domains", SIGGRAPH 2015.
- Feydy et al., "Interpolating between Optimal Transport and MMD using
  Sinkhorn Divergences", AISTATS 2019.
- POT documentation for exact OT, dense Sinkhorn, and lazy/large-scale notes:
  https://pythonot.github.io/user_guide.html
- CuPy install documentation:
  https://docs.cupy.dev/en/stable/install.html
