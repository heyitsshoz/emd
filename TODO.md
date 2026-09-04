# Research TODO: Physical Distance Bottleneck

Goal: determine whether modern OT algorithms can remove the computational
bottleneck in the physical-distance framework without changing the science
silently.

## Rules For Honest Work

- Always label exact EMD, dense entropic Sinkhorn, FFT entropic Sinkhorn, GPU
  matrix-free Sinkhorn, and any alternative distance as different objects.
- Never tune epsilon, tolerances, or smoothing after looking at the desired
  chaos result without reporting the sweep.
- Always report convergence status and marginal error.
- Always compare against exact EMD or dense Sinkhorn on small grids before
  trusting a large-grid method.
- Synthetic measures are only stress tests. The actual physics claim must use
  probability distributions produced from quantum states in the paper's chosen
  basis.
- Keep `--floor 0` for physics inputs unless there is a separately justified
  numerical experiment. A positive floor changes the distribution.
- Keep failed runs. A non-converged small-epsilon run is real evidence about the
  method's limitations.

## Phase 1: Reproduce The Numerical Bottleneck

Recommended automated route:

```bash
python3 ot_bottleneck/run_ot_experiment_suite.py --suite standard
```

On the GPU desktop:

```bash
python3 ot_bottleneck/run_ot_experiment_suite.py \
  --suite standard \
  --include-gpu
```

After that succeeds, run the heavier version:

```bash
python3 ot_bottleneck/run_ot_experiment_suite.py \
  --suite aggressive \
  --include-gpu
```

The runner writes a timestamped folder under `ot_bottleneck/results/`. Send
back `summary.jsonl`, `summary_compact.csv`, and `suite_info.json`.

Manual/debug route:

1. Run the current dense/POT validation at small grid sizes.

   ```bash
   python3 ot_bottleneck/grid_sinkhorn_fft.py \
     --preset kicked-rotor \
     --m 12 \
     --epsilon 0.2 \
     --sigma 0.4 \
     --validate-dense \
     --json
   ```

2. Increase `m` with dense validation until dense memory/time becomes annoying.
   Do not run dense validation at `m=96`; it is not useful on ordinary RAM.

   Suggested:

   ```bash
   for m in 8 12 16 20 24; do
     python3 ot_bottleneck/grid_sinkhorn_fft.py \
       --preset kicked-rotor \
       --m "$m" \
       --epsilon 0.2 \
       --sigma 0.35 \
       --validate-dense \
       --max-dense-n 900 \
       --json
   done
   ```

3. Save the outputs. The agreement with dense Sinkhorn validates the FFT
   convolution implementation, not exact EMD.

## Phase 2: CPU/GPU Scaling Benchmark

Run the same measure and epsilon values on CPU and GPU.

CPU:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --backend numpy \
  --preset kicked-rotor \
  --m 96 \
  --measure random-mixture \
  --components 8 \
  --sigma 0.25 \
  --seed 2026 \
  --epsilon-values 0.5 0.3 0.2 0.15 0.1 \
  --max-iter 5000 \
  --tol 1e-8 \
  --json \
  --output-jsonl cpu_kicked_rotor_m96_eps.jsonl
```

GPU:

```bash
python3 ot_bottleneck/grid_sinkhorn_fft.py \
  --backend cupy \
  --preset kicked-rotor \
  --m 96 \
  --measure random-mixture \
  --components 8 \
  --sigma 0.25 \
  --seed 2026 \
  --epsilon-values 0.5 0.3 0.2 0.15 0.1 \
  --max-iter 5000 \
  --tol 1e-8 \
  --json \
  --output-jsonl gpu_kicked_rotor_m96_eps.jsonl
```

Then test larger grids:

```bash
for m in 96 128 192 256 384 512; do
  python3 ot_bottleneck/grid_sinkhorn_fft.py \
    --backend cupy \
    --preset kicked-rotor \
    --m "$m" \
    --measure random-mixture \
    --components 8 \
    --sigma 0.25 \
    --seed 2026 \
    --epsilon 0.2 \
    --max-iter 5000 \
    --tol 1e-8 \
    --json \
    --output-jsonl gpu_kicked_rotor_scale.jsonl
done
```

Send back the JSONL files.

## Phase 3: Entropic Bias Study

Question: does entropic OT preserve the physical-distance growth behavior
needed for quantum Lyapunov estimates?

1. For small `m`, compare:

   - exact EMD from POT, if feasible;
   - dense entropic Sinkhorn;
   - FFT entropic Sinkhorn;
   - several epsilon values.

2. For each pair of measures, plot distance versus epsilon. Look for a stable
   plateau as epsilon decreases before numerical failure.

3. Repeat on measure families:

   ```bash
   # Nearby localized packets
   python3 ot_bottleneck/grid_sinkhorn_fft.py \
     --preset kicked-rotor --m 96 \
     --measure gaussian-pair \
     --center-a 20 30 \
     --center-b 21 31 \
     --sigma 0.20 \
     --epsilon-values 0.5 0.3 0.2 0.15 0.1 \
     --json \
     --output-jsonl gaussian_near_eps.jsonl

   # Far localized packets
   python3 ot_bottleneck/grid_sinkhorn_fft.py \
     --preset kicked-rotor --m 96 \
     --measure gaussian-pair \
     --center-a 20 30 \
     --center-b 50 70 \
     --sigma 0.20 \
     --epsilon-values 0.5 0.3 0.2 0.15 0.1 \
     --json \
     --output-jsonl gaussian_far_eps.jsonl

   # Tiny shift, useful for Lyapunov initial separation
   python3 ot_bottleneck/grid_sinkhorn_fft.py \
     --preset kicked-rotor --m 96 \
     --measure translated-mixture \
     --shift 1 1 \
     --components 8 \
     --sigma 0.25 \
     --seed 2026 \
     --epsilon-values 0.5 0.3 0.2 0.15 0.1 \
     --json \
     --output-jsonl translated_small_eps.jsonl
   ```

4. Interpret only after checking `converged=true` and small
   `marginal_l1_error`.

## Phase 4: Use Real Quantum Measures

The next real step is to generate probability arrays from evolved quantum
states.

For the kicked rotor:

1. Implement or reuse the kicked-rotor evolution.
2. Project each state into the paper's Wannier phase-space basis.
3. Save arrays shaped `(m, m)`:

   ```python
   np.save("state_a_probs.npy", probs_a.reshape(m, m))
   np.save("state_b_probs.npy", probs_b.reshape(m, m))
   ```

4. Run:

   ```bash
   python3 ot_bottleneck/grid_sinkhorn_fft.py \
     --preset kicked-rotor \
     --m 96 \
     --a-file state_a_probs.npy \
     --b-file state_b_probs.npy \
     --epsilon-values 0.5 0.3 0.2 0.15 0.1 \
     --max-iter 5000 \
     --tol 1e-8 \
     --json \
     --output-jsonl real_kicked_rotor_pair_eps.jsonl
   ```

For Lyapunov studies, do this at each time step and store:

- time/kick index;
- exact model parameters;
- initial state labels;
- epsilon;
- physical-distance estimate;
- convergence diagnostics.

## Phase 5: Bose-Hubbard Tests

The script already supports the paper's `L x L x L x L` BH phase-space metric.

CPU/GPU scaling:

```bash
for L in 8 10 12 16 20; do
  python3 ot_bottleneck/grid_sinkhorn_fft.py \
    --backend cupy \
    --preset bose-hubbard \
    --L "$L" \
    --measure random-mixture \
    --components 8 \
    --sigma 0.12 \
    --seed 2026 \
    --epsilon 0.08 \
    --max-iter 5000 \
    --tol 1e-8 \
    --json \
    --output-jsonl gpu_bh_scale.jsonl
done
```

Then replace synthetic measures with actual BH phase-space probability arrays
shaped `(L, L, L, L)`.

## Phase 6: XXZ Spin Chain

The FFT method does not directly solve XXZ because the basis is a combinatorial
fixed-magnetization subspace, not a rectangular translation-invariant grid.

Test these separately:

- exact combinatorial EMD for small chains;
- sparse/local Sinkhorn where moves beyond a distance cutoff are omitted, with
  the cutoff swept and reported;
- low-rank/Nystrom approximation of `exp(-C/epsilon)`;
- sliced Wasserstein after embedding each spin basis state as ordered positions
  of up-spins;
- physically cheaper alternatives: Hamming distance, ordered-particle mean
  displacement, local magnetization profile distances, MMD with a physically
  meaningful kernel.

Validation criterion: the cheaper distance should reproduce the qualitative
regular/chaotic separation on small systems where the original physical
distance is still computable.

## What To Tell Professor Wu After Phase 2

Do not say "we solved Wasserstein distance." A scientifically honest summary is:

> For the phase-space grid cases in the 2021 paper, the dense cost matrix is not
> necessary when using entropically regularized OT. Because the kicked-rotor
> metric is translation-invariant on a torus, Sinkhorn matrix-vector products
> can be written as FFT convolutions, reducing memory from `O(N^2)` to `O(N)`
> and per-iteration cost from dense `O(N^2)` to about `O(N log N)`. This appears
> to remove the `m=96` cost-matrix bottleneck for entropic physical distance,
> but we still need to quantify entropic bias against exact EMD on small systems
> and test whether Lyapunov/chaos diagnostics are stable across epsilon.

That is strong, but honest.
