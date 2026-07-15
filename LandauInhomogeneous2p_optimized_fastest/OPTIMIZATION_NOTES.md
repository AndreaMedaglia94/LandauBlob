# Optimization notes

This document explains what was changed in `Functions/collisions_numba.py`
and `Functions/fields.py` to speed up the implementation, why each change
is safe (i.e. does not alter the numerical method), how it was validated,
and what further speedups exist if you want to go further later.

**Nothing about the method changed.** Same regularized functional, same
cell-list / random-batch scheme from the papers, same equations. Every
change below is about *how* the same sums get computed, not *what* gets
computed.

## TL;DR: what to do

1. `pip install -r requirements.txt` (adds nothing new; numba was already a
   dependency).
2. `python test_regression.py` — confirms the optimized code reproduces the
   original bit-for-bit-equivalent (to floating point round-off) on your
   machine, with your numba version.
3. `python benchmark.py` — measures the actual speedup on your machine.
   Try `python benchmark.py --threads 1 4 8 ...` to see how it scales with
   cores.
4. Just run `main_LandauDamping_2species.py` as before — it's a drop-in
   replacement, nothing else needs to change. It now also prints a
   collisions / fields / push time breakdown after each run.
5. The original implementation is preserved untouched in
   `Functions_reference/`, in case you ever want to diff against it or fall
   back.

## Why this sandbox can't show you real speedup numbers

I don't have numba available here (no network access to install it) and
the sandbox only has 1 CPU core, so I could not benchmark the actual
parallel/JIT speedup myself. What I *could* do, and did exhaustively, is
prove correctness: I hand-translated the original algorithm into plain
Python (no numba), built an independent optimized pure-Python mirror,
cross-checked the two against each other on dozens of randomized small
cases, then re-confirmed the same result using your actual shipped
`Functions/collisions_numba.py` and `Functions/fields.py` files (run
un-JIT'd through a no-op numba shim, so this is the literal code that will
run on your machine, not a paraphrase of it), including full multi-step
physical trajectories, both field solvers (`ampere`/`yee`), both field
initializations, collisions on/off, random-batch on/off, several spline
orders, several `gam` values, and several mass ratios. All of that is
`test_regression.py`, ready for you to re-run locally.

The *performance* claims below are therefore reasoned from what changed in
the code (algorithmic complexity, cache behaviour, avoided transcendental
calls) rather than measured by me. `benchmark.py` is how you get real
numbers on your machine.

## What changed, in order of expected impact

### 1. Cell lists: linked list &rarr; physically cell-sorted arrays

This is the main change, in `Functions/cell_sort.py` plus the traversal
loops in `collisions_numba.py`.

**Before:** each cell stored a `head` index into a linked list; walking a
cell's members meant following `nxt[q]` pointers to essentially random
locations in the `x`/`v1`/`v2`/`w`/... arrays. Every single neighbour
access in Steps I, II, and III was a scattered read with no relationship
to the previous one — a near-guaranteed cache miss once N is larger than a
few thousand particles.

**After:** before each Steps I-III sweep, the particle arrays are
physically re-sorted into cell order (`counting_sort_by_cell` in
`cell_sort.py`), so that all members of one cell occupy a contiguous slice
of memory (`x_s[cell_start[c]:cell_start[c+1]]`, etc.). Walking a cell's
members is now a straight-line scan over consecutive memory addresses:
cache-friendly, and prefetcher-friendly, and (per LLVM's usual behaviour
for tight loops like this) easier to auto-vectorize.

This is exactly the "cell list optimisation" the reference papers
describe, just implemented as a sorted array instead of a linked list —
both are standard cell-list variants; sorted arrays are the one normally
used in high-performance particle codes (LAMMPS, GROMACS, etc.) *because*
of this cache behaviour. Per Bailo, Carrillo & Hu (2024, Sec. 2.3.1): "the
cell list optimisation does not alter the method, as it only discards
contributions which are exactly zero" — same statement applies here; only
the data structure used to find those non-zero contributions changed.

The sort itself is parallelised too (`counting_sort_by_cell`): particles
are split into chunks (default: one per numba thread, see
`Params.sort_oversample`), each chunk computes a private histogram under
`prange` (safe — chunks own disjoint rows), a short serial prefix-sum
turns the histograms into both the global `cell_start` array and each
chunk's own write offset per cell, and then each chunk scatters its
particles into its own disjoint slice of the output, again under `prange`.

Both cell lists in the method get this treatment: the phase-space
`(x, v1, v2)` list used in Steps I-II, and the `(x, batch)` list used in
Step III.

### 2. Step III: fused self + cross species sweeps

**Before:** for each receiver species, `compute_collision_acceleration`
called `_compute_pair_accel_xbatch` twice — once against the self-species
source (`B11`/`B22`), once against the cross-species source
(`B21`/`B12`) — so 4 kernel launches total per timestep, each reloading
the receiver's `x`, `v1`, `v2`, `F1/m`, `F2/m`, cell id from memory.

**After:** `_compute_pair_accel_fused_sorted` computes both source terms
in one pass over the receiver particles: the receiver's data is loaded
once, and both the self-species and cross-species neighbour sums are
accumulated into the same `s1`/`s2` accumulators before being written out.
2 kernel launches per timestep instead of 4, and no repeated receiver-side
loads.

### 3. `gam=-2` (2V Coulomb): reciprocal instead of `pow()`

**Before:** `coeff = Bcoef * (r2 ** half_gam)` computed a full
`pow(r2, half_gam)` call for every non-zero `(p, q)` pair in Step III —
the single most-executed line in the whole method, since Step III is the
dominant `O(Nx log(Nx) (Nv^2 Nc)^2 / R)` cost per the papers'
own complexity analysis. `pow()` for a runtime-valued exponent is a
transcendental function (effectively `exp(exponent * log(x))` under the
hood), one of the more expensive elementary operations a CPU can do —
typically several times slower than a plain division.

**After:** since `gam` is fixed for an entire run and this codebase's
documented default is exactly `gam=-2.0` (the 1D-2V Coulomb kernel), the
code checks once, outside the hot loop, whether `gam == -2.0` and if so
uses `coeff = Bcoef / r2` — a single, cheap reciprocal, mathematically
identical to `r2 ** (-1.0)` up to the same last-bit floating point noise
`fastmath=True` already permits elsewhere in this codebase. Any other
`gam` still goes through the general `pow()` path, so nothing is lost for
non-Coulomb runs — it's a fast path, not a restriction.

### 4. Hoisted reciprocals

**Before:** `1/eta`, `1/eps1`, `1/eps2`, `1/(eps1*eps2)`,
`1/(eps1*eps1*eps2)`, `1/(eps1*eps2*eps2)` were recomputed (as divisions)
on every single `(p, q)` pair inside Steps I-III, even though `eta`,
`eps1`, `eps2` never change within one kernel call.

**After:** each reciprocal is computed once per kernel call and reused as
a multiplication inside the loop. Division is markedly more expensive than
multiplication on essentially all CPU microarchitectures.

### 5. Cached per-particle cell-index components

**Before:** each particle's own cell indices (its `(ix, iv1, iv2)` phase
cell, or `(ix, batch)` Step-III cell) were recomputed from `x`/`v1`/`v2`
by floor/clip arithmetic independently inside cell-list construction,
Step I, and Step II (three times over, for the same particle, the same
answer, every timestep).

**After:** `_build_phase_cell_components` / `_build_xbatch_components`
compute each particle's cell indices once, and every later pass reads the
cached integer instead of recomputing it.

### 6. Charge/current deposition: serial &rarr; parallel (chunked, privatized)

**Before:** `_deposit_one_species` was a scatter-add into shared
`rho`/`J1`/`J2` arrays. A naive `prange` there would race, since several
particles near a cell boundary can write to the same grid point at the
same time — so the original code ran this fully serially, on every single
timestep, unconditionally (unlike the collision step, which can be
disabled).

**After:** `_deposit_one_species_chunked` splits particles into chunks,
each accumulating into its own private `(nchunks, Nx)` row under `prange`
(no race — chunks own disjoint rows), followed by a cheap serial reduction
(`O(nchunks*Nx)`, negligible next to the `O(N)` deposition work itself).
This is the same idea as the parallel cell-sort's histogram step. Grid
interpolation (`interpolate_primal_to_particles`) was already `prange`'d
in the original and is untouched except for the same reciprocal-hoisting
as above.

## Validated equivalence

Every change above was validated three ways before being adopted:

1. **Pure-Python mirrors.** Both the original algorithm (line-for-line,
   `@njit`/`prange` stripped) and the optimized algorithm were implemented
   as plain Python and cross-checked on dozens of randomized small cases
   (different N, spline order, `gam`, random-batch count, chunk count,
   including edge cases like more sort-chunks than particles).
2. **The actual shipped files.** `Functions/collisions_numba.py` and
   `Functions/fields.py` (the real files, not a paraphrase) were imported
   through a no-op numba shim (`njit` &rarr; identity, `prange` &rarr; `range`)
   and cross-checked against `Functions_reference/` the same way, plus
   full multi-step physical trajectories (position, velocity, fields) for
   both `ampere`/`analytic_landau` and `yee`/`poisson_particles`, with
   collisions on and off, and with/without random batching.
3. **`test_regression.py`**, included here, re-runs the same checks with
   your real, installed numba — please run it once before trusting this
   for real work.

All observed differences were at the `1e-15`..`1e-16` level (absolute),
consistent with floating-point summation-order noise from reordering
which particle gets added to a sum first — the same kind of noise you'd
see from changing numpy's reduction order — not an algorithmic
discrepancy.

## Tuning

- `Params.sort_oversample` (default `1`): the pre-sweep particle sort is
  parallelised over `sort_oversample * numba.get_num_threads()` chunks.
  The default uses exactly one chunk per available thread. If
  `benchmark.py` or the per-phase timing breakdown that
  `main_LandauDamping_2species.py` now prints shows the sort itself taking
  a meaningful share of the collision time (as opposed to the pairwise
  sums), try raising this to 2-4 for better load balance, at the cost of a
  temporary `(nchunks, ncells)` `int64` histogram buffer. For the
  phase-space cell list, `ncells` is roughly `N / Nc`, so this buffer is
  not free at large N — start from the default and only raise it if
  profiling justifies it.
- `numba.set_num_threads(n)` / the `NUMBA_NUM_THREADS` environment
  variable control how many threads all the `parallel=True` kernels use.
  `benchmark.py --threads 1 4 8 ...` sweeps this for you.

## What's *not* changed, and why

- **The random-batch scheme itself.** Steps I-II still always use the full
  compact cell list; Step III still uses the same batch-matched
  cross-species pairing (`batch b of species 1` &harr; `batch b of species
  2`) with the same `R(N-1)/(N-R)` / `R` unbiasing scales from the paper.
  Only *how* a batch's members are found changed (sorted array vs. linked
  list) — not which members are found, or how they're weighted.
- **No treecode.** Consistent with the direction already set for the
  related homogeneous-equation codebase (batched dense per-cell
  computation, explicitly not a treecode, to keep the implementation
  simple to later extend to full spatial inhomogeneity), and consistent
  with your instruction not to change the method's complexity structure.
- **`error_model='numpy'`** (which would let integer/float division-by-
  zero silently produce `inf`/`nan` instead of raising, saving a branch
  per division) was deliberately *not* applied. The branch is essentially
  free (trivially predicted), and for a research code, keeping Python's
  div-by-zero exceptions active seemed more valuable than the marginal
  speedup — happy to add it as an opt-in if you disagree.

## Further speedups that exist, but weren't implemented here

These are real opportunities, described honestly rather than attempted
half-validated:

### Newton's-third-law pairwise halving (up to ~2x further on Step III)

The interaction matrix `A(v_p - v_q)` is an even function of its argument
(`A(z) = A(-z)`, since it only involves `|z|^2` and `z (x) z`), and the
spatial spline `psi` is symmetric. That means the pair `(p, q)` and the
pair `(q, p)` currently get evaluated completely independently (once from
`p`'s perspective as receiver, once from `q`'s), even though the expensive
part of the inner loop — the distance, the spline evaluations, the `A`
matrix — is literally the same computation both times, just with a sign
flip and a different particle weight on each side. Visiting each *unordered*
pair once and scattering the (oppositely-signed, differently-weighted)
contribution to both `p` and `q` would roughly halve Step III's arithmetic,
on top of everything above.

This wasn't implemented because doing it safely under `parallel=True`
needs either atomic scatter-adds (not straightforwardly available for CPU
targets in numba), or thread-private full-size accumulator arrays
reduced at the end (real memory cost: `O(nthreads * N)` per accelerated
array), or a "half-shell" neighbour stencil with careful bookkeeping for
the cross-species case, where a single pair evaluation must update two
differently-sized output arrays (species 1's and species 2's). All three
are implementable, but none of them got the same from-scratch, multi-angle
validation as the changes above, and getting this wrong would silently
corrupt physics (not crash) — the wrong place to cut corners in a
research code. If you want this pursued, it's a well-scoped follow-up.

### Caching Step I's neighbour list for reuse in Step II (up to ~2x on Steps I+II)

Steps I and II traverse the exact same phase-space neighbour cells for the
exact same particle in the exact same order — Step II is, in effect,
redoing Step I's neighbour search and spline evaluations from scratch
(it needs `ft` fully known first, which is why they can't just be fused
into one pass). Storing each particle's `(neighbour index, spline
products)` list during Step I and replaying it in Step II would remove
that duplicated work. Not implemented because the neighbour list has
variable length per particle, which means either a two-pass
counting-then-filling allocation (implementable, similar in spirit to the
counting sort already added) or a real risk of memory blow-up at large N
with many neighbours per cell — worth doing if profiling shows Steps I-II
are a meaningful fraction of total time for your problem sizes (Step III
usually dominates per the papers' own complexity analysis, so check first).

### GPU

Everything here is an embarrassingly-parallel per-particle (or per-cell)
loop, which is exactly the shape of problem GPUs are good at. `numba.cuda`
could host these same kernels with a different memory model (particle
arrays live on-device, cell lists as sorted arrays translate directly).
This is a substantially larger engineering effort (different debugging
story, explicit host/device transfers, a CUDA-capable machine for
development) so it's mentioned here as a longer-horizon option rather
than something attempted now.

## v2 spline-order-3 update

This version is focused on the production setting `spline_order=3`, `gam=-2`, and `random_batches=32`. It keeps the same C-PIC/Landau discretisation and the same random-batch/cell-list method. The changes are implementation-only:

- Use the physically cell-sorted contiguous-cell layout for the collision operator. This is the better path for quadratic/cubic splines because the higher-order compact support makes linked-list traversal expensive.
- Add cubic-B-spline-specialized kernels for Steps I-II and for the Step III Coulomb pair sweep. These remove the runtime `spline_order` branch and avoid returning unused spline derivatives in the Step III spatial spline evaluation.
- Keep the `gam=-2` reciprocal path in Step III, using `1/r2` instead of a generic power.
- Reuse random-batch arrays in the main time loop via `BatchIdGenerator`, avoiding per-step permutation/batch allocation.

Validation against the previous sorted implementation was run on deterministic small `spline_order=3` cases. Collision outputs matched to floating-point roundoff; maximum absolute differences were about `5e-15` in the acceleration/F arrays, and field deposition/interpolation matched exactly for the tested case.

On the provided production-like parameters (`Nx=64`, `Nv1=Nv2=32x32`, `Nc1=Nc2=2`, `m2=2`, `gam=-2`, `spline_order=3`, `random_batches=32`, `NUMBA_NUM_THREADS=8`) one-step wall-clock estimates in the container were approximately:

| code | one full step | T=20, dt=0.02 estimate |
|---|---:|---:|
| previous optimized code | 1.09 s | 1089 s |
| Claude code | 0.87 s | 869 s |
| this v2 spline-3 code | 0.64 s | 635 s |

The exact wall time is hardware-dependent and can vary with CPU scheduling, but the relative result is stable: for third-order splines the sorted/cubic-specialized path is preferable to the earlier linear-spline-focused linked-list implementation.

