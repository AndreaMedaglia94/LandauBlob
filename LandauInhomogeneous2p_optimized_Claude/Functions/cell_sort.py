"""
Parallel "counting sort by cell id" utility.

Both collision cell lists in collisions_numba.py (the phase-space (x,v1,v2)
list used in Steps I-II, and the x-batch (x,batch) list used in Step III)
were originally built as linked lists: each cell stores a `head` index, and
`nxt[q]` chains to the next particle in the same cell. Walking a cell's
members means following `nxt` pointers to essentially random locations in
the (x, v1, v2, w, ...) arrays -- a cache miss on almost every step, since
particle order has nothing to do with cell membership.

This module replaces that with the standard particle-in-cell technique of
*physically sorting* the particle arrays into cell order (a counting sort),
so that every "walk this cell's members" loop becomes a scan over a
contiguous slice of memory instead of a pointer chase. This does not
change which particles are found or what is summed over them -- it only
changes how they are located, which is exactly the kind of "cell list
optimisation" the reference papers describe as method-preserving (Bailo,
Carrillo & Hu 2024, Sec. 2.3.1: "the cell list optimisation does not alter
the method, as it only discards contributions which are exactly zero").

The sort itself is also parallelised (`counting_sort_by_cell`), using the
standard two-pass parallel-histogram technique: particles are split into
`nchunks` contiguous chunks; each chunk computes a private histogram
(Phase 1, safe under prange because chunks own disjoint rows); a serial
prefix sum turns those histograms into both a global `cell_start` array
and each chunk's starting write offset per cell (Phase 2); then each chunk
scatters its own particles directly into its private, non-overlapping
slice of the output (Phase 3, safe under prange for the same reason as
Phase 1). With nchunks=1 this degenerates to an ordinary serial counting
sort.
"""
import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def counting_sort_by_cell(cell_of, ncells, nchunks):
    """
    Parallel counting sort of particle indices by cell_of[p] in [0, ncells).

    Returns
    -------
    sort_idx : int64[N]
        sort_idx[i] is the original particle index now placed at sorted
        position i. Gather any per-particle array with arr[sort_idx] to
        get it in cell-sorted order; scatter a sorted-order result back
        with out[sort_idx] = result_sorted.
    cell_start : int64[ncells + 1]
        Particles of cell c occupy sort_idx[cell_start[c]:cell_start[c+1]].
    """
    N = cell_of.shape[0]
    nchunks = max(1, min(nchunks, max(1, N)))
    chunk_size = (N + nchunks - 1) // nchunks

    # ---- Phase 1: per-chunk histogram (parallel over chunks) ----
    local_counts = np.zeros((nchunks, ncells), dtype=np.int64)
    for c in prange(nchunks):
        start = c * chunk_size
        end = start + chunk_size
        if end > N:
            end = N
        for p in range(start, end):
            local_counts[c, cell_of[p]] += 1

    # ---- Phase 2: prefix sums (serial, O(nchunks*ncells)) ----
    total_counts = np.zeros(ncells, dtype=np.int64)
    for c in range(nchunks):
        for cell in range(ncells):
            total_counts[cell] += local_counts[c, cell]

    cell_start = np.zeros(ncells + 1, dtype=np.int64)
    for cell in range(ncells):
        cell_start[cell + 1] = cell_start[cell] + total_counts[cell]

    chunk_offset = np.zeros((nchunks, ncells), dtype=np.int64)
    for cell in range(ncells):
        running = cell_start[cell]
        for c in range(nchunks):
            chunk_offset[c, cell] = running
            running += local_counts[c, cell]

    # ---- Phase 3: scatter (parallel over chunks; disjoint write ranges) ----
    sort_idx = np.empty(N, dtype=np.int64)
    for c in prange(nchunks):
        start = c * chunk_size
        end = start + chunk_size
        if end > N:
            end = N
        cursor = chunk_offset[c].copy()
        for p in range(start, end):
            cell = cell_of[p]
            pos = cursor[cell]
            sort_idx[pos] = p
            cursor[cell] += 1

    return sort_idx, cell_start


def default_nchunks(n_particles, oversample=1):
    """Sensible default chunk count for counting_sort_by_cell: one chunk
    per available thread (times an optional oversampling factor for better
    load balance), capped so tiny particle counts don't create more chunks
    than particles."""
    import numba as _nb
    nthreads = _nb.get_num_threads()
    return max(1, min(n_particles, int(oversample) * nthreads))
