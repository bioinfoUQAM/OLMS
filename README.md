# 7D Local-Minima Grid Search (Numba-Optimized)

A fast, **Numba-parallelized** implementation that evaluates a 7‑dimensional transformation grid and finds **all local minima** :

* Parameter ranges (tx, ty, rot, sx, sy, skx, sky)
* Distance metric (sum of Euclidean distances over rows)
* Neighborhood rule (3^7−1 neighbors in $-1,0,1$ offsets; **strictly smaller** neighbor disqualifies a center; ties are allowed)

This version reduce runtime by pushing hot loops into compiled, parallel kernels.

---

## Features

* 🔧 **Drop-in** `find_local_minima(floating, standard_stall, Nr=7)` API
* ⚡ **Parallel CPU** traversal of the full 7D grid (no Python loops)
* 🔁 **Local-minima mask** computed in parallel using the exact neighbor rule
* 🧮 **Precomputed trig** per rotation bin to avoid repeated `sin/cos`
* 💾 Modest memory footprint (for `Nr=7`: `7^7 = 823,543` doubles ≈ 6.6 MB)

---

## Requirements

* Python ≥ 3.9 (3.10/3.11 OK)
* `numpy`
* `numba`

```bash
pip install numpy numba
```

> **Note:** The very first run on a machine will JIT‑compile kernels and can take a few seconds. Subsequent runs are much faster.

---

## Quick Start

```python
import numpy as np
from your_module import find_local_minima  # or paste the script and call directly

# Example data (6×2 points)
data_2 = np.array([
    [-0.401341, -0.509649], [-0.425543, -0.182018],
    [-0.401341,  0.647982], [ 0.373106,  0.662544],
    [ 0.427559, -0.116491], [ 0.427559, -0.502368]
], dtype=np.float64)

standard_stall = np.array([
    [-0.416547, -0.536824], [-0.416547, -0.092251],
    [-0.416547,  0.629075], [ 0.416547,  0.629075],
    [ 0.416547, -0.092251], [ 0.416547, -0.536824]
], dtype=np.float64)

# Search entire 7D grid with Nr=7 bins/dimension
minima = find_local_minima(data_2, standard_stall, Nr=7)

# Print results (1‑based indices for MATLAB comparison)
for i, (index, value) in enumerate(minima, 1):
    matlab_index = tuple(dim + 1 for dim in index)
    print(f"Minima {i}: Index {matlab_index}, Value = {value:.6f}")
```

**Return format**: `List[Tuple[index_tuple, distance_value]]`, where `index_tuple` is a 7‑tuple of 0‑based indices for `(tx, ty, rot, sx, sy, skx, sky)` bins.

---

## Parameters & Grids

* `Nr` (default 7): number of equally spaced bins per dimension
* Parameter ranges (can be customized):

  * `tx, ty, rot, skx, sky` ∈ `linspace(-0.5, 0.5, Nr)`
  * `sx, sy` ∈ `linspace(0.5, 1.5, Nr)`

You can change these ranges inside `find_local_minima` if needed. Increasing `Nr` raises compute as `Nr^7`; the implementation scales well on multi-core CPUs, but total work still grows exponentially.

---

## What Makes It Fast

1. **Numba-compiled loops:** Hot loops are annotated with `@njit(parallel=True)`, turning Python loops into multi-threaded native code.
2. **Flat traversal:** We iterate over the grid as a single flat index (`0..Nr^7-1`) and map to 7D indices using a Numba‑friendly `_unravel_index`, which improves cache locality.
3. **Precomputed trig:** `cos(rot)`/`sin(rot)` are computed once per rotation bin and reused in the kernel.
4. **Scalar transform math:** The 7‑DoF transform is expanded into scalar coefficients and applied directly, avoiding per‑cell matrix allocations or Python overhead.
5. **Two parallel passes:** One pass to fill the distance grid, a second pass to produce a boolean minima mask using the exact neighbor rule.

---

## Exactness & Tie Rule

* **Distance metric:** Sum of Euclidean distances over point rows (includes the `sqrt`).
* **Local-min rule:** A center is a local minimum if **no in-bounds neighbor has a strictly smaller value**. Equal neighbors are allowed (plateau minima). To change this, flip the condition in the minima kernel from `if v < center_val` to `if v <= center_val`.

---

## Determinism & Precision

* Kernels are compiled with `fastmath=True` for speed. If you need stricter numerical determinism (e.g., bit‑exact reproducibility across environments), set `fastmath=False` on `@njit` decorators.
* Dtype is `float64` throughout. You can switch to `float32` for lower memory, but verify it doesn’t perturb marginal plateaus.

---

## Environment & Threading

Parallel performance can depend on BLAS/OpenMP settings. For best results, try:

```bash
# Use all physical cores for Numba, but keep MKL/BLAS single‑threaded
export NUMBA_NUM_THREADS=$(nproc)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

On Windows (PowerShell):

```powershell
$env:NUMBA_NUM_THREADS = 8   # set to your core count
$env:OMP_NUM_THREADS   = 1
$env:MKL_NUM_THREADS   = 1
```

---

## Troubleshooting

* **“It runs then seems to hang” on first call**: That’s usually JIT compilation. Give it a moment the first time; subsequent calls are fast. If it persists, temporarily change `@njit(parallel=True, ...)` to `parallel=False` to rule out a thread config issue.
* **No output / freeze with parallel=True**: Ensure per‑iteration temporaries are **allocated inside** the `prange` loop (the provided code does this). Shared buffers can cause stalls.
* **Different results vs MATLAB**: Ensure identical ranges, bin counts, and tie rule. If you changed `fastmath`, tiny numeric differences may reorder equal plateaus.

---

## API Reference

### `find_local_minima(floating, standard_stall, Nr=7) -> List[Tuple[Tuple[int,...], float]]`

**Inputs**

* `floating`: `(N, 2)` array of source points
* `standard_stall`: `(N, 2)` array of target/reference points
* `Nr`: integer bins per dimension (defaults to 7)

**Outputs**

* A list of `(index_tuple, score)` where `index_tuple` is a 7‑tuple of 0‑based indices `(ix_tx, ix_ty, ix_rot, ix_sx, ix_sy, ix_skx, ix_sky)` and `score` is the sum of Euclidean distances after applying the corresponding transform to `floating`.

**Notes**

* If you want the *global* minimum only, you can do: `idx = np.unravel_index(np.argmin(dist_grid), dist_grid.shape)` inside the code after filling `dist_grid`. The current API returns **all** local minima by design.

---

## Internals (Functions)

* `_apply_7dof(points, tx, ty, rot_c, rot_s, sx, sy, skx, sky, out)`

  * Applies the 7‑DoF transform using scalar coefficients; writes into `out` buffer.
* `_stall_distance_sum(A, B)`

  * Sum of row‑wise Euclidean distances between point sets.
* `_unravel_index(idx, shape, out)`

  * Numba‑friendly version of `np.unravel_index`.
* `_compute_dist_grid(...)` *(parallel)*

  * Fills the 7D grid of distances; one transformed evaluation per cell.
* `_compute_minima_mask(dist_grid, offsets)` *(parallel)*

  * Produces a boolean mask of local minima according to the neighbor rule.

---

## Extending

* **Coarse→Fine Search:** Add a multi‑resolution schedule (e.g., `Nr=5 → 9 → 13`) to zoom around promising regions. This will not enumerate *all* local minima but can find good minima much faster.
* **Custom Distance:** Replace `_stall_distance_sum` with your preferred metric (e.g., weighted distances) to bias certain points.
* **Different Neighborhoods:** Change `offsets` to evaluate a different stencil (e.g., king’s move only, or radius‑2 neighborhoods).

---

## License

Well-e.
