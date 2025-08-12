import numpy as np
import time
import numba as nb


"""This script finds best local minima in a 7D space defined by a set of transformation parameters.
The transformations include translation (2D), rotation (1D), scale (2D), and skew (2D).
The script :
    push the heavy loops into Numba → compiled, parallel C loops, which is much faster than pure Python.
    precompute trig for rotation bins.
    avoid building matrices & temporary arrays inside Python.
    It uses parallel processing to efficiently search through the parameter space.
"""

# Function to calculate the distance between two sets of points
# --------- math helpers compiled with numba ---------
@nb.njit(fastmath=True, cache=True)
def _stall_distance_sum(points1, points2):
    s = 0.0
    for i in range(points1.shape[0]):
        dx = points1[i,0] - points2[i,0]
        dy = points1[i,1] - points2[i,1]
        s += (dx*dx + dy*dy)**0.5
    return s
    #return np.sum(np.sqrt(np.sum((points1 - points2) ** 2, axis=1)))

# Function to apply a 7D transformation to a set of points
# this will avoid allocating a new array each call
@nb.njit(fastmath=True, cache=True)
def _apply_7dof(points, tx, ty, rot_c, rot_s, sx, sy, skx, sky, out):
    # transform:
    # [[ sx*( c + sky*s),  sx*( skx*c + s),  tx],
    #  [ sy*(-s + sky*c),  sy*(-skx*s + c), ty],
    #  [ 0,                0,               1 ]]
    a00 = sx * (rot_c + sky * rot_s)
    a01 = sx * (skx * rot_c + rot_s)
    a10 = sy * (-rot_s + sky * rot_c)
    a11 = sy * (-skx * rot_s + rot_c)
    a02 = tx
    a12 = ty

    n = points.shape[0]
    for i in range(n):
        x = points[i,0]; y = points[i,1]
        out[i,0] = x*a00 + y*a01 + a02
        out[i,1] = x*a10 + y*a11 + a12

# Function to unravel a flat index into a multi-dimensional index
# This is used to convert a flat index into a multi-dimensional index (7D) for the distance grid.
# same as np.unravel_index but not that compatible with njit kernels
@nb.njit(cache=True)
def _unravel_index(idx, shape, out):
    for i in range(len(shape)-1, -1, -1):
        out[i] = idx % shape[i]
        idx //= shape[i]

# Compute the distance grid in parallel
@nb.njit(parallel=True, fastmath=True, cache=True)
def _compute_dist_grid(dist_grid, points, target,
                       tx_vals, ty_vals, rot_cos, rot_sin, sx_vals, sy_vals, skx_vals, sky_vals):
    Nr = tx_vals.shape[0]
    shape = (Nr, Nr, Nr, Nr, Nr, Nr, Nr)
    total = 1
    for s in shape:
        total *= s

    # preallocate the distance grid
    for flat in nb.prange(total):
        # per-iteration locals (avoid shared state!)
        idx_vec = np.empty(7, dtype=np.int64)
        _unravel_index(flat, shape, idx_vec)

        ix0, ix1, ix2, ix3, ix4, ix5, ix6 = (
            idx_vec[0], idx_vec[1], idx_vec[2], idx_vec[3],
            idx_vec[4], idx_vec[5], idx_vec[6]
        )
        tx  = tx_vals[ix0]
        ty  = ty_vals[ix1]
        c   = rot_cos[ix2]
        s   = rot_sin[ix2]
        sx  = sx_vals[ix3]
        sy  = sy_vals[ix4]
        skx = skx_vals[ix5]
        sky = sky_vals[ix6]

        # thread-local buffer
        # apply the transformation and compute the distance
        tmp = np.empty_like(points)
        _apply_7dof(points, tx, ty, c, s, sx, sy, skx, sky, tmp)
        d = _stall_distance_sum(tmp, target)
        # store the distance in the grid
        dist_grid[ix0, ix1, ix2, ix3, ix4, ix5, ix6] = d

# Function to transform points using the given parameters
# This function applies translation, rotation, scaling, and skewing to the input points.
@nb.njit(parallel=True, cache=True)
def _compute_minima_mask(dist_grid, offsets):
    Nr = dist_grid.shape[0]
    shape = dist_grid.shape
    total = 1
    for s in shape:
        total *= s

    mask = np.ones(shape, dtype=np.uint8)  # 1 = is local min

    for flat in nb.prange(total):
        # per-iteration locals
        idx_vec = np.empty(7, dtype=np.int64)
        _unravel_index(flat, shape, idx_vec)
        center_val = dist_grid[idx_vec[0], idx_vec[1], idx_vec[2],
                               idx_vec[3], idx_vec[4], idx_vec[5], idx_vec[6]]

        for k in range(offsets.shape[0]):
            # check neighbors
            # if any neighbor is out of bounds, skip this neighbor
            oob = False
            neigh0 = idx_vec[0] + offsets[k,0]
            neigh1 = idx_vec[1] + offsets[k,1]
            neigh2 = idx_vec[2] + offsets[k,2]
            neigh3 = idx_vec[3] + offsets[k,3]
            neigh4 = idx_vec[4] + offsets[k,4]
            neigh5 = idx_vec[5] + offsets[k,5]
            neigh6 = idx_vec[6] + offsets[k,6]
            if (neigh0 < 0 or neigh0 >= Nr or
                neigh1 < 0 or neigh1 >= Nr or
                neigh2 < 0 or neigh2 >= Nr or
                neigh3 < 0 or neigh3 >= Nr or
                neigh4 < 0 or neigh4 >= Nr or
                neigh5 < 0 or neigh5 >= Nr or
                neigh6 < 0 or neigh6 >= Nr):
                oob = True
            if oob:
                continue
            # get the neighbor value
            # if the neighbor value is less than the center value, mark the center as not a local minimum
            v = dist_grid[neigh0, neigh1, neigh2, neigh3, neigh4, neigh5, neigh6]
            if v < center_val:
                mask[idx_vec[0], idx_vec[1], idx_vec[2],
                     idx_vec[3], idx_vec[4], idx_vec[5], idx_vec[6]] = 0
                break
    return mask

# Function to find local minima in the parameter space
# This function iterates through the parameter space and checks for local minima.
def find_local_minima(floating, standard_stall, Nr=7):
    Nt = 7

    # ensure contiguous and consistent dtype
    # access memory directly (C order => faster)
    floating = np.ascontiguousarray(floating, dtype=np.float64)
    standard_stall = np.ascontiguousarray(standard_stall, dtype=np.float64)

    # parameter ranges for the transformation coefficients
    tx_vals  = np.linspace(-0.5, 0.5, Nr)
    ty_vals  = np.linspace(-0.5, 0.5, Nr)
    rot_vals = np.linspace(-0.5, 0.5, Nr)
    sx_vals  = np.linspace(0.5, 1.5, Nr)
    sy_vals  = np.linspace(0.5, 1.5, Nr)
    skx_vals = np.linspace(-0.5, 0.5, Nr)
    sky_vals = np.linspace(-0.5, 0.5, Nr)

    # precompute all possible trig for rotations
    rot_cos = np.cos(rot_vals)
    rot_sin = np.sin(rot_vals)

    # build offsets [-1,0,1]^7 \ {(0,...,0)}
    # create the iffset grid
    grid = np.array([-1, 0, 1], dtype=np.int8)
    offsets = np.empty((3**Nt - 1, Nt), dtype=np.int8)
    idx = 0
    for a in grid:
        for b in grid:
            for c in grid:
                for d in grid:
                    for e in grid:
                        for f in grid:
                            for g in grid:
                                # skip the center
                                if (a|b|c|d|e|f|g) != 0:
                                    offsets[idx, 0] = a
                                    offsets[idx, 1] = b
                                    offsets[idx, 2] = c
                                    offsets[idx, 3] = d
                                    offsets[idx, 4] = e
                                    offsets[idx, 5] = f
                                    offsets[idx, 6] = g
                                    idx += 1

    # create the distance grid 7D
    shape = (Nr,)*Nt
    dist_grid = np.empty(shape, dtype=np.float64)

    print("Computing distance grid (parallel)…")
    t0 = time.time()
    # compute the distance grid in parallel
    _compute_dist_grid(dist_grid, floating, standard_stall,
                       tx_vals, ty_vals, rot_cos, rot_sin,
                       sx_vals, sy_vals, skx_vals, sky_vals)
    print(f"Distance grid computed in {time.time() - t0:.2f}s")

    print("Searching for local minima (parallel)…")
    t1 = time.time()
    mask = _compute_minima_mask(dist_grid, offsets)
    print(f"Minima search completed in {time.time() - t1:.2f}s")

    # collect minima
    mins = []
    iterate = np.nditer(mask, flags=['multi_index'])
    # iterating through the mask to find local minima (when == 1)
    for m in iterate:
        if m.item() == 1:
            idx7 = iterate.multi_index
            mins.append((idx7, float(dist_grid[idx7])))

    print(f"Found {len(mins)} local minima")
    return mins


# ---------------- example usage ----------------
if __name__ == "__main__":
    t0 = time.time()
    
    data_1 = np.array([
        [-0.422727, -0.517149], [-0.4166, -0.146654],
        [-0.31245, 0.640647], [0.392094, 0.710115],
        [0.367588, -0.16981], [0.392094, -0.517149]
    ], dtype=np.float64)
    data_2 = np.array([
        [-0.401341, -0.509649], [-0.425543, -0.182018],
        [-0.401341, 0.647982], [0.373106, 0.662544],
        [0.427559, -0.116491], [0.427559, -0.502368]
    ], dtype=np.float64)
    data_3 = np.array([
        [-0.391437, -0.49987], [-0.391437, -0.144258],
        [-0.445222, 0.573676], [0.319723, 0.593805],
        [0.439246, -0.104], [0.469126, -0.419354]
    ], dtype=np.float64)
    data_4 = np.array([
        [-0.280136, -0.535159], [-0.31197, -0.172383],
        [-0.439304, 0.630355], [0.235569, 0.645793],
        [0.38837, -0.118352], [0.407471, -0.450254]
    ], dtype=np.float64) 
    data_5 = np.array([
        [-0.415778, -0.503298], [-0.409839, -0.173594],
        [-0.332623, 0.658159], [0.397959, 0.695626],
        [0.38014, -0.143621], [0.38014, -0.533271]
    ], dtype=np.float64)
    data_6 = np.array([
        [-0.421186, -0.500212], [-0.421186, -0.160199],
        [-0.404487, 0.572138], [0.369234, 0.644064],
        [0.424897, -0.114428], [0.452729, -0.441363]
    ], dtype=np.float64)
    data_7 = np.array([
        [-0.342202, -0.51443], [-0.371367, -0.172237], 
        [-0.429697, 0.62165], [0.293594, 0.628494],
        [0.404421, -0.117486], [0.445252, -0.445991]
    ], dtype=np.float64)
    data_8 = np.array([
        [-0.26059, -0.559601], [-0.304021, -0.189107],
        [-0.421907, 0.652225], [0.248181, 0.629069],
        [0.335044, -0.11192], [0.403294, -0.420666]
    ], dtype=np.float64)
    data_9 = np.array([
        [-0.416038, -0.519159], [-0.416038, -0.07753],
        [-0.417564, 0.623187], [0.410952, 0.634963],
        [0.41553, -0.106972], [0.423159, -0.554489]
    ], dtype=np.float64)
    data_10 = np.array([
        [-0.373615, -0.517401], [-0.386457, -0.153166],
        [-0.410715, 0.580909], [0.359821, 0.692981],
        [0.395494, -0.102733], [0.415471, -0.50059]
    ], dtype=np.float64)
    data_11 = np.array([
        [-0.3901, -0.522934], [-0.387048, -0.076501],
        [-0.393151, 0.652881], [0.381962, 0.665457],
        [0.394168, -0.114228], [0.394168, -0.604675]
    ], dtype=np.float64)
    standard_stall = np.array([
        [-0.416547, -0.536824], [-0.416547, -0.092251],
        [-0.416547,  0.629075], [ 0.416547,  0.629075],
        [ 0.416547, -0.092251], [ 0.416547, -0.536824]
    ], dtype=np.float64)

    minima = find_local_minima(data_11, standard_stall, Nr=7)
    print("\nLocal minima found (index, value):")
    for i, (index, value) in enumerate(minima):
        matlab_index = tuple(dim + 1 for dim in index)  # 1-based indexing
        print(f"Minima {i+1}: Index {matlab_index}, Value = {value:.6f}")
    
    print(f"\nTime taken {time.time()-t0:.5f} s")