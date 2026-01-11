import jax
import jax.numpy as jnp

from jaxtyping import Array
NUM_SEEDS = 200
# Increase for larger cells in lower regions, decrease for smaller cells in lower regions
LOWER_THRESHOLD = 40 # was 40 then 80
# Increase for larger cells in higher regions, decrease for smaller cells in lower regions
UPPER_THRESHOLD = 100 # was 100 then 200


offset_vectors = jnp.array(
    [[-1, -1], [-1, 0], [-1, 1],
     [ 0, -1], [ 0, 0], [ 0, 1],
     [ 1, -1], [ 1, 0], [ 1, 1]]
)  # For jump, scale by offset


class Voronoi:
    def __init__(self, height: int, width: int, seeds: Array):  # CHANGED
        self.setup(height, width, seeds)

    def setup(self, height: int, width: int, seeds: Array):      # CHANGED
        self.seeds = jnp.round(jnp.array(seeds)).astype(jnp.int16)
        self.height = height                                    # CHANGED
        self.width = width                                      # CHANGED
        self.numseeds = seeds.shape[0]

        with jax.default_device(jax.devices("cpu")[0]):
            arr = jnp.zeros((height, width, 2, 2), dtype=self.seeds.dtype)  # CHANGED
            seed_values = jnp.tile(self.seeds[:, None, :], (1, 2, 1))
            self.seed_map = arr.at[self.seeds[:, 0], self.seeds[:, 1]].set(seed_values)

            # Use the larger dimension to define JFA step schedule. [web:16]
            max_dim = max(height, width)                        # CHANGED
            max_exp = int(jnp.floor(jnp.log2(max_dim // 2)))    # CHANGED
            base = max_dim // 2                                 # CHANGED
            self.jfa_offsets = [base // (2**i) for i in range(max_exp + 1)] + [1, 1]

    @staticmethod
    @jax.jit
    def _make_seeded_arr(height: int, width: int, seeds: Array) -> Array:  # CHANGED
        arr = jnp.zeros((height, width, 2))
        return arr.at[seeds[:, 0], seeds[:, 1]].set(seeds)

    @staticmethod
    @jax.jit
    def _jfa_step(arr: Array, offset: int):
        H, W = arr.shape[0], arr.shape[1]                      # CHANGED

        def step(i, j):
            pos = jnp.array([i, j])
            shifts = offset_vectors * offset
            candidates = pos + shifts
            # Clip separately in y (0..H-1) and x (0..W-1)          # CHANGED
            candidates_y = jnp.clip(candidates[:, 0], 0, H - 1)
            candidates_x = jnp.clip(candidates[:, 1], 0, W - 1)
            info = arr[candidates_y, candidates_x]
            all_sites = info.reshape(-1, 2)
            dists = jnp.linalg.norm(all_sites - pos, axis=-1)

            idxs = jnp.argsort(dists)
            first = all_sites[idxs[0]]  # Closest

            mask = ~jnp.all(all_sites == first, axis=1)
            dists_masked = jnp.where(mask, dists, jnp.inf)
            idx2 = jnp.argmin(dists_masked)
            second = all_sites[idx2]
            return jnp.stack([first, second], axis=0)

        vmapped = jax.vmap(jax.vmap(step, in_axes=(None, 0)), in_axes=(0, None))
        grid_y = jnp.arange(H)                                   # CHANGED
        grid_x = jnp.arange(W)                                   # CHANGED
        return vmapped(grid_y, grid_x)                           # CHANGED

    def jfa(self):
        arr = self.seed_map  # shape (H, W, 2, 2)
        for offset in self.jfa_offsets:
            arr = self._jfa_step(arr, offset)

        H, W = arr.shape[0], arr.shape[1]
        grid_y, grid_x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")  # CHANGED
        pos_grid = jnp.stack([grid_y, grid_x], axis=-1)  # shape: (H, W, 2)
        first_seed_grid = arr[..., 0, :]
        second_seed_grid = arr[..., 1, :]
        dist_first = jnp.linalg.norm(first_seed_grid - pos_grid, axis=-1)
        dist_second = jnp.linalg.norm(second_seed_grid - pos_grid, axis=-1)

        mask = dist_first < dist_second
        closest = jnp.where(mask[..., None], first_seed_grid, second_seed_grid)
        second_closest = jnp.where(mask[..., None], second_seed_grid, first_seed_grid)
        result = jnp.stack([closest, second_closest], axis=-2)
        return result

    @staticmethod
    @jax.jit
    def get_distance_transform(jfa_map: Array, dist_idx: int = 0):
        arr = jfa_map[:, :, dist_idx]
        H, W = jfa_map.shape[0], jfa_map.shape[1]               # CHANGED

        def step(i, j):
            pos = jnp.array([i, j])
            seed_pos = arr[i, j]
            return jnp.linalg.norm(seed_pos - pos)

        vmapped = jax.vmap(jax.vmap(step, in_axes=(None, 0)), in_axes=(0, None))
        grid_y = jnp.arange(H)                                  # CHANGED
        grid_x = jnp.arange(W)                                  # CHANGED
        dist_transform = vmapped(grid_y, grid_x)

        # Set 1-pixel border to 0
        dist_transform = dist_transform.at[0:2, :].set(0)
        dist_transform = dist_transform.at[-2:, :].set(0)
        dist_transform = dist_transform.at[:, 0:2].set(0)
        dist_transform = dist_transform.at[:, -2:].set(0)

        return dist_transform

    @staticmethod
    @jax.jit
    def get_border_distance_transform(jfa_map: Array):
        H, W, _, _ = jfa_map.shape
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        pos = jnp.stack([yy, xx], axis=-1)

        a = jfa_map[..., 0, :]
        b = jfa_map[..., 1, :]

        ba = b - a
        numerator = jnp.abs(jnp.sum(ba * (2 * pos - a - b), axis=-1))
        denominator = 2 * jnp.linalg.norm(ba, axis=-1) + 1e-12
        border_distance = numerator / denominator

        border_distance = border_distance.at[0:2, :].set(0)
        border_distance = border_distance.at[-2:, :].set(0)
        border_distance = border_distance.at[:, 0:2].set(0)
        border_distance = border_distance.at[:, -2:].set(0)

        return border_distance

    @staticmethod
    def get_voro_centroids(index_map: Array, num_seeds: int) -> Array:
        H, W = index_map.shape                                   # CHANGED
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")  # CHANGED
        coords = jnp.stack([yy.ravel(), xx.ravel()], axis=-1)    # CHANGED

        flat_index_map = index_map.ravel()
        mask = flat_index_map >= 0

        sum_y = jnp.bincount(flat_index_map[mask], weights=coords[mask, 0], length=num_seeds)
        sum_x = jnp.bincount(flat_index_map[mask], weights=coords[mask, 1], length=num_seeds)

        index_sum = jnp.bincount(flat_index_map[mask], length=num_seeds)

        centroids_y = sum_y / index_sum
        centroids_x = sum_x / index_sum

        return jnp.stack([centroids_y, centroids_x], axis=1)

    # get_neighbours does not assume square, uses index_map/jfa_map as-is
    def get_neighbours(self, jfa_map: Array, index_map: Array, index: int):
       idxs = jnp.where(index_map == index)
       neighbour_seeds = jnp.unique(jfa_map[idxs[0], idxs[1]][:, 1], axis=0)
       mask = ~jnp.all(neighbour_seeds == 0, axis=1) # Remove [0,0] seed
       neighbour_seeds = neighbour_seeds[mask]
       matches = (self.seeds[None, :, :] == neighbour_seeds[:, None, :]).all(-1)
       found_indices = jnp.where(matches)[1]
       return found_indices

    @staticmethod
    def get_inscribing_circles(index_map: Array, distance_transform: Array, num_seeds: int):
        H, W = index_map.shape                                   # CHANGED
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        coords = jnp.stack([yy.ravel(), xx.ravel()], axis=-1)

        flat_index_map = index_map.ravel()
        flat_dist_transform = distance_transform.ravel()
        max_masked_array = jax.ops.segment_max(
            flat_dist_transform, flat_index_map, num_seeds
        )
        seg_max_map = max_masked_array[flat_index_map]
        is_max = flat_dist_transform == seg_max_map
        indices = jnp.arange(flat_index_map.size)

        def argmax_group(seg_id):
            possible = jnp.where(
                (flat_index_map == seg_id) & is_max, indices, flat_index_map.size
            )
            return jnp.min(possible)

        max_indices = jax.vmap(argmax_group)(jnp.arange(num_seeds))
        return coords[max_indices], max_masked_array

    @staticmethod
    def get_largest_extent(index_map: Array, distance_transform: Array, seeds: Array):
        # Gets the coordinate of the max value for distance transform from closest seed
        coords, max_masked_array = Voronoi.get_inscribing_circles(
            index_map, distance_transform, len(seeds)
        )

        seed_vectors = coords - seeds  # (num_seeds, 2)
        seed_norms = jnp.linalg.norm(
            seed_vectors, axis=1, keepdims=True
        )  # (num_seeds, 1)
        unit_vectors = seed_vectors / (
            seed_norms + 1e-8
        )  # Add epsilon to avoid division by zero
        return coords, unit_vectors

    @staticmethod
    def get_split(unit_vectors: Array, dists: Array, seeds: Array):
        lower_pos = seeds + (-dists[:, None] * unit_vectors)
        upper_pos = seeds + (dists[:, None] * unit_vectors)
        return jnp.stack([lower_pos, upper_pos], axis=1)

    @staticmethod
    def get_index_map(jfa_map: Array, seeds: Array):
        arr = jfa_map[:, :, 0]
        flat_coords = arr.reshape(-1, arr.shape[-1])
        seeds = jnp.round(jnp.array(seeds)).astype(jnp.int32)
        eq = jnp.all(flat_coords[:, None, :] == seeds[None, :, :], axis=-1)
        indices = jnp.argmax(eq, axis=-1)
        any = jnp.any(eq, axis=1)
        indices = jnp.where(~any, -1, indices)
        return indices.reshape(arr.shape[0], arr.shape[1])

    def create_weighted_palette(self, index_map: Array, data: Array):
        data_normed = (data - data.min()) / (data.max() - data.min())

        flat_index_map = index_map.ravel()
        flat_weights = data_normed.ravel()

        mask = flat_index_map >= 0

        weights_sum = jnp.bincount(flat_index_map[mask], weights=flat_weights[mask])
        index_sum = jnp.bincount(flat_index_map[mask])

        avg_voro_w = weights_sum / index_sum

        palette = jnp.stack(
            [avg_voro_w * 255, avg_voro_w * 255, avg_voro_w * 255], axis=1
        )
        palette = jnp.round(palette).astype(jnp.int32)

        return palette

    def create_colour_palette(self, index_map: Array, data: Array):
        data_normed = (data - data.min()) / (data.max() - data.min())

        flat_index_map = index_map.ravel()
        flat_weights = data_normed.ravel()

        mask = flat_index_map >= 0

        weights_sum = jnp.bincount(flat_index_map[mask], weights=flat_weights[mask])
        index_sum = jnp.bincount(flat_index_map[mask])

        # Average normalized weight per Voronoi region, in [0, 1]
        avg_voro_w = weights_sum / index_sum

        # Linear interpolation: color = (1 - t) * COLOUR_1 + t * COLOUR_2
        t = avg_voro_w[:, None]  # shape (N, 1)
        palette = (1.0 - t) * jnp.array([75, 0, 130]) + t * jnp.array([255, 255, 0])

        palette = jnp.round(palette).astype(jnp.int32)

        return palette

    def create_lbg_palette(self, lower_mask, upper_mask):
        size = lower_mask.shape[0]
        red = jnp.tile(jnp.array([252, 54, 5]), (size, 1))
        white = jnp.tile(jnp.array([252, 252, 252]), (size, 1))
        green = jnp.tile(jnp.array([54, 252, 5]), (size, 1))

        palette = jnp.where(lower_mask[:, None], red, white)
        palette = jnp.where(upper_mask[:, None], green, palette)
        return palette

    def create_random_palette(self):
        with jax.default_device(jax.devices("cpu")[0]):

            def set_colour(i):
                key = jax.random.PRNGKey(i)
                return (jax.random.uniform(key, shape=(3,)) * 255).astype(jnp.int32)

            return jax.vmap(set_colour)(jnp.arange(self.numseeds + 1))

    def get_colour_map(self, index_map: Array, palette: Array):
        return palette[index_map]


@staticmethod
@jax.jit
def lloyd_step(index_map: Array, data: Array, seeds: Array) -> Array:
    H, W = index_map.shape
    if data.shape[0] != H or data.shape[1] != W:
        print("Voronoi size does not match data size")
        return seeds

    data_normed = 1 - (data - data.min()) / (data.max() - data.min())

    yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    coords = jnp.stack([yy.ravel(), xx.ravel()], axis=-1)

    flat_index_map = index_map.ravel()
    flat_weights = data_normed.ravel()

    offset_y = coords[:, 0] * flat_weights
    offset_x = coords[:, 1] * flat_weights
    sum_y = jnp.bincount(flat_index_map, weights=offset_y, length=len(seeds))
    sum_x = jnp.bincount(flat_index_map, weights=offset_x, length=len(seeds))
    weights_sum = jnp.bincount(flat_index_map, weights=flat_weights, length=len(seeds))

    lerp_t = 0.1
    float_seeds = jnp.stack(
        [
            jnp.where(weights_sum > 0, sum_y / weights_sum, seeds[:, 0]),
            jnp.where(weights_sum > 0, sum_x / weights_sum, seeds[:, 1]),
        ],
        axis=-1,
    )
    new_seeds = (1 - lerp_t) * seeds + lerp_t * float_seeds
    return new_seeds


def lbg_step(jfa_map: Array, data: Array, seeds: Array):
    H, W = jfa_map.shape[0], jfa_map.shape[1]
    if data.shape[0] != H or data.shape[1] != W:
        print("Voronoi size does not match data size")
        return seeds

    num_seeds = len(seeds)

    index_map = Voronoi.get_index_map(jfa_map, seeds)
    border_dist_transform = Voronoi.get_border_distance_transform(jfa_map)
    dist_transform = Voronoi.get_distance_transform(jfa_map, 0)
    _, unit_vectors = Voronoi.get_largest_extent(index_map, dist_transform, seeds)
    flat_index_map = index_map.ravel()
    flat_dist_transform = border_dist_transform.ravel()
    circ_r = jax.ops.segment_max(flat_dist_transform, flat_index_map, num_seeds)
    centroids = Voronoi.get_voro_centroids(index_map, num_seeds)
    split_coords = Voronoi.get_split(unit_vectors, circ_r / 2, centroids)

    data_normed = (data - data.min()) / (data.max() - data.min())
    data_normed = data_normed + 0.0001

    flat_index_map = index_map.ravel()
    flat_weights = data_normed.ravel()

    mask = flat_index_map >= 0

    weights_sum = jnp.bincount(flat_index_map[mask], weights=flat_weights[mask], length=num_seeds)

    lower_mask = weights_sum < LOWER_THRESHOLD
    upper_mask = weights_sum > UPPER_THRESHOLD

    remove_mask = lower_mask | upper_mask
    new_seeds = jnp.concatenate(
        [centroids[~remove_mask], split_coords[upper_mask].reshape(-1, 2)]
    )

    return new_seeds, lower_mask, upper_mask

prev_changed_percent = 0

def should_lbg_step(lower_mask, upper_mask, num_seeds):
    global prev_changed_percent
    changed_percent = (lower_mask.sum() + upper_mask.sum()) / num_seeds
    rate_of_change = jnp.abs(changed_percent - prev_changed_percent)
    print(f"Changed Percent: {changed_percent}, Rate of Change: {rate_of_change}")
    prev_changed_percent = changed_percent
    return (changed_percent > 0.25 and rate_of_change > 0.03 and num_seeds < 1800)

def fit_voronoi_lbg(seeds, num_seeds, voronoi, mean_map):
    count = 0
    jfa_map = voronoi.jfa()
    while True:
        seeds, lower_mask, upper_mask = lbg_step(jfa_map, mean_map, seeds)
        print(f"{count}): {seeds.shape}")
        prev_num_seeds = num_seeds
        num_seeds = seeds.shape[0]
        voronoi.setup(mean_map.shape[0], mean_map.shape[1], seeds)
        jfa_map = voronoi.jfa()
        if should_lbg_step(lower_mask, upper_mask, prev_num_seeds):
            count += 1
            continue
        else:
            break
    return seeds, num_seeds, jfa_map

if __name__ == "__main__":
    height, width = 1200, 2000  # example non-square                 # CHANGED
    key = jax.random.PRNGKey(0)
    with jax.default_device(jax.devices("cpu")[0]):
        seeds = (jax.random.uniform(key, shape=(NUM_SEEDS, 2))
                 * jnp.array([height, width])).astype(jnp.int32)     # CHANGED

    voro = Voronoi(height=height, width=width, seeds=seeds)          # CHANGED
    jfa_map = voro.jfa()
    index_map = voro.get_index_map(jfa_map, seeds)
    # pass a palette explicitly when you call get_colour_map
    # colour_map = voro.get_colour_map(index_map, palette)
