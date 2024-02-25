#pragma once

#include "types.h"

__device__ entropy_location grd_min_entropy(uint64_t *states, uint64_t grid_size, uint64_t block_size);
__device__ uint64_t entropy_collapse_state(uint64_t choice, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t seed, uint64_t iteration);
__device__ void propagate(uint64_t *states, uint64_t grid_size, uint64_t block_size, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t new_state);

__host__ __device__ static inline uint64_t
wfc_control_states_count(uint64_t grid_size, uint64_t block_size)
{
    return grid_size * grid_size * block_size * block_size;
}

// Access to the final states mask of the given line
__host__ __device__ static inline uint64_t *
states_line(uint64_t *states, uint64_t line)
{
    // Skip two cells for the seed and the range.
    return states + 2 + line;
}

// Access to the final states mask of the given column
__host__ __device__ static inline uint64_t *
states_col(uint64_t *states, uint64_t grid_size, uint64_t block_size, uint64_t column)
{
    // Skip enough for one line
    const uint64_t skip = block_size * grid_size;
    return states_line(states, 0) + skip + column;
}

// Access to the final states mask of the given block
__host__ __device__ static inline uint64_t *
states_block(uint64_t *states, uint64_t grid_size, uint64_t block_size, uint64_t block)
{
    // Skip enough for one block
    const uint64_t skip = block_size * grid_size;
    return states_col(states, grid_size, block_size, 0) + skip + block;
}

__host__ __device__ static inline uint64_t *
grd_at(uint64_t *states, uint64_t grid_side, uint64_t block_side, uint32_t gx, uint32_t gy)
{
    const uint64_t bs   = block_side * block_side;
    const uint64_t skip = block_side * block_side;

    /*
     * gs is the number of states in one block
     *
     * Skipping gx lines: gx * blocks->grid_side * bs
     * Skipping gy columns: gy * bs
     */
    return states_block(states, grid_side, block_side, 0) + skip + grid_side * bs * gx + gy * bs;
}

__host__ __device__ static inline uint64_t *
blk_at(uint64_t *states, uint64_t grid_side, uint64_t block_side, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
    // Getting the pointer to the start of the block
    uint64_t *start = grd_at(states, grid_side, block_side, gx, gy);

    /*
     * Skipping x lines: x * blocks->block_side
     * Skipping y columns: y
     */
    return start + block_side * x + y;
}
