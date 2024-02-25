#include <inttypes.h>
#include "types.h"
#include "wfc.cuh"
#include "md5.cuh"

__host__ void
wfc_clone_into(const wfc_blocks *init, const uint64_t seed, uint64_t *d_states)
{
    const uint64_t block_size = init->block_side;
    const uint64_t grid_size  = init->grid_side;
    const uint64_t size       = 3 * block_size * block_size + 2 + wfc_control_states_count(grid_size, block_size);

    init->states[0] = seed;
    cudaMemcpy(d_states, init->states, size * sizeof(uint64_t), cudaMemcpyHostToDevice);
}

__device__ static uint64_t
possible_states(uint64_t *states, uint64_t grid_side, uint64_t block_side, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
    const uint64_t line_id  = gx * block_side + x;
    const uint64_t col_id   = gy * block_side + y;
    const uint64_t block_id = gx * grid_side + gy;

    const uint64_t mask_line  = *states_line(states, line_id);
    const uint64_t mask_col   = *states_col(states, grid_side, block_side, col_id);
    const uint64_t mask_block = *states_block(states, grid_side, block_side, block_id);

    // Range is the mask of all the possible states
    const uint64_t range = states[1];

    // Return all the state have haven't been seen in the line, color or block
    return range & ~(mask_line | mask_col | mask_block);
}

__device__ static inline uint8_t
entropy_compute(const uint64_t state)
{
    /*
     * Compute the entropy of a state.
     * The entropy is the number of bits set in the state.
     * The state with the lowest entropy is the one that will
     * be collapsed first.
     */
    return (uint8_t)__popcll(state);
}

__device__ entropy_location
grd_min_entropy(uint64_t *states, uint64_t grid_side, uint64_t block_side)
{
    entropy_location ret = { 0 };
    ret.entropy          = UINT8_MAX;

    /*
     * Find a non-final state that has the lowest entropy.
     */
    for (uint32_t gy = 0; gy < grid_side; ++gy) {
        for (uint32_t gx = 0; gx < grid_side; ++gx) {
            for (uint32_t y = 0; y < block_side; ++y) {
                for (uint32_t x = 0; x < block_side; ++x) {
                    if (!*blk_at(states, grid_side, block_side, gx, gy, x, y)) {
                        const uint64_t choice = possible_states(states, grid_side, block_side, gx, gy, x, y);
                        const uint8_t entropy = entropy_compute(choice);

                        if (entropy < ret.entropy) {
                            ret.location.x      = x;
                            ret.location.y      = y;
                            ret.grid_location.x = gx;
                            ret.grid_location.y = gy;
                            ret.choice          = choice;
                            ret.entropy         = entropy;
                        }
                    }
                }
            }
        }
    }

    return ret;
}

__device__ static inline uint64_t
find_nth_set_bit(uint64_t mask, uint64_t n)
{
    uint64_t t, i = n, r = 0;
    const uint64_t m1  = 0x5555555555555555ULL; // even bits
    const uint64_t m2  = 0x3333333333333333ULL; // even 2-bit groups
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0fULL; // even nibbles
    const uint64_t m8  = 0x00ff00ff00ff00ffULL; // even bytes
    const uint64_t c1  = mask;
    const uint64_t c2  = c1 - ((c1 >> 1) & m1);
    const uint64_t c4  = ((c2 >> 2) & m2) + (c2 & m2);
    const uint64_t c8  = ((c4 >> 4) + c4) & m4;
    const uint64_t c16 = ((c8 >> 8) + c8) & m8;
    const uint64_t c32 = (c16 >> 16) + c16;
    const uint64_t c64 = (uint64_t)(((c32 >> 32) + c32) & 0x7f);
    t                  = (c32)&0x3f;
    if (i >= t) {
        r += 32;
        i -= t;
    }
    t = (c16 >> r) & 0x1f;
    if (i >= t) {
        r += 16;
        i -= t;
    }
    t = (c8 >> r) & 0x0f;
    if (i >= t) {
        r += 8;
        i -= t;
    }
    t = (c4 >> r) & 0x07;
    if (i >= t) {
        r += 4;
        i -= t;
    }
    t = (c2 >> r) & 0x03;
    if (i >= t) {
        r += 2;
        i -= t;
    }
    t = (c1 >> r) & 0x01;
    if (i >= t) {
        r += 1;
    }
    if (n >= c64) {
        /* FIXME */
    }
    return r;
}

__device__ static inline uint64_t
bitfield_only_nth_set(uint64_t x, uint8_t n)
{
    return 1llu << find_nth_set_bit(x, n);
}

__device__ uint64_t
entropy_collapse_state(uint64_t state, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t seed, uint64_t iteration)
{
    uint8_t digest[16] = { 0 };
    struct {
        uint32_t gx, gy, x, y;
        uint64_t seed, iteration;
    } random_state = {
        .gx        = gx,
        .gy        = gy,
        .x         = x,
        .y         = y,
        .seed      = seed,
        .iteration = iteration,
    };

    md5((uint8_t *)&random_state, sizeof(random_state), digest);

    /*
     * The goal is the select a random state to collapse the current state.
     * For that we will use the MD5 hash as a random number generator.
     * First we will need to do a popcount to see how many possible states
     * there are. Then we will use the MD5 hash to select a random number
     * between 0 and the number of possible states.
     */

    // Number of states
    const uint8_t popcount = (uint8_t)__popcll(state);

    // Select a random state
    uint64_t index = digest[0] % popcount;

    return bitfield_only_nth_set(state, (uint8_t)index);
}

__device__ void
propagate(uint64_t *states, uint64_t grid_side, uint64_t block_side, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t state)
{
    /*
     * We store the final state of each line, column and block.
     * This will allow us to determine the entropy/choice of a state
     * quickly.
     */
    const uint64_t line_id  = gx * block_side + x;
    const uint64_t col_id   = gy * block_side + y;
    const uint64_t block_id = gx * grid_side + gy;

    *states_line(states, line_id) |= state;
    *states_col(states, grid_side, block_side, col_id) |= state;
    *states_block(states, grid_side, block_side, block_id) |= state;
}
