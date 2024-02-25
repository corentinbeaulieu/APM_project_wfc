#define _GNU_SOURCE

#include "wfc.h"
#include "bitfield.h"
#include "utils.h"
#include "md5.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <strings.h>

#include <omp.h>

uint64_t
entropy_collapse_state(uint64_t state,
                       uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                       uint64_t seed,
                       uint64_t iteration)
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
    const uint8_t popcount = (uint8_t)__builtin_popcountll(state);

    // Select a random state
    uint64_t index = digest[0] % popcount;

    return bitfield_only_nth_set(state, (uint8_t)index);
}

uint8_t
entropy_compute(uint64_t state)
{
    /*
     * Compute the entropy of a state.
     * The entropy is the number of bits set in the state.
     * The state with the lowest entropy is the one that will
     * be collapsed first.
     */
    return (uint8_t)__builtin_popcountll(state);
}

void
wfc_clone_into(wfc_blocks_ptr *const restrict ret_ptr, uint64_t seed, const wfc_blocks_ptr blocks)
{
    const uint8_t grid_size  = blocks->grid_side;
    const uint8_t block_size = blocks->block_side;
    const uint64_t blkcnt    = 3 * block_size * block_size + 2 + wfc_control_states_count(grid_size, block_size);

    wfc_blocks_ptr ret = safe_malloc(blkcnt);
    ret->grid_side     = grid_size;
    ret->block_side    = block_size;

    memcpy(ret->states, blocks->states, blkcnt * sizeof(uint64_t));
    ret->states[0] = seed;
    *ret_ptr       = ret;
}

entropy_location
grd_min_entropy(const wfc_blocks_ptr blocks)
{
    entropy_location ret = { 0 };
    ret.entropy          = UINT8_MAX;

    /*
     * Find a non-final state that has the lowest entropy.
     */
    for (uint32_t gy = 0; gy < blocks->grid_side; ++gy) {
        for (uint32_t gx = 0; gx < blocks->grid_side; ++gx) {
            for (uint32_t y = 0; y < blocks->block_side; ++y) {
                for (uint32_t x = 0; x < blocks->block_side; ++x) {
                    if (!*blk_at(blocks, gx, gy, x, y)) {
                        const uint64_t choice = possible_states(blocks, gx, gy, x, y);
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

bool
grd_check_error(wfc_blocks_ptr blocks)
{
    /*
     * Only used to check the validity of the grid
     * at the input sanity check.
     * A call to this function can be avoided in runtime
     * by returning the choice in grd_min_entropy.
     */
    for (uint32_t gy = 0; gy < blocks->grid_side; ++gy) {
        for (uint32_t gx = 0; gx < blocks->grid_side; ++gx) {
            for (uint32_t y = 0; y < blocks->block_side; ++y) {
                for (uint32_t x = 0; x < blocks->block_side; ++x) {
                    if (!*blk_at(blocks, gx, gy, x, y)) {
                        const uint64_t choice = possible_states(blocks, gx, gy, x, y);

                        /*
                         * No possible state for non-final cell.
                         * This is impossible.
                         */
                        if (!choice)
                            return true;
                    }
                }
            }
        }
    }

    return false;
}

// Store a mask into the final states
void
propagate(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t state)
{
    /*
     * We store the final state of each line, column and block.
     * This will allow us to determine the entropy/choice of a state
     * quickly.
     */
    const uint64_t line_id  = gx * blocks->block_side + x;
    const uint64_t col_id   = gy * blocks->block_side + y;
    const uint64_t block_id = gx * blocks->grid_side + gy;

    *states_line(blocks, line_id) |= state;
    *states_col(blocks, col_id) |= state;
    *states_block(blocks, block_id) |= state;
}

uint64_t
possible_states(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
    const uint64_t line_id  = gx * blocks->block_side + x;
    const uint64_t col_id   = gy * blocks->block_side + y;
    const uint64_t block_id = gx * blocks->block_side + gy;

    const uint64_t mask_line  = *states_line(blocks, line_id);
    const uint64_t mask_col   = *states_col(blocks, col_id);
    const uint64_t mask_block = *states_block(blocks, block_id);

    // Range is the mask of all the possible states
    const uint64_t range = blocks->states[1];

    // Return all the state have haven't been seen in the line, color or block
    return range & ~(mask_line | mask_col | mask_block);
}

/** Redefinitions for OMP target
 */
#pragma omp declare target
uint64_t
entropy_collapse_state_target(uint64_t state,
                              uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                              uint64_t seed,
                              uint64_t iteration)
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
#pragma omp allocate(digest, random_state) allocator(omp_thread_mem_alloc)

    md5_target((uint8_t *)&random_state, sizeof(random_state), digest);

    /*
     * The goal is the select a random state to collapse the current state.
     * For that we will use the MD5 hash as a random number generator.
     * First we will need to do a popcount to see how many possible states
     * there are. Then we will use the MD5 hash to select a random number
     * between 0 and the number of possible states.
     */

    // Number of states
    const uint8_t popcount = (uint8_t)__builtin_popcountll(state);

    // Select a random state
    uint64_t index = digest[0] % popcount;

    return bitfield_only_nth_set_target(state, (uint8_t)index);
}

entropy_location
grd_min_entropy_target(const wfc_blocks_ptr blocks)
{
    entropy_location ret = { 0 };
    ret.entropy          = UINT8_MAX;

    /*
     * Find a non-final state that has the lowest entropy.
     */
    for (uint32_t gy = 0; gy < blocks->grid_side; ++gy) {
        for (uint32_t gx = 0; gx < blocks->grid_side; ++gx) {
            for (uint32_t y = 0; y < blocks->block_side; ++y) {
                for (uint32_t x = 0; x < blocks->block_side; ++x) {
                    if (!*blk_at(blocks, gx, gy, x, y)) {
                        const uint64_t choice = possible_states_target(blocks, gx, gy, x, y);
                        const uint8_t entropy = (uint8_t)__builtin_popcountll(choice);

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

bool
grd_check_error_target(wfc_blocks_ptr blocks)
{
    /*
     * Only used to check the validity of the grid
     * at the input sanity check.
     * A call to this function can be avoided in runtime
     * by returning the choice in grd_min_entropy.
     */
    for (uint32_t gy = 0; gy < blocks->grid_side; ++gy) {
        for (uint32_t gx = 0; gx < blocks->grid_side; ++gx) {
            for (uint32_t y = 0; y < blocks->block_side; ++y) {
                for (uint32_t x = 0; x < blocks->block_side; ++x) {
                    if (!*blk_at(blocks, gx, gy, x, y)) {
                        const uint64_t choice = possible_states(blocks, gx, gy, x, y);

                        /*
                         * No possible state for non-final cell.
                         * This is impossible.
                         */
                        if (!choice)
                            return true;
                    }
                }
            }
        }
    }

    return false;
}

// Store a mask into the final states
void
propagate_target(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t state)
{
    /*
     * We store the final state of each line, column and block.
     * This will allow us to determine the entropy/choice of a state
     * quickly.
     */
    const uint64_t line_id  = gx * blocks->block_side + x;
    const uint64_t col_id   = gy * blocks->block_side + y;
    const uint64_t block_id = gx * blocks->grid_side + gy;

    *states_line(blocks, line_id) |= state;
    *states_col(blocks, col_id) |= state;
    *states_block(blocks, block_id) |= state;
}

uint64_t
possible_states_target(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
    const uint64_t line_id  = gx * blocks->block_side + x;
    const uint64_t col_id   = gy * blocks->block_side + y;
    const uint64_t block_id = gx * blocks->block_side + gy;

    const uint64_t mask_line  = *states_line(blocks, line_id);
    const uint64_t mask_col   = *states_col(blocks, col_id);
    const uint64_t mask_block = *states_block(blocks, block_id);

    // Range is the mask of all the possible states
    const uint64_t range = blocks->states[1];

    // Return all the state have haven't been seen in the line, color or block
    return range & ~(mask_line | mask_col | mask_block);
}
#pragma omp end declare target
