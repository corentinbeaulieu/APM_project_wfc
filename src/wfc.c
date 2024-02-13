#define _GNU_SOURCE

#include "wfc.h"
#include "bitfield.h"
#include "utils.h"
#include "md5.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <limits.h>
#include <errno.h>
#include <string.h>
#include <strings.h>

uint64_t
entropy_collapse_state(uint64_t state,
                       uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                       uint64_t seed,
                       uint64_t iteration)
{
    uint8_t digest[16]     = { 0 };
    uint64_t random_number = 0;
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
    const uint8_t popcount = bitfield_count(state);

    // Random number
#define COMBINE(a, b) (a) ^ ((b) + 0x517cc1b727220a95 + ((a) << 6) + ((a) >> 2))
    uint64_t *digest64 = (uint64_t *)digest;
    random_number      = COMBINE(digest64[0], digest64[1]);

    // Select a random state
    uint8_t index = ((uint8_t)random_number) % popcount;

    uint64_t ret = bitfield_only_nth_set(state, index);

    return ret;
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
    uint8_t popcount = bitfield_count(state);

    if (popcount == 1)
        return UINT8_MAX; // The state is set
    return popcount;
}

void
wfc_clone_into(wfc_blocks_ptr *const restrict ret_ptr, uint64_t seed, const wfc_blocks_ptr blocks)
{
    const uint64_t grid_size  = blocks->grid_side;
    const uint64_t block_size = blocks->block_side;
    wfc_blocks_ptr ret        = *ret_ptr;

    const uint64_t size = (wfc_control_states_count(grid_size, block_size) * sizeof(uint64_t)) +
                          (grid_size * grid_size * block_size * block_size * sizeof(uint64_t)) +
                          sizeof(wfc_blocks);

    if (NULL == ret) {
        if (NULL == (ret = malloc(size))) {
            fprintf(stderr, "failed to clone blocks structure\n");
            exit(EXIT_FAILURE);
        }
    } else if (grid_size != ret->grid_side || block_size != ret->block_side) {
        fprintf(stderr, "size mismatch!\n");
        exit(EXIT_FAILURE);
    }

    memcpy(ret, blocks, size);
    ret->states[0] = seed;
    *ret_ptr       = ret;
}

entropy_location
blk_min_entropy(const wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
{
    vec2 the_location   = { 0 };
    uint8_t min_entropy = UINT8_MAX;

    entropy_location ret = { 0 };

    // Iterate over the block
    for (uint32_t x = 0; x < blocks->block_side; ++x) {
        for (uint32_t y = 0; y < blocks->block_side; ++y) {
            const uint64_t state  = *blk_at(blocks, gx, gy, x, y);
            const uint8_t entropy = entropy_compute(state);

            if (entropy < min_entropy) {
                min_entropy    = entropy;
                the_location.x = x;
                the_location.y = y;
            }
        }
    }

    ret.entropy  = min_entropy;
    ret.location = the_location;

    return ret;
}

// static inline uint64_t
// blk_filter_mask_for_column(wfc_blocks_ptr blocks,
//                            uint32_t gy, uint32_t y,
//                            uint64_t collapsed)
// {
//     /*
//      * Compute the mask for the column.
//      * The mask is the OR of all the states that are not collapsed.
//      */
//     uint64_t mask = 0;
//
//     // Slide over the column
//     for (uint32_t gx = 0; gx < blocks->grid_side; ++gx) {
//         for (uint32_t x = 0; x < blocks->block_side; ++x) {
//             const uint64_t state = *blk_at(blocks, gx, gy, x, y);
//
//             // Skip the collapsed states
//             mask |= state & ~collapsed;
//         }
//     }
//
//     //
//     return mask;
// }

// static inline uint64_t
// blk_filter_mask_for_row(wfc_blocks_ptr blocks,
//                         uint32_t gx, uint32_t x,
//                         uint64_t collapsed)
// {
//     /*
//      * Compute the mask for the row.
//      * The mask is the OR of all the states that are not collapsed.
//      */
//     uint64_t mask = 0;
//
//     // Slide over the row
//     for (uint32_t gy = 0; gy < blocks->grid_side; ++gy) {
//         for (uint32_t y = 0; y < blocks->block_side; ++y) {
//             const uint64_t state = *blk_at(blocks, gx, gy, x, y);
//
//             // Skip the collapsed states
//             mask |= state & ~collapsed;
//         }
//     }
//
//     //
//     return mask;
// }

// static inline uint64_t
// blk_filter_mask_for_block(wfc_blocks_ptr blocks,
//                           uint32_t gy, uint32_t gx,
//                           uint64_t collapsed)
// {
//     /*
//      * Compute the mask for the block.
//      * The mask is the OR of all the states that are not collapsed.
//      */
//     uint64_t mask = 0;
//
//     // Slide over the block
//     for (uint32_t y = 0; y < blocks->block_side; ++y) {
//         for (uint32_t x = 0; x < blocks->block_side; ++x) {
//             const uint64_t state = *blk_at(blocks, gx, gy, x, y);
//
//             // Skip the collapsed states
//             mask |= state & ~collapsed;
//         }
//     }
//
//     //
//     return mask;
// }

bool
grd_check_error_in_column(wfc_blocks_ptr blocks, uint32_t gx)
{
    for (uint32_t y = 0; y < blocks->block_side; ++y) {
        uint64_t seen = 0;
        for (uint32_t gy = 0; gy < blocks->grid_side; ++gy) {
            for (uint32_t x = 0; x < blocks->block_side; ++x) {
                const uint64_t state   = *blk_at(blocks, gx, gy, x, y);
                const uint8_t popcount = bitfield_count(state);

                // Not assigned ?
                if (popcount != 1)
                    continue;

                // Error
                if (seen & state) {
                    printf("failed at: %lu %lu\n", state, seen);
                    return true;
                }

                // Saving the state
                seen |= state;
            }
        }
    }

    return false;
}

void
blk_propagate(wfc_blocks_ptr blocks,
              uint32_t gx, uint32_t gy,
              uint64_t collapsed)
{
    /*
     * Slide over the block and remove the collapsed states from the possible states.
     */
    for (uint32_t y = 0; y < blocks->block_side; ++y) {
        for (uint32_t x = 0; x < blocks->block_side; ++x) {
            uint64_t *state = blk_at(blocks, gx, gy, x, y);
            uint64_t mask   = *state & ~collapsed;
            if (mask)
                *state = mask;
        }
    }
}

void
grd_propagate_row(wfc_blocks_ptr blocks,
                  uint32_t gx, uint32_t gy, uint32_t x, uint32_t y,
                  uint64_t collapsed)
{
    /*
     * Slide over the row and remove the collapsed states from the possible states.
     */
    for (gy = 0; gy < blocks->grid_side; ++gy) {
        for (y = 0; y < blocks->block_side; ++y) {
            uint64_t *state = blk_at(blocks, gx, gy, x, y);
            uint64_t mask   = *state & ~collapsed;
            if (mask)
                *state = mask;
        }
    }
}

void
grd_propagate_column(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy,
                     uint32_t x, uint32_t y, uint64_t collapsed)
{
    /*
     * Slide over the column and remove the collapsed states from the possible states.
     */
    for (gx = 0; gx < blocks->grid_side; ++gx) {
        for (x = 0; x < blocks->block_side; ++x) {
            uint64_t *state = blk_at(blocks, gx, gy, x, y);
            uint64_t mask   = *state & ~collapsed;
            if (mask)
                *state = mask;
        }
    }
}
