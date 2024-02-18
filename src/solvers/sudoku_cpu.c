#define _GNU_SOURCE

#include "wfc.h"

#include <omp.h>

void
grk_recompute(wfc_blocks_ptr blocks);

bool
solve_cpu(wfc_blocks_ptr blocks)
{
    uint64_t iteration = 0;

    // Seed in the first cell
    const uint64_t seed = blocks->states[0];

    forever {
        // Find next cell
        entropy_location el = grd_min_entropy(blocks);
        const uint32_t gx = el.grid_location.x, gy = el.grid_location.y;
        const uint32_t x = el.location.x, y = el.location.y;
        const uint64_t choice = el.choice;

        // Check for finish
        if (el.entropy == UINT8_MAX)
            break;

        // Check for error
        if (!choice)
            return fprintf(stderr, " Error at iteration %lu\n", iteration), false;

        // Update internal states
        uint64_t new_state            = entropy_collapse_state(choice, gx, gy, x, y, seed, iteration);
        *blk_at(blocks, gx, gy, x, y) = new_state;
        propagate(blocks, gx, gy, x, y, new_state);
        iteration++;
    }

    return true;
}
