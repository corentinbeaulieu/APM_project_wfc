#define _GNU_SOURCE

#include "wfc.h"

#include <omp.h>

void
grk_recompute(wfc_blocks_ptr blocks);

bool
solve_cpu(wfc_blocks_ptr blocks)
{
    uint64_t iteration  = 0;
    const uint64_t seed = blocks->states[0];

    forever {
        bool changed = false;

        // 1. Collapse
        entropy_location el;
        el                = grd_min_entropy(blocks);
        const uint32_t gx = el.grid_location.x, gy = el.grid_location.y;
        const uint32_t x = el.location.x, y = el.location.y;
        uint64_t state                = *blk_at(blocks, gx, gy, x, y);
        uint64_t new_state            = entropy_collapse_state(state, gx, gy, x, y, seed, iteration);
        *blk_at(blocks, gx, gy, x, y) = new_state;
        changed                       = state != new_state;

        // 2. Propagate
        grk_recompute(blocks);

        // 3. Check Error
        if (grd_check_error(blocks)) {
            fprintf(stderr, " Error at iteration %lu\n", iteration);
            return false;
        }

        // 4. Check for completed grid
        const uint8_t popcount = (uint8_t)__builtin_popcountll(state);
        if (popcount == 1) {
            break;
        }

        // 5. Fixed point
        iteration += 1;
        if (!changed) {
            fprintf(stderr, "Not changed\n");
            return false;
        }
    }

    return true;
}
