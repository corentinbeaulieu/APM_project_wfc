#define _GNU_SOURCE

#include "wfc.h"
#include "utils.h"

#include <omp.h>

static inline bool
solve(wfc_blocks_ptr blocks)
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
        if (!choice) {
            fprintf(stderr, " Error at iteration %lu\n", iteration);
            return false;
        }

        // Update internal states
        uint64_t new_state            = entropy_collapse_state(choice, gx, gy, x, y, seed, iteration);
        *blk_at(blocks, gx, gy, x, y) = new_state;
        propagate(blocks, gx, gy, x, y, new_state);
        iteration++;
    }

    return true;
}

bool
solve_cpu(wfc_blocks_ptr init, wfc_args args, wfc_blocks_ptr *res)
{
    bool solved                   = false;
    const uint64_t max_iterations = args.seeds.count;

    for (uint64_t i = 0; i < args.seeds.count; ++i) {
        wfc_blocks_ptr tmp_blocks = NULL;
        const uint64_t seed       = args.seeds.start + i;

        wfc_clone_into(&tmp_blocks, seed, init);
        solved = solve(tmp_blocks);

        if (solved) {
            wfc_clone_into(res, seed, tmp_blocks);
            free(tmp_blocks);
            return true;
        }

        free(tmp_blocks);
        print_progress(i, max_iterations);
    }

    return false;
}
