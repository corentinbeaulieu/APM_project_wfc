#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <omp.h>
#include "utils.h"

static inline bool
_solve(wfc_blocks_ptr blocks)
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

bool
solve_openmp(wfc_blocks_ptr init, wfc_args args, wfc_blocks_ptr *res)
{
    (void)init;

    _Atomic unsigned char quit    = false;
    _Atomic unsigned char solved  = false;
    const uint64_t num_threads    = args.parallel;
    const uint64_t max_iterations = args.seeds.count;

#pragma omp parallel num_threads(num_threads) firstprivate(init) shared(quit, solved)
#pragma omp single
    {
        for (uint64_t i = 0; i < args.seeds.count; ++i) {
            const uint64_t seed = args.seeds.start + i;

#pragma omp task
            {
                wfc_blocks_ptr tmp_blocks = NULL;
                wfc_clone_into(&tmp_blocks, seed, init);
                bool _solved = _solve(tmp_blocks);

                if (_solved) {
#pragma omp critical
                    {
                        solved = true;
                        quit   = true;
                    }
                    wfc_clone_into(res, seed, tmp_blocks);
                }
                free(tmp_blocks);
            }

#pragma omp critical
            if (quit) {
                i = args.seeds.count; // FIXME: hardcore break out of loop
            }
            print_progress(i, max_iterations);
        }
    }

    return solved;
}
