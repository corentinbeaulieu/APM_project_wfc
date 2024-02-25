#define _GNU_SOURCE

#include "utils.h"
#include "wfc.h"
#include <string.h>

#include <omp.h>
#include "utils.h"

static inline bool
solve(wfc_blocks_ptr blocks)
{
    uint64_t iteration = 0;

    // Seed in the first cell
    const uint64_t seed = blocks->states[0];

    forever {
        // Find next cell
        entropy_location el = grd_min_entropy_target(blocks);
        const uint32_t gx = el.grid_location.x, gy = el.grid_location.y;
        const uint32_t x = el.location.x, y = el.location.y;
        const uint64_t choice = el.choice;

        // Check for finish
        if (el.entropy == UINT8_MAX)
            break;

        // Check for error
        if (!choice) {
            return false;
        }

        // Update internal states
        uint64_t new_state = entropy_collapse_state_target(choice, gx, gy, x, y, seed, iteration);

        *blk_at(blocks, gx, gy, x, y) = new_state;
        propagate_target(blocks, gx, gy, x, y, new_state);
        iteration++;
    }
    return true;
}

bool
solve_target(wfc_blocks_ptr init, wfc_args args, wfc_blocks_ptr *res, uint64_t *const iterations)
{
    unsigned char quit            = false;
    unsigned char solved          = false;
    uint64_t iter                 = *iterations;
    const uint64_t num_threads    = args.parallel;
    const uint64_t num_teams      = args.teams;
    const uint64_t max_iterations = args.seeds.count;

    const uint64_t states_count = (wfc_control_states_count(init->grid_side, init->block_side) +
                                   ((2 + 3 * init->block_side * init->block_side)));

    wfc_blocks final = {
        calloc(states_count, sizeof(uint64_t)),
        init->grid_side,
        init->block_side
    };
    uint64_t *const tmp_blocks = calloc(states_count * num_threads * num_teams, sizeof(uint64_t));

#pragma omp target map(to : init, init->block_side, init->grid_side, init->states[ : states_count], quit, args.seeds, args.seeds.count, args.seeds.start) \
    map(tofrom : iter, solved, final, final.states[ : states_count], final.grid_side, final.block_side)\
    map(alloc : tmp_blocks[ : states_count * num_threads * num_teams])
    {
#pragma omp teams shared(init, quit, solved, res, iter) num_teams(num_teams)
        {
            const uint64_t seed_per_team = args.seeds.count > (uint64_t)omp_get_num_teams() ? args.seeds.count / omp_get_num_teams() : 1;
            const uint64_t remaining = args.seeds.count % omp_get_num_teams();
            const uint64_t start     = omp_get_team_num() * seed_per_team;
            const uint64_t stop      = omp_get_team_num() == omp_get_num_teams() - 1 ? start + seed_per_team + remaining : start + seed_per_team;
            if(omp_is_initial_device() && omp_get_team_num() == 0) printf("NOT ON GPU");

#pragma omp parallel for num_threads(num_threads) shared(init, quit, res, iter, solved)
            for (uint64_t i = start; i < stop; ++i) {
                if (quit) {
                    continue;
                    //i = args.seeds.count; // FIXME: hardcore break out of loop
                }

                const uint64_t seed = args.seeds.start + i;
                const int global_id = omp_get_num_threads() * omp_get_team_num() + omp_get_thread_num();

                wfc_blocks tmp_block = { tmp_blocks + global_id * states_count, init->block_side, init->grid_side };

                memcpy(tmp_block.states, init->states, states_count * sizeof(uint64_t));
                tmp_block.states[0] = seed;
                bool _solved        = solve(&tmp_block);
#pragma omp critical
                {
                    iter++;

                    if (_solved && !solved) {
                        solved = true;
                        quit = true;

                        memcpy(final.states, tmp_block.states, states_count * sizeof(uint64_t));
                    }

                    if (iter % num_threads == 0) {
                        print_progress_target(iter, max_iterations);
                    }
                }
            }
        }
    }

    *iterations    = iter;
    if(*res == NULL) {
        *res = safe_malloc(states_count);
    }

    memcpy((*res)->states, final.states, states_count * sizeof(uint64_t));
    (*res)->block_side = final.grid_side;
    (*res)->grid_side = final.block_side;

    return solved;
}
