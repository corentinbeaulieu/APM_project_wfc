extern "C" {
#include "types.h"
#include "utils.h"

bool
solve_cuda(wfc_blocks_ptr init, wfc_args args, wfc_blocks_ptr *res, uint64_t *iterations);
} /* extern "C" */

#include "wfc.cuh"
#include <stdio.h>

__global__ void
solve(uint64_t *states, uint64_t grid_size, uint64_t block_size, int *solved)
{
    uint64_t iteration = 0;

    // Seed in the first cell
    const uint64_t seed = states[0];

    // Not solved
    *solved = false;

    for (;;) {
        // Find next cell
        entropy_location el = grd_min_entropy(states, grid_size, block_size);
        const uint32_t gx = el.grid_location.x, gy = el.grid_location.y;
        const uint32_t x = el.location.x, y = el.location.y;
        const uint64_t choice = el.choice;

        // Check for finish
        if (el.entropy == UINT8_MAX) {
            break;
        }

        // Check for error
        if (!choice) {
            return;
        }

        // Update internal states
        uint64_t new_state                                   = entropy_collapse_state(choice, gx, gy, x, y, seed, iteration);
        *blk_at(states, grid_size, block_size, gx, gy, x, y) = new_state;
        propagate(states, grid_size, block_size, gx, gy, x, y, new_state);
        iteration++;
    }

    // Happy breakdown
    *solved = true;
}

bool
solve_cuda(wfc_blocks_ptr init, wfc_args args, wfc_blocks_ptr *res, uint64_t *iterations)
{
    bool solved                   = false;
    const uint64_t max_iterations = args.seeds.count;
    const uint64_t grid_size      = init->grid_side;
    const uint64_t block_size     = init->block_side;
    const uint64_t size           = 3 * block_size * block_size + 2 + wfc_control_states_count(grid_size, block_size);

    uint64_t *d_states;
    cudaMalloc(&d_states, size * sizeof(uint64_t));
    int *d_solved, tmp_solved;
    cudaMalloc(&d_solved, sizeof(*d_solved));

    for (uint64_t i = 0; i < args.seeds.count; ++i) {
        const uint64_t seed = args.seeds.start + i;

        wfc_clone_into(init, seed, d_states);
        solve<<<1, 1>>>(d_states, grid_size, block_size, d_solved);
        cudaMemcpy(&tmp_solved, d_solved, sizeof(tmp_solved), cudaMemcpyDeviceToHost);

        (*iterations)++;
        solved |= tmp_solved;
        if (solved) {
            printf("Solved at iteration %lu\n", i);

            // Allocate the result
            wfc_blocks *ret = (wfc_blocks *)malloc(sizeof(wfc_blocks));
            *res            = ret;

            // Copy the result
            ret->grid_side  = grid_size;
            ret->block_side = block_size;
            ret->states     = (uint64_t *)malloc(size * sizeof(uint64_t));
            ret->states[0]  = seed;
            cudaMemcpy(ret->states, d_states, size * sizeof(uint64_t), cudaMemcpyDeviceToHost);

            return true;
        }

        print_progress(i, max_iterations);
    }

    cudaFree(d_states);

    return false;
}
