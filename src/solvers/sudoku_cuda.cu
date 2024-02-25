extern "C" {
#include "types.h"
#include "utils.h"

bool
solve_cuda(wfc_blocks_ptr init, wfc_args args, wfc_blocks_ptr *res, uint64_t *iterations);
} /* extern "C" */

#include "wfc.cuh"
#include <stdio.h>

__global__ void
solve(uint64_t *states, uint64_t grid_size, uint64_t block_size, uint64_t seed, int *solved)
{
    //
    const uint64_t bid       = blockIdx.x;
    const uint64_t tid       = threadIdx.x;
    const uint64_t n_threads = blockDim.x;

    // Iteration
    uint64_t iteration  = 0;
    const uint64_t size = 3 * block_size * block_size + 2 + wfc_control_states_count(grid_size, block_size);

    // Fast access to the states
    __shared__ uint64_t s_states[4290];

    // Copy the states to shared memory using all the threads
    const uint64_t n_per_thread = size / n_threads;
    const uint64_t start        = tid * n_per_thread;
    uint64_t end                = start + n_per_thread;
    if (tid == blockDim.x - 1)
        end = size;
    for (uint64_t i = start; i < end; ++i)
        s_states[i] = states[i];

    __syncthreads();

    // Seed in the first cell
    seed += bid;

    // Not solved
    *solved = false;

    for (;;) {
        // Find next cell
        entropy_location el = grd_min_entropy(s_states, grid_size, block_size);
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
        uint64_t new_state                                     = entropy_collapse_state(choice, gx, gy, x, y, seed, iteration);
        *blk_at(s_states, grid_size, block_size, gx, gy, x, y) = new_state;
        propagate(s_states, grid_size, block_size, gx, gy, x, y, new_state);
        iteration++;
    }

    // Copy the states back
    for (uint64_t i = start; i < end; ++i)
        states[i] = s_states[i];

    // Store the seed
    states[0] = seed;

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

    //
    int gs = 500;
    int bs = 32;
    cudaMemcpy(d_states, init->states, size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    for (uint64_t i = 0; i < args.seeds.count; i += gs) {
        const uint64_t seed = args.seeds.start + i;

        solve<<<gs, bs>>>(d_states, grid_size, block_size, seed, d_solved);
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
