#pragma once

#include "types.h"

#include <stdbool.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#define forever for (;;)

/// Parses the arguments, prints the help message if needed and abort on error.
wfc_args wfc_parse_args(int argc, char **argv);

/// Load the positions from a file. You must free the thing yourself. On error
/// kill the program.
wfc_blocks_ptr wfc_load(uint64_t, const char *);

/// Clone the blocks structure. You must free the return yourself.
void wfc_clone_into(wfc_blocks_ptr *const restrict, uint64_t, const wfc_blocks_ptr);

/// Save the grid to a folder by creating a new file or overwrite it, on error kills the program.
void wfc_save_into(const wfc_blocks_ptr, const char data[], const char folder[], const bool box_drawing);

static inline uint64_t
wfc_control_states_count(uint64_t grid_size, uint64_t block_size)
{
    /*
     * grid_size * grid_size is the number of blocks in the grid
     * block_size * block_size is the number of states in each block
     */
    return grid_size * grid_size * block_size * block_size;
}

// Access to the final states mask of the given line
static inline uint64_t *
states_line(wfc_blocks_ptr blocks, uint64_t line)
{
    // Skip two cells for the seed and the range.
    return blocks->states + 2 + line;
}

// Access to the final states mask of the given column
static inline uint64_t *
states_col(wfc_blocks_ptr blocks, uint64_t column)
{
    // Skip enough for one line
    const uint64_t skip = blocks->block_side * blocks->grid_side;
    return states_line(blocks, 0) + skip + column;
}

// Access to the final states mask of the given block
static inline uint64_t *
states_block(wfc_blocks_ptr blocks, uint64_t block)
{
    // Skip enough for one block
    const uint64_t skip = blocks->block_side * blocks->grid_side;
    return states_col(blocks, 0) + skip + block;
}

// All the possible states of a given cell
uint64_t
possible_states(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y);

static inline uint64_t *
grd_at(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy)
{
    const uint64_t bs   = blocks->block_side * blocks->block_side;
    const uint64_t skip = blocks->block_side * blocks->block_side;

    /*
     * gs is the number of states in one block
     *
     * Skipping gx lines: gx * blocks->grid_side * bs
     * Skipping gy columns: gy * bs
     */
    return states_block(blocks, 0) + skip + blocks->grid_side * bs * gx + gy * bs;
}

static inline uint64_t *
blk_at(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y)
{
    // Getting the pointer to the start of the block
    uint64_t *start = grd_at(blocks, gx, gy);

    /*
     * Skipping x lines: x * blocks->block_side
     * Skipping y columns: y
     */
    return start + blocks->block_side * x + y;
}

// Printing functions
void blk_print(FILE *const, const wfc_blocks_ptr block, uint32_t gx, uint32_t gy);
void grd_print(FILE *const, const wfc_blocks_ptr block);

// Entropy functions
entropy_location grd_min_entropy(const wfc_blocks_ptr blocks);
uint8_t entropy_compute(uint64_t);
uint64_t entropy_collapse_state(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint64_t, uint64_t);

// Propagation functions
void propagate(wfc_blocks_ptr blocks, uint32_t gx, uint32_t gy, uint32_t x, uint32_t y, uint64_t state);

// Check functions
bool grd_check_error(wfc_blocks_ptr blocks);

// Solvers
bool solve_cpu(wfc_blocks_ptr, wfc_args, wfc_blocks_ptr *);
bool solve_openmp(wfc_blocks_ptr, wfc_args, wfc_blocks_ptr *);
bool solve_target(wfc_blocks_ptr, wfc_args, wfc_blocks_ptr *);
#if defined(WFC_CUDA)
bool solve_cuda(wfc_blocks_ptr, wfc_args, wfc_blocks_ptr *);
#endif

static const wfc_solver solvers[] = {
    { "cpu", solve_cpu },
    { "omp", solve_openmp },
    { "target", solve_target },
#if defined(WFC_CUDA)
    { "cuda", solve_cuda },
#endif
};
