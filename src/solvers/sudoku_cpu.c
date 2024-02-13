#define _GNU_SOURCE

#include "bitfield.h"
#include "wfc.h"

#include <omp.h>

static inline void
print_grid(const wfc_blocks_ptr blocks)
{
    uint32_t gx = 0, gy = 0;
    for (uint32_t x = 0; x < blocks->block_side; x++) {
        for (uint32_t y = 0; y < blocks->block_side; y++) {
            printf("%5lu ", *blk_at(blocks, gx, gy, x, y));
        }
        printf("\n");
    }
}

bool
solve_cpu(wfc_blocks_ptr blocks)
{
    uint64_t iteration  = 0;
    const uint64_t seed = blocks->states[0];
    struct {
        uint32_t gy, x, y, _1;
        uint64_t state;
    } row_changes[blocks->grid_side];
    uint64_t collapsed = 0;

    (void)row_changes; /* WTF is this ? */

    forever {
        bool changed = false;

        // 1. Collapse
        entropy_location el;
        uint32_t gx = 0, gy = 0;
        el                                                    = blk_min_entropy(blocks, gx, gy);
        uint64_t state                                        = *blk_at(blocks, gx, gy, el.location.x, el.location.y);
        uint64_t new_state                                    = entropy_collapse_state(state, gx, gy, el.location.x, el.location.y, seed, iteration);
        *blk_at(blocks, gx, gy, el.location.x, el.location.y) = new_state;
        changed                                               = state != new_state;
        collapsed |= new_state;

        // 2. Propagate
        blk_propagate(blocks, gx, gy, collapsed);
        grd_propagate_row(blocks, gx, gy, el.location.x, el.location.y, collapsed);
        grd_propagate_column(blocks, gx, gy, el.location.x, el.location.y, collapsed);

        //
        printf("move: %lu -> %lu\n", state, new_state);
        print_grid(blocks);

        // 3. Check Error
        bool err = grd_check_error_in_column(blocks, gy);
        if (err) {
            printf(" Error in column %d\n", gy);
            return false;
        }

        // 4. Check for completed grid
        const uint8_t popcount = bitfield_count(state);
        if (popcount == 1)
            break;

        // 5. Fixed point
        iteration += 1;
        if (!changed) {
            printf("Not changed\n");
            return false;
        }
    }

    /* Convert the grid in readable values */
    for (uint32_t x = 0; x < blocks->block_side; ++x) {
        for (uint32_t y = 0; y < blocks->block_side; ++y) {
            // uint32_t gx = 0, gy = 0;
            // uint64_t *addr = blk_at(blocks, gx, gy, x, y);
            // uint64_t state = *addr;
            // uint8_t popcout = bitfield_count(state);
            // *addr           = popcout;
        }
    }

    return true;
}
