#pragma once

#include <stddef.h>
#include "types.h"

void print_progress(const size_t iter, const size_t max_iter);
void print_progress_target(const size_t iter, const size_t max_iter);

wfc_blocks *safe_malloc(uint64_t blkcnt);
void safe_free(wfc_blocks_ptr blk);
