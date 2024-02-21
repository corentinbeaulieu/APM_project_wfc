#pragma once

#include <inttypes.h>
#include <stdbool.h>

typedef struct seeds_info {
    uint64_t count;
    uint64_t start;
} seeds_info;

typedef struct {
    uint32_t x, y;
} vec2;

typedef struct {
    vec2 location;
    vec2 grid_location;
    uint64_t choice;
    uint8_t entropy;
} entropy_location;

typedef struct {
    uint8_t block_side;
    uint8_t grid_side;
    uint64_t states[];
} wfc_blocks;

typedef wfc_blocks *wfc_blocks_ptr;

typedef struct wfc_args {
    const char *const data_file;
    const char *const output_folder;
    seeds_info seeds;
    const uint64_t parallel;
    bool (*const solver)(wfc_blocks_ptr, struct wfc_args, wfc_blocks_ptr *);
    const bool box_drawing;
} wfc_args;

typedef struct {
    const char *const name;
    bool (*function)(wfc_blocks_ptr, wfc_args, wfc_blocks_ptr *);
} wfc_solver;
