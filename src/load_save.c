#define _GNU_SOURCE

#include "wfc.h"
#include "utils.h"
#include "bitfield.h"

#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/// With a block side of 8, we have blocks of 8*8 := 64, which is the number of bits in an uint64_t.
static const uint8_t BLOCK_SIDE_U64 = 8;

static void
trim(char *restrict str)
{
    unsigned long start = 0, end = strlen(str) - 1;

    while (isspace(str[start])) {
        start++;
    }

    while (end > start && isspace(str[end])) {
        end--;
    }

    if (start > 0 || end < (strlen(str) - 1)) {
        memmove(str, str + start, end - start + 1);
        str[end - start + 1] = '\0';
    }
}

static char *
next(char *restrict str, char sep)
{
    char *ret = strchr(str, sep);
    if (NULL == ret) {
        fprintf(stderr, "failed to find character '%c'\n", sep);
        exit(EXIT_FAILURE);
    }
    ret[0] = '\0';
    ret += 1;
    return ret;
}

static inline uint32_t
to_u32(const char *string)
{
    char *end          = NULL;
    const long integer = strtol(string, &end, 10);
    if (integer < 0) {
        fprintf(stderr, "expected positive integer, got %ld\n", integer);
        exit(EXIT_FAILURE);
    }
    return (uint32_t)integer;
}

static inline uint64_t
to_u64(const char *string)
{
    char *end               = NULL;
    const long long integer = strtoll(string, &end, 10);
    if (integer < 0) {
        fprintf(stderr, "expected positive integer, got %lld\n", integer);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)integer;
}

// Convert string to a binary state
static inline uint64_t
to_binary_state(const char *string)
{
    uint64_t integer = to_u64(string);

    return 1llu << (integer - 1);
}

wfc_blocks_ptr
wfc_load(uint64_t seed, const char *path)
{
    srandom((uint32_t)seed);

    ssize_t read    = -1;
    char *line      = NULL;
    size_t len      = 0;
    wfc_blocks *ret = NULL;
    uint64_t blkcnt = 0;

    FILE *restrict const f = fopen(path, "r");
    if (NULL == f) {
        fprintf(stderr, "failed to open `%s`: %s\n", path, strerror(errno));
        exit(EXIT_FAILURE);
    }

    if ((read = getline(&line, &len, f)) != -1) {
        const uint32_t block_side = to_u32(&line[1]);
        if (block_side > BLOCK_SIDE_U64) {
            fprintf(stderr, "invalid header of .dat file\n");
            exit(EXIT_FAILURE);
        }

        if (line[0] == 's') {
            blkcnt          = 3 * block_side * block_side + 2;
            ret             = safe_malloc(blkcnt + wfc_control_states_count(1, block_side));
            ret->block_side = (uint8_t)block_side;
            ret->grid_side  = 1u;
        } else if (line[0] == 'g') {
            blkcnt          = 3 * block_side * block_side + 2;
            ret             = safe_malloc(blkcnt + wfc_control_states_count(block_side, block_side));
            ret->block_side = (uint8_t)block_side;
            ret->grid_side  = (uint8_t)block_side;
        } else {
            fprintf(stderr, "invalid header of .dat file\n");
            exit(EXIT_FAILURE);
        }
    } else {
        fprintf(stderr, "invalid header of .dat file\n");
        exit(EXIT_FAILURE);
    }

    {
        const uint64_t range = (1lu << (ret->block_side * ret->block_side)) - 1lu;
        const uint64_t base  = wfc_control_states_count(ret->grid_side, ret->block_side);
        ret->states[0]       = seed;
        ret->states[1]       = range;

        for (uint64_t i = 2; i < blkcnt + base; i += 1) {
            ret->states[i] = 0;
        }
    }

    while ((read = getline(&line, &len, f)) != -1) {
        trim(line);

        char *str_gx      = line;
        char *str_gy      = next(str_gx, ',');
        char *str_x       = next(str_gy, ',');
        char *str_y       = next(str_x, ',');
        char *str_state   = next(str_y, '=');
        const uint32_t gx = to_u32(str_gx), gy = to_u32(str_gy), x = to_u32(str_x),
                       y = to_u32(str_y);

        if (gx >= ret->grid_side || gy >= ret->grid_side) {
            fprintf(stderr, "invalid grid coordinates (%u, %u)\n", gx, gy);
            exit(EXIT_FAILURE);
        } else if (x >= ret->block_side || y >= ret->block_side) {
            fprintf(stderr, "invalid block coordinates (%u, %u) in grid (%u, %u)\n", x, y, gx, gy);
            exit(EXIT_FAILURE);
        }

        const uint64_t collapsed = to_binary_state(str_state);
        propagate(ret, gx, gy, x, y, collapsed);
        *blk_at(ret, gx, gy, x, y) = collapsed;
        if (grd_check_error(ret)) {
            fprintf(stderr, "wrong propagation in block (%u, %u) from (%u, %u)\n", gx, gy, x, y);
            exit(EXIT_FAILURE);
        }
    }

    free(line);
    fclose(f);
    return ret;
}

void
wfc_save_into(const wfc_blocks_ptr blocks, const char data[], const char folder[], const bool box_drawing)
{
    const size_t data_len = strlen(data);
    FILE *restrict f      = stdout;

    if (folder != NULL) {
        char destination[1024] = { 0 };
        const char *file_name  = &data[data_len - 1];
        while (file_name != data && file_name[0] != '/') {
            file_name -= 1;
        }
        const char *file_end = strchr(file_name, '.');
        long length          = (file_end - file_name);
        if (length >= 1024) {
            length = 1023;
        } else if (length < 0) {
            length = 0;
        }

        const size_t folder_len = strlen(folder);
        if (folder[folder_len - 1] == '/' && file_name[0] == '/') {
            snprintf(destination, 1023, "%.*s%.*s.%lu.save", (int)(folder_len - 1), folder, (int)length,
                     file_name, blocks->states[0]);
        } else if ((folder[folder_len - 1] == '/' && file_name[0] != '/') ||
                   (folder[folder_len - 1] != '/' && file_name[0] == '/')) {
            snprintf(destination, 1023, "%s%.*s.%lu.save", folder, (int)length, file_name,
                     blocks->states[0]);
        } else {
            snprintf(destination, 1023, "%s/%.*s.%lu.save", folder, (int)length, file_name,
                     blocks->states[0]);
        }
        fprintf(stdout, "save result to file: %s\n", destination);

        f = fopen(destination, "w");

        if (NULL == f) {
            fprintf(stderr, "failed to open file: %s\n", strerror(errno));
            exit(EXIT_FAILURE);
        }
    }

    if (fprintf(f, "grid:  %hhu\n", blocks->grid_side) < 0) {
        fprintf(stderr, "failed to write: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    if (fprintf(f, "block: %hhu\n", blocks->block_side) < 0) {
        fprintf(stderr, "failed to write: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    const uint64_t ends = blocks->grid_side * blocks->grid_side * blocks->block_side *
                          blocks->block_side;
    if (box_drawing == true) {
        for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
            if (gy == 0)
                fputs(u8"\t┏", f);
            else
                fputs(u8"\t┣", f);
            for (int i = 0; i < blocks->grid_side; i++) {
                for (int j = 0; j < (blocks->block_side - 1); j++) {
                    fputs(u8"━━━", f);
                }
                if (i < blocks->grid_side - 1) {
                    if (gy == 0)
                        fputs(u8"━━┳", f);
                    else
                        fputs(u8"━━╋", f);
                } else {
                    if (gy == 0)
                        fputs(u8"━━┓\n", f);
                    else
                        fputs(u8"━━┫\n", f);
                }
            }
            for (uint32_t y = 0; y < blocks->block_side; y++) {
                fputs(u8"\t┃", f);
                for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
                    for (uint32_t x = 0; x < blocks->block_side; x++) {
                        const uint64_t state      = *blk_at(blocks, gx, gy, x, y);
                        const uint64_t real_value = bitfield_to_integer(state);

                        if (fprintf(f, "%2lu ", real_value) < 0) {
                            fprintf(stderr, "failed to write: %s\n", strerror(errno));
                            exit(EXIT_FAILURE);
                        }
                    }
                    fputs(u8"\b┃", f);
                }
                fputs("\n", f);
            }
        }
        fputs(u8"\t┗", f);
        for (int i = 0; i < blocks->grid_side; i++) {
            for (int j = 0; j < blocks->block_side - 1; j++) {
                fputs(u8"━━━", f);
            }
            if (i < blocks->grid_side - 1)
                fputs(u8"━━┻", f);
            else
                fputs(u8"━━┛\n", f);
        }
    } else {
        for (uint32_t gx = 0; gx < blocks->grid_side; gx++) {
            for (uint32_t gy = 0; gy < blocks->grid_side; gy++) {
                for (uint32_t x = 0; x < blocks->block_side; x++) {
                    for (uint32_t y = 0; y < blocks->block_side; y++) {
                        const uint64_t state      = *blk_at(blocks, gx, gy, x, y);
                        const uint64_t real_value = bitfield_to_integer(state);

                        if (fprintf(f, "%2lu ", real_value) < 0) {
                            fprintf(stderr, "failed to write: %s\n", strerror(errno));
                            exit(EXIT_FAILURE);
                        }
                    }
                }
                fputs("\n", f);
            }
        }
    }

    if (folder != NULL) {
        fprintf(stdout, "saved successfully %lu states\n", ends);
        fclose(f);
    }
}
