#include "utils.h"
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>

#define linewidth (80)

void
print_progress(const size_t iter, const size_t max_iter)
{
    const double completed = (double)iter / ((double)max_iter);

    char green_bar_buffer[(linewidth / 2) * 3 + 1] = { 0 };
    char white_bar_buffer[(linewidth / 2) * 3 + 1] = { 0 };

    const char bar[4]      = u8"─";
    const char bar_bold[4] = u8"━";

    const size_t nb_completed = (size_t)(completed * 40);

    for (size_t i = 0; i < nb_completed * 3; i += 3) {
        green_bar_buffer[i]     = bar_bold[0];
        green_bar_buffer[i + 1] = bar_bold[1];
        green_bar_buffer[i + 2] = bar_bold[2];
    }

    for (size_t i = 0; i < (linewidth / 2 - nb_completed) * 3; i += 3) {
        white_bar_buffer[i]     = bar[0];
        white_bar_buffer[i + 1] = bar[1];
        white_bar_buffer[i + 2] = bar[2];
    }

    fprintf(stdout, "\r%9lu / %9lu       \x1b[32m%s\x1b[0m%s % 7.2f %%", iter, max_iter, green_bar_buffer, white_bar_buffer, completed * 100);
    fflush(stdout);
}

void
safe_free(wfc_blocks_ptr blk)
{
    if (blk == NULL)
        return;

    if (blk->states != NULL) {
        free(blk->states);
        blk->states = NULL;
    }

    free(blk);
}

wfc_blocks *
safe_malloc(uint64_t blkcnt)
{
#define CHECK_PTR(ret)          \
    do {                        \
        if (ret == NULL) {      \
            perror("malloc");   \
            exit(EXIT_FAILURE); \
        }                       \
    } while (0)

    wfc_blocks *ret = malloc(sizeof(*ret));
    CHECK_PTR(ret);

    ret->states = malloc(blkcnt * sizeof(*ret->states));
    CHECK_PTR(ret->states);

    return ret;
}
