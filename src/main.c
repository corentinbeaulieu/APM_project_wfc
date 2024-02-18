#define _GNU_SOURCE

#include "wfc.h"

#include <stdio.h>
#include <omp.h>
#include <stdatomic.h>
#include <string.h>
#include <wchar.h>

#define linewidth (80)

static void
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

int
main(int argc, char *argv[])
{
    omp_set_dynamic(false);

    wfc_args args             = wfc_parse_args(argc, argv);
    const wfc_blocks_ptr init = wfc_load(0, args.data_file);

    _Atomic unsigned char quit  = false;
    _Atomic uint64_t iterations = 0;
    _Atomic int solving_thread  = 0;
    wfc_blocks_ptr blocks       = NULL;
    wfc_blocks_ptr final_result = NULL;

    const uint64_t num_threads    = args.parallel;
    const uint64_t max_iterations = count_seeds(args.seeds);
    const double start            = omp_get_wtime();

#pragma omp parallel num_threads(num_threads) firstprivate(blocks) shared(quit, iterations, init, final_result, solving_thread)
    {
#pragma omp for nowait
        for (size_t iter = 1; iter <= max_iterations; iter++) {
            if (atomic_load(&quit) == true) {
                continue;
            }

            bool has_next_seed = false;
            uint64_t next_seed = 0;

#pragma omp critical
            {
                has_next_seed = try_next_seed(&args.seeds, &next_seed);
            }

            if (has_next_seed == false) {
                continue;
            }

            wfc_clone_into(&blocks, next_seed, init);
            const bool solved = args.solver(blocks);
            atomic_fetch_add(&iterations, 1);

            if (solved == true && atomic_fetch_or(&quit, solved) == false) {
                atomic_store(&solving_thread, omp_get_thread_num());
                wfc_clone_into(&final_result, blocks->states[0], blocks);
            } else {
                print_progress(atomic_load(&iterations), max_iterations);
            }
        }
        if (blocks) {
            free(blocks);
            blocks = NULL;
        }
    }

    const double stop = omp_get_wtime();

    fflush(stdout);
    if (final_result) {
        if (args.output_folder != NULL)
            fputs("\n\n", stdout);
        else
            fputs("\n\nsuccess with result:\n", stdout);

        printf("thread %d:\nseed: %lu\n", atomic_load(&solving_thread), final_result->states[0]);

        wfc_save_into(final_result, args.data_file, args.output_folder, args.box_drawing);

        free(final_result);
    } else {
        fprintf(stderr, "\n\n\x1b[1mNo solution found\x1b[0m\n");
    }

    fprintf(stdout, "%9lu / %9lu = %6.2f%% ➜ %.16f s\n",
            atomic_load(&iterations), max_iterations,
            ((double)(atomic_load(&iterations)) / (double)(max_iterations)) * 100.0,
            stop - start);

    free(init);
    return 0;
}
