#define _GNU_SOURCE

#include "wfc.h"

#include <stdio.h>
#include <omp.h>
#include <stdatomic.h>

int
main(int argc, char *argv[])
{
    omp_set_dynamic(false);

    wfc_args args             = wfc_parse_args(argc, argv);
    const wfc_blocks_ptr init = wfc_load(0, args.data_file);

    _Atomic unsigned char quit  = false;
    _Atomic uint64_t iterations = 0;
    wfc_blocks_ptr blocks       = NULL;

    const uint64_t num_threads    = args.parallel;
    const uint64_t max_iterations = count_seeds(args.seeds);
    const double start            = omp_get_wtime();

#pragma omp parallel for num_threads(num_threads)
    for (size_t iter = 0; iter < max_iterations; iter++) {
        if (atomic_load(&quit) == true)
            continue;

        bool has_next_seed = false;
        uint64_t next_seed = 0;

#pragma omp critical
        {
            has_next_seed = try_next_seed(&args.seeds, &next_seed);
        }

        if (has_next_seed == false) {
            atomic_fetch_or(&quit, true);
            fprintf(stderr, "no more seed to try\n");
            continue;
        }

        wfc_clone_into(&blocks, next_seed, init);
        const bool solved = args.solver(blocks);
        atomic_fetch_add(&iterations, 1);

        if (solved == true && atomic_fetch_or(&quit, solved) == false) {
            if (args.output_folder != NULL)
                fputc('\n', stdout);
            else
                fputs("\nsuccess with result:\n", stdout);

            wfc_save_into(blocks, args.data_file, args.output_folder);
        }
    }

    fprintf(stdout, "\r%.2f%% -> %.2fs",
            ((double)(atomic_load(&iterations)) / (double)(max_iterations)) * 100.0,
            omp_get_wtime() - start);
    return 0;
}
