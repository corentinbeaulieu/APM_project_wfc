#define _GNU_SOURCE

#include "wfc.h"
#include "utils.h"

#include <stdio.h>
#include <omp.h>
#include <stdatomic.h>
#include <string.h>

int
main(int argc, char *argv[])
{
    omp_set_dynamic(false);

    wfc_args args       = wfc_parse_args(argc, argv);
    wfc_blocks_ptr init = wfc_load(0, args.data_file);
    wfc_blocks_ptr res  = NULL;

    _Atomic uint64_t iterations = 0;

    const uint64_t max_iterations = args.seeds.count;
    const double start            = omp_get_wtime();

    bool solved = args.solver(init, args, &res);

    const double stop = omp_get_wtime();

    fflush(stdout);
    if (solved && res) {
        if (args.output_folder != NULL)
            fputs("\n\n", stdout);
        else
            fputs("\n\nsuccess with result:\n", stdout);

        printf("seed: %lu\n", res->states[0]);

        wfc_save_into(res, args.data_file, args.output_folder, args.box_drawing);
    } else {
        fprintf(stderr, "\n\n\x1b[1mNo solution found\x1b[0m\n");
    }

    fprintf(stdout, "%9lu / %9lu = %6.2f%% ➜ %.16f s\n",
            atomic_load(&iterations), max_iterations,
            ((double)(atomic_load(&iterations)) / (double)(max_iterations)) * 100.0,
            stop - start);

    safe_free(init);
    if (res)
        safe_free(res);
    return 0;
}
