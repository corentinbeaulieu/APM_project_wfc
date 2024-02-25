#define _GNU_SOURCE

#include "wfc.h"

#include <string.h>
#include <strings.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/stat.h>
#include <stdio.h>

_Noreturn static inline void
print_help(const char *exec)
{
    fprintf(stdout, "usage: %s [-hb] [-o folder/] [-l solver] [-p count] [-s seeds...] <path/to/file.data>\n", exec);
    puts("  -h          print this help message.");
    puts("  -o folder   output folder to save solutions. adds the seed to the data file name.");
    puts("  -b          draw a box using utf8 box drawing caracter on output (default: off).");
    puts("  -p count    number of seeds that can be processed in parallel");
    puts("  -t count    number of teams (OMP target)");
    puts("  -s seeds    seeds to use. can an integer or a range: `from-to`.");

    fprintf(stdout, "  -l solver   solver to use. possible values are:");
    for (unsigned long i = 0; i < sizeof(solvers) / sizeof(wfc_solver); i += 1) {
        (i == 0) ? fprintf(stdout, " [%s]", solvers[i].name)
                 : fprintf(stdout, ", %s", solvers[i].name);
    }
    fputc('\n', stdout);

    exit(EXIT_SUCCESS);
}

static inline uint32_t
to_u32(const char *arg, char **end)
{
    const long value = strtol(arg, end, 10);
    if (value < 0) {
        fprintf(stderr, "negative seeds are not possible: %ld\n", value);
        exit(EXIT_FAILURE);
    }
    return (uint32_t)value;
}

static inline bool
str_ends_with(const char str[], const char suffix[])
{
    size_t lenstr    = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix > lenstr)
        return 0;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
}

wfc_args
wfc_parse_args(int argc, char **argv)
{
    int opt;
    seeds_info seeds                                                       = { .count = 100, .start = 0 };
    const char *output                                                     = NULL;
    bool box_drawing                                                       = false;
    uint64_t parallel                                                      = 1;
    uint64_t teams                                                         = 1;
    bool (*solver)(wfc_blocks_ptr, wfc_args, wfc_blocks_ptr *, uint64_t *) = NULL;
    char *end                                                              = NULL;

    while ((opt = getopt(argc, argv, "hbs:o:l:p:t:")) != -1) {
        switch (opt) {
        case 's': {
            const uint32_t from = to_u32(optarg, &end);
            if ('\0' == *end) {
                seeds = (seeds_info){ .count = 1, .start = from };
            } else if ('-' == *end) {
                const uint32_t to = to_u32(end + 1, &end);
                if (*end != '\0') {
                    fprintf(stderr, "failed to get the upper value of the range\n");
                    exit(EXIT_FAILURE);
                } else if (from >= to) {
                    fprintf(stderr, "invalid range: %u >= %u\n", from, to);
                    exit(EXIT_FAILURE);
                }
                seeds.start = from;
                seeds.count = to - from + 1;
            } else {
                fprintf(stderr, "invalid range delimiter: '%c'\n", *end);
                exit(EXIT_FAILURE);
            }
            break;
        }

        case 'l': {
            for (uint64_t i = 0; i < sizeof(solvers) / sizeof(wfc_solver); i += 1) {
                if (0 == strcasecmp(optarg, solvers[i].name)) {
                    solver = solvers[i].function;
                }
            }
            if (NULL == solver) {
                fprintf(stderr, "unknown solver `%s`\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        }

        case 'p': {
            if ((parallel = to_u32(optarg, &end)) <= 0) {
                fputs("you must at least process one seed at a time...", stderr);
                fprintf(stderr, "invalid p argument %lu\n", parallel);
                exit(EXIT_FAILURE);
            } else if ('\0' != *end) {
                fprintf(stderr, "invalid integer: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        }

        case 't': {
            if ((teams = to_u32(optarg, &end)) <= 0) {
                fputs("you must at least have one team of threads...", stderr);
                fprintf(stderr, "invalid t argument %lu\n", parallel);
                exit(EXIT_FAILURE);
            } else if ('\0' != *end) {
                fprintf(stderr, "invalid integer: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        }

        case 'o': {
            struct stat sb;
            if (stat(optarg, &sb) == 0 && S_ISDIR(sb.st_mode)) {
                output = optarg;
                break;
            } else {
                fprintf(stderr, "folder `%s` doesn't exist\n", optarg);
                exit(EXIT_FAILURE);
            }
        }

        case 'b': {
            box_drawing = true;
            break;
        }
        case 'h':
        default: print_help(argv[0]);
        }
    }

    if (optind >= argc) {
        print_help(argv[0]);
    } else if (!str_ends_with(argv[optind], ".data")) {
        fprintf(stderr, "expected the suffix `.data` for the data file: %s\n", argv[optind]);
        exit(EXIT_FAILURE);
    }

    return (wfc_args){
        .data_file     = argv[optind],
        .seeds         = seeds,
        .output_folder = output,
        .parallel      = parallel,
        .teams         = teams,
        .solver        = (NULL == solver) ? solvers[0].function : solver,
        .box_drawing   = box_drawing,
    };
}
