#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

#include <getopt.h>
#include <omp.h>

#include "sudoku_omp.h"
#include "sudoku_seq.h"
#include "cudoku.h"
#include "util.h"

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -f  --file     <FILENAME>   Path to the input file.\n");
    printf("  -n  --trials   <INT>        The number of boards to read from the input file and solve.\n");
    printf("  -s  --size     <INT>        The size of one side of the input board. Must be a perfect square.\n");
    printf("  -t  --threads  <INT>        The number of OMP threads to use (1 by default)\n");
    printf("  -?  --help                  This message\n");
}

int main(int argc, char** argv){

    int opt;
    static struct option options[] = {
        {"help",     no_argument,       0,  '?'},
        {"file",     required_argument, 0,  'f'},
        {"size",     required_argument, 0,  's'},
        {"trials",   required_argument, 0,  'n'},
        {"threads",  optional_argument, 0,  't'},

        {0 ,0, 0, 0}
    };

    int board_size = 0;
    int trials = 1;
    int num_threads = 1;
    std::string board_filename;

    // Parse arguments
    while ((opt = getopt_long(argc, argv, "f:s:n:t:?", options, NULL)) != EOF) {
        switch (opt) {
            case 'f':
                board_filename = optarg;
                break;
            case 's':
                board_size = atoi(optarg);
                break;
            case 'n':
                trials = atoi(optarg);
                break;
            case 't':
                num_threads = atoi(optarg);
                break;
            case '?':
            default:
                usage(argv[0]);
                return 1;
        }
    }
    if(board_filename == "" or board_size == 0){
        usage(argv[0]);
        exit(1);
    }

    omp_set_num_threads(num_threads);

    // Read board from input file
    std::ifstream fin(board_filename);
    std::vector<std::vector<char>> boards(trials);
    int tmp;

    for(int trial = 0; trial < trials; ++trial){
        boards[trial].resize(board_size * board_size);
        for(int i = 0; i < board_size * board_size; ++i){
            fin >> tmp;
            boards[trial][i] = (char)tmp;
        }
    }
    
    printf("Sequential: %lf\n", solveBoardsSeq(boards));
    printf("OpenMP: %lf\n", solveBoardsOMP(boards));
    printf("CUDA: %lf\n", solveBoardHost(boards));

    return 1;
}
