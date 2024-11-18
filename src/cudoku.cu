#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <vector>

#include <algorithm>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

#include <unistd.h>
#include <omp.h>

const int NUM_BLOCKS = 20000;

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -f  --file  <FILENAME>     Path to the input file\n");
    printf("  -s  --size  <INT>          The size of one side of the input board\n");
    printf("  -?  --help                 This message\n");
}

// __global__ void solveBoard(char* board)

int main(int argc, char** argv){
    
    int opt;
    static struct option options[] = {
        {"help",     0, 0,  '?'},
        {"file",     1, 0,  'f'},
        {"size",     1, 0,  's'},
        {0 ,0, 0, 0}
    };

    int board_size;
    std::string board_filename;

    while ((opt = getopt_long(argc, argv, "f:s:?", options, NULL)) != EOF) {
        switch (opt) {
            case 'f':
                board_filename = optarg;
                break;
            case 's':
                board_size = atoi(optarg);
                break;
            case '?':
            default:
                usage(argv[0]);
                return 1;
        }
    }

    std::ifstream fin(board_filename);
    std::vector<char> first_board(board_size * board_size);

    for(int i = 0; i < board_size * board_size; ++i){
        int tmp;
        fin >> tmp;
        first_board[i] = (char)tmp;
    }

    char* boards;
    

    cudaMalloc(&boards, sizeof(char) * board_size * board_size * NUM_BLOCKS);
    cudaMemcpy(boards, &first_board, sizeof(char) * board_size * board_size, cudaMemcpyHostToDevice);

    // kernel launch
}
/**

1 2 3 4 5 6

*/