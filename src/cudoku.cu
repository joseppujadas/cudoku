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

// both parameters are num_blocks size arrays
__global__ void solveBoard(char* boards, char* statuses, int board_size){
    __shared__ int progress_flag;
    __shared__ int min_possibility_count;
    __shared__ int min_possibility_thread_idx_x;
    __shared__ int min_possibility_thread_idx_y;

    char* board = &boards[sizeof(char) * board_size * board_size * blockIdx.x];
    char status = statuses[sizeof(char) * blockIdx.x];

    int board_dim = board_size * board_size;
    int inner_board_dim = sqrtf(board_size);

    // status = 0 if idle, 1 if running, 2 if done?
    if(status == 1){
        if(threadIdx.x < board_size && threadIdx.y < board_size){
            int possibilities_count = 0;
            char possibles = 0; // a bitmask for 1-board_size all possible
            progress_flag = 1;

            if(blockIdx.x == 0 && blockIdx.y == 0){
                min_possibility_count = board_size;
                min_possibility_thread_idx_x = board_size;
                min_possibility_thread_idx_y = board_size;
            }

            while(progress_flag){
                progress_flag = 0;
                __syncthreads();

                int board_value = board[threadIdx.x * board_size + threadIdx.y];
                if(board_value){
                    return;
                }

                possibles = 0;
                for(int i = 0; i < board_size; ++i){
                    // Check the current row: (threadIdx.x, i)
                    int row_value = board[threadIdx.x * board_size + i];
                    if(row_value){
                        possibles |= (1 << (row_value-1));
                    }

                    // Current column: (i, threadIdx.y)
                    int col_value = board[i * board_size + threadIdx.y];
                    if(col_value){
                        possibles |= (1 << (col_value-1));
                    }

                }

                int inner_board_x = threadIdx.x - ( threadIdx.x % inner_board_dim);
                int inner_board_y = threadIdx.y - ( threadIdx.y % inner_board_dim);

                // check 3x3 subboard
                for(int i = inner_board_x; i < inner_board_x + inner_board_dim; ++i ){
                    for(int j = inner_board_y; j < inner_board_y + inner_board_dim; ++j ){
                        int inner_board_value = board[i * board_size + j];
                        if(inner_board_value){
                            possibles |= 1 << (inner_board_value-1);
                        }
                    }
                }

                // Find Deterministic updates first
                possibilities_count = 0;
                int temp = possibles;
                int update = 0;
                for(int i = 0; i < board_size; ++i){
                    if(!(temp & 1)){
                        possibilities_count += 1;
                        update = i + 1;
                    }
                    temp >>= 1;
                }

                // Deterministic Progress can be made
                if(possibilities_count == 1){
                    board[threadIdx.x * board_size + threadIdx.y] = update;
                    progress_flag = 1;
                }
                __syncthreads();
            }

            // No Deterministic Progress can be made in any cell.
            // First, find cell with minimum number of possibilities
            if(possibilities_count != 0){
                atomicMin(&min_possibility_count, possibilities_count);
            }
            __syncthreads();

            if(possibilities_count == min_possibility_count){
                atomicMin(&min_possibility_thread_idx_x, threadIdx.x);
            }
            __syncthreads();

            if(possibilities_count == min_possibility_count && threadIdx.x == min_possibility_thread_idx_x){
                atomicMin(&min_possibility_thread_idx_y, threadIdx.y);
            }
            __syncthreads();

            // Fork on possibilities of cell with mininum possibilities
            if(threadIdx.x == min_possibility_thread_idx_x && threadIdx.y == min_possibility_thread_idx_y){
                for(int i = 0; i < board_size; ++i){
                    if(!(possibles & 1)){
                        int possible_value = i + 1;
                        // Fork on possible_value
                    }
                    possibles >>= 1;
                }
            }
        }
    }
}

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
    int tmp;

    for(int i = 0; i < board_size * board_size; ++i){
        fin >> tmp;
        first_board[i] = (char)tmp;
    }

    char* boards;
    char* statuses;
    int status = 1;


    cudaMalloc(&boards, sizeof(char) * board_size * board_size * NUM_BLOCKS);
    cudaMalloc(&statuses, sizeof(char) * NUM_BLOCKS);

    cudaMemcpy(boards, first_board.data(), sizeof(char) * board_size * board_size, cudaMemcpyHostToDevice);
    cudaMemcpy(statuses, &status, sizeof(char), cudaMemcpyHostToDevice);

    dim3 blockDim(9,9);
    dim3 gridDim(NUM_BLOCKS);

    solveBoard<<<gridDim, blockDim>>>(boards, statuses, 9);

    cudaDeviceSynchronize();

    // kernel launch
}
/**

1 2 3 4 5 6

*/
