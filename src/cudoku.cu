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
__device__ bool solution_found;
__device__ int solution_idx;

// both parameters are num_blocks size arrays
__global__ void solveBoard(char* boards, int* statuses, int board_size){
    __shared__ int progress_flag;
    __shared__ int done_flag;
    __shared__ int error_flag;
    __shared__ int min_possibility_count;
    __shared__ int min_possibility_thread_idx_x;
    __shared__ int min_possibility_thread_idx_y;

    char* board = &boards[sizeof(char) * board_size * board_size * blockIdx.x];
    int status = statuses[sizeof(char) * blockIdx.x];

    int board_dim = board_size * board_size;
    int inner_board_dim = sqrtf(board_size);

    // status = 0 if idle, 1 if running, 2 if done?
    if(status == 1){
        if(threadIdx.x < board_size && threadIdx.y < board_size){
            int possibilities_count = 0;
            char possibles = 0; // a bitmask for 1-board_size all possible

            // First thread in each block should reset the reductions.
            if( threadIdx.x == 0 && threadIdx.y == 0){
                progress_flag = 1;
                error_flag = 0;
                min_possibility_count = board_size;
                min_possibility_thread_idx_x = board_size;
                min_possibility_thread_idx_y = board_size;
            }

            while(progress_flag){
                if(threadIdx.x == 0 && threadIdx.y == 0){
                    progress_flag = 0;
                    done_flag = 1;
                }
                __syncthreads();

                // Get cell value and check if it has been filled
                int board_value = board[threadIdx.x * board_size + threadIdx.y];
                if(board_value){
                    break;
                }

                done_flag = 0;
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

                // If unfilled cell has no possibilities, then error
                if(possibilities_count == 0){
                    error_flag = 1;
                }
                __syncthreads();
            }

            // Flag is set only when every cell has been filled
            if(done_flag){
                solution_found = true;
                solution_idx = sizeof(char) * board_dim * blockIdx.x;
                return;
            }

            // If error flag is set, set status to idle
            if(error_flag){
                statuses[sizeof(char) * blockIdx.x] = 0;
                return;
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

            if(possibilities_count == min_possibility_count){
                atomicMin(&min_possibility_thread_idx_y, threadIdx.y);
            }
            __syncthreads();

            // Fork on possibilities of cell with mininum possibilities
            if(threadIdx.x == min_possibility_thread_idx_x && threadIdx.y == min_possibility_thread_idx_y){
                int next_block_index = blockIdx.x;

                for(int i = 0; i < board_size; ++i){
                    if(!(possibles & 1)){
                        int possible_value = i + 1;

                        if(next_block_index != blockIdx.x){
                            // next_block == 0 ? 1 : 0, i.e. atomic compare a block to 0 (idle) and set to 1 (working)
                            while(next_block_index < NUM_BLOCKS && atomicCAS(statuses + next_block_index, 0, 1) == 0)
                                next_block_index++;
                        }

                        if(next_block_index <= NUM_BLOCKS){
                            printf("Scheduling %d, %d to take a value of %d\n", threadIdx.x, threadIdx.y, possible_value);
                            char* new_board = &boards[sizeof(char) * board_dim * next_block_index];
                            memcpy(new_board, board, sizeof(char) * board_dim);
                            new_board[ threadIdx.x * board_size + threadIdx.y] = possible_value;
                        }
                        else{
                            break;
                        }
                    }
                    possibles >>= 1;
                }
            }
        }
    }
}

int solveBoardHost(std::vector<char> first_board){

    int board_size = first_board.size();
    char* boards;
    int* statuses;
    int status = 1;


    cudaMalloc(&boards, sizeof(char) * board_size * NUM_BLOCKS);
    cudaMalloc(&statuses, sizeof(int) * NUM_BLOCKS);

    cudaMemcpy(boards, first_board.data(), sizeof(char) * board_size, cudaMemcpyHostToDevice);
    cudaMemcpy(statuses, &status, sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(9,9);
    dim3 gridDim(NUM_BLOCKS);

    solveBoard<<<gridDim, blockDim>>>(boards, statuses, 9);

    cudaDeviceSynchronize();

}
