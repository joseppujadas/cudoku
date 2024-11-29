#include <vector>
#include <chrono>
#include <cstdio>

#include "cudoku.h"
#include "util.h"

const int NUM_BLOCKS = 50000;


__global__ void solveBoard(char* boards, int* statuses, int* solution_idx, int* solution_found){
    
    //.Static shared memory for block constants
    __shared__ int progress_flag;
    __shared__ int done_flag;
    __shared__ int error_flag;
    __shared__ int min_possibility_count;
    __shared__ int min_possibility_thread_idx;

    // Various size consants
    int board_size = blockDim.x;
    int board_dim = board_size * board_size;
    int inner_board_size = sqrtf(board_size);
    int inner_row = threadIdx.x / inner_board_size;
    int inner_col = threadIdx.y / inner_board_size;
    int inner_idx = inner_row * inner_board_size + inner_col;
    
    // Dynamic shared memory for possibility sets based on board size
    extern __shared__ int dynamic_shared_mem[];

    int* row_possibles = dynamic_shared_mem;
    int* col_possibles = &dynamic_shared_mem[board_size];
    int* inner_possibles = &col_possibles[board_size];

    char* board = &boards[board_size * board_size * blockIdx.x];
    int status = statuses[blockIdx.x];

    // status = 0 if idle, 1 if running, 2 if done?
    if(status == 1 && threadIdx.x < board_size && threadIdx.y < board_size){
        int possible_ct = 0;
        int possibles = 0; // a bitmask for 1-board_size all possible
        // char update = 0;

        // First thread in each block should reset the reductions.
        if( threadIdx.x == 0 && threadIdx.y == 0){
            progress_flag = 1;
            error_flag = 0;
            done_flag = 0;
            min_possibility_count = board_size;
            min_possibility_thread_idx = board_size*board_size;
        }
        __syncthreads();

        // Loop while progress can be made
        while(progress_flag && !done_flag && !error_flag){

            if( threadIdx.x == 0 && threadIdx.y == 0){
                progress_flag = false;
                error_flag = false;
                done_flag = true;
            }

            if(threadIdx.x == 0 and threadIdx.y < board_size){
                row_possibles[threadIdx.y] = 0;
                col_possibles[threadIdx.y] = 0;
                inner_possibles[threadIdx.y] = 0;
            }
            possible_ct = 0;
            __syncthreads();
            
            // Get cell value and check if it has been filled
            int val = board[ threadIdx.x * board_size + threadIdx.y]; 
            int mask = 1 << (val-1);

            // Generate row, column, inner board possibilities cooperatively.
            if(val){
                int old;

                // Old & mask indicates the bit was masked BEFORE this update, so there
                // is a conflict.
                old = atomicOr(&row_possibles[threadIdx.x], mask);
                if(old & mask){
                    error_flag = true;
                }

                old = atomicOr(&col_possibles[threadIdx.y], mask);
                if(old & mask){
                    error_flag = true;
                }

                old = atomicOr(&inner_possibles[inner_idx], mask);
                if(old & mask){
                    error_flag = true;
                }
            }
            __syncthreads();

            // Update deterministically if possible
            if(!val){
                
                done_flag = false;

                possibles = row_possibles[ threadIdx.x ];
                possibles |= col_possibles[ threadIdx.y ];
                possibles |= inner_possibles[inner_idx];

                int last_possible = 0;
                possible_ct = 0;

                for(int possible = 1; possible < board_size + 1; ++possible){
                    if(!(possibles & (1 << (possible - 1)))){
                        last_possible = possible;
                        possible_ct += 1;
                    }
                }
                // No possible values --> this solution is wrong somewhere.
                if(possible_ct == 0)
                    error_flag = true;
                
                // One possible value --> deterministic update.
                if(possible_ct == 1){
                    board[threadIdx.x * board_size + threadIdx.y] = last_possible;
                    progress_flag = true;
                }
            }
            __syncthreads();
        }

        // If error flag is set, set status to idle for rescheduling
        if(error_flag){
            if( threadIdx.x + threadIdx.y == 0)
                statuses[blockIdx.x] = 0;
            return;
        }

        // Flag is set only when every cell has been filled
        if(done_flag){
            if(threadIdx.x + threadIdx.y == 0){
                *solution_found = true;
                *solution_idx = blockIdx.x * board_dim;
            }
            return;
        }

        // No Deterministic Progress can be made in any cell.
        // First, find cell with minimum number of possibilities
        if(possible_ct != 0){
            atomicMin(&min_possibility_count, possible_ct);
        }
        __syncthreads();

        // Then find minimum cell index of those to update (arbitrary but fixed choice)
        if(possible_ct == min_possibility_count){
            atomicMin(&min_possibility_thread_idx, threadIdx.x * board_size + threadIdx.y);
        }

        __syncthreads();

        // Fork on possibilities of cell with mininum possibilities
        if(min_possibility_thread_idx == ( threadIdx.x * board_size + threadIdx.y)){
            int next_block_index = blockIdx.x;

            for(int possible = 1; possible < board_size+1; ++possible){
                if(!(possibles & (1 << (possible - 1)))){

                    if(next_block_index != blockIdx.x){
                        
                        // next_block == 0 ? 1 : 0, i.e. atomic compare a block to 0 (idle) and set to 1 (working)
                        while(next_block_index < NUM_BLOCKS && atomicCAS(&statuses[next_block_index], 0, 1) == 1)
                            next_block_index++;
                    }

                    if(next_block_index < NUM_BLOCKS){
                        char* new_board = &boards[next_block_index * board_size * board_size];
                        memcpy(new_board, board, sizeof(char) * board_dim);
                        new_board[ threadIdx.x * board_size + threadIdx.y] = possible;

                        next_block_index++;
                    }
                }
            }
        }
    }
}

double solveBoardHost(std::vector<std::vector<char>> boards){

    int board_size = boards[0].size();
    char* boards_device;
    int* statuses;
    int status = 1;

    int solution_found = 0;
    int* solution_found_device;
    int solution_idx;
    int* solution_idx_device;

    // Allocate and initialize global memory
    cudaMalloc(&boards_device, sizeof(char) * board_size * NUM_BLOCKS);
    cudaMalloc(&statuses, sizeof(int) * NUM_BLOCKS);
    cudaMalloc(&solution_found_device, sizeof(int));
    cudaMalloc(&solution_idx_device, sizeof(int));
    

    dim3 blockDim(sqrt(board_size),sqrt(board_size));
    dim3 gridDim(NUM_BLOCKS);

    int shared_memory_req = sizeof(int) * sqrt(board_size) * 3;

    std::vector<char> solution(board_size);
    double total_time = 0;
    for(const auto& board : boards){
        cudaMemset(statuses, 0, sizeof(int) * NUM_BLOCKS);
        cudaMemcpy(statuses, &status, sizeof(int), cudaMemcpyHostToDevice);
        solution_found = 0;
        cudaMemcpy(solution_found_device, &solution_found, sizeof(int), cudaMemcpyHostToDevice);

        
        cudaMemcpy(boards_device, board.data(), sizeof(char) * board_size, cudaMemcpyHostToDevice);

        // Call kernel in a loop to reschedule blocks until one finds a solution
        while(!solution_found){
            solveBoard<<<gridDim, blockDim, shared_memory_req>>>(
                boards_device, statuses, solution_idx_device, solution_found_device
            );
            
            const auto compute_start = std::chrono::steady_clock::now();
            cudaMemcpy(&solution_found, solution_found_device, sizeof(int), cudaMemcpyDeviceToHost);
            total_time += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
        }
        
        // Copy board data back to host
        cudaMemcpy(&solution_idx, solution_idx_device, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(solution.data(), boards_device + solution_idx, board_size, cudaMemcpyDeviceToHost);
        // const auto compute_time = 
        
        if(!verifySolve(board, solution)) printf("uh oh\n");

       
    }
    
    return total_time;
}
