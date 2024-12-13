#include <vector>
#include <chrono>
#include <cstdio>

#include "cudoku.h"
#include "util.h"

const int NUM_BLOCKS = 1000;

/**
 * @brief Solves the board given at the first index of boards. Leaves
 * solved board in the device memory and sets solution_idx to indicate
 * where to find it to controller kernel.
 * 
 * @param boards Contiguous board memory from the controller.
 * @param statuses Contiguous status memory from the controller. Block i will
 * have statuses[i] = 1 if it is working and 0 if not and it can be scheduled
 * by another block.
 * @param[out] solution_idx The index in board memory where a solution was found.
 */
__global__ void solveBoard(char* boards, int* statuses, int* solution_idx){

    
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

    // Only work if status is running (1)
    if(status == 1 && threadIdx.x < board_size && threadIdx.y < board_size){
        int possible_ct = 0;
        int possibles = 0; // a bitmask for 1-board_size all possible

        // First thread in each block should reset the reductions.
        if( threadIdx.x == 0 && threadIdx.y == 0){
            progress_flag = 1;
            error_flag = 0;
            done_flag = 0;
            min_possibility_count = board_size;
            min_possibility_thread_idx = board_size*board_size;
        }
        __syncthreads();

        // Loop while progress can be made, no solution, no error.
        while(progress_flag && !done_flag && !error_flag){

            // Reset shared variables
            if( threadIdx.x == 0 && threadIdx.y == 0){
                progress_flag = false;
                error_flag = false;
                done_flag = true;
            }
            
            // Reset possibility bitmasks
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
                // Old & mask indicates the bit was masked BEFORE this update, so there
                // is a conflict and an error occurs.

                int old = atomicOr(&row_possibles[threadIdx.x], mask);
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
                // If here, some cell is still 0.
                done_flag = false;

                // Generate unique possibility set.
                possibles = row_possibles[ threadIdx.x ];
                possibles |= col_possibles[ threadIdx.y ];
                possibles |= inner_possibles[inner_idx];

                int last_possible = 0;
                possible_ct = 0;

                // Count possibilities
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

        // If error flag is set, set status to idle for rescheduling.
        if(error_flag){
            if( threadIdx.x + threadIdx.y == 0)
                statuses[blockIdx.x] = 0;
            return;
        }

        // Flag is set only when every cell has been filled. Set
        // solution index so kernels stop launching.
        if(done_flag){
            if(threadIdx.x + threadIdx.y == 0){
                *solution_idx = blockIdx.x * board_dim;
            }
            return;
        }

        // No Deterministic Progress can be made in any cell.
        // First, find cell with minimum number of possibilities
        if(possible_ct != 0){
            atomicMin(&min_possibility_count, possible_ct);
        }

        // Then find minimum cell index of those to update (arbitrary but fixed choice).
        if(possible_ct == min_possibility_count){
            atomicMin(&min_possibility_thread_idx, threadIdx.x * board_size + threadIdx.y);
        }

        __syncthreads();

        // Fork on possibilities of cell with mininum possibilities.
        if(min_possibility_thread_idx == ( threadIdx.x * board_size + threadIdx.y)){
            int next_block_index = blockIdx.x;

            for(int possible = 1; possible < board_size+1; ++possible){
                if(!(possibles & (1 << (possible - 1)))){

                    if(next_block_index != blockIdx.x){
                        // Atomic CAS ensures no two blocks schedule the same block.
                        while(next_block_index < NUM_BLOCKS and atomicCAS(&statuses[next_block_index], 0, 1) == 1)
                            next_block_index++;
                    }

                    // Find new board address, copy board data into it.
                    char* new_board = &boards[next_block_index * board_size * board_size];
                    if(next_block_index != blockIdx.x){
                        memcpy(new_board, board, sizeof(char) * board_dim);
                    }

                    // Update with guessed value.
                    new_board[ threadIdx.x * board_size + threadIdx.y] = possible;
                    next_block_index++;
                }
            }
        }
    }
}

/**
 * @brief Controller kernel for solving boards. Solves all boards given in device
 * memory and copies their solutions back to original location.
 * 
 * @param all_boards Pointer to memory containing contiguous board memory. Each
 * board should be board_size*board_size contiguous characters, where board_size is
 * a perfect square. Each board should only contain the numbers
 * @param num_trials The number of boards in all_boards.
 * @param board_size The width/height of each board, a perfect square.
 * @param boards_device A pointer to the device memory used for actually solving
 * each board. Will be used by the solveBoard kernel for every board. Should be
 * sizeof(char) * num_trials * board_size * board_size bytes.
 * @param statuses A pointer to the device memory used for status indicators. Should
 * be sizeof(int) * num_trials bytes.
 * @param solution_idx A pointer to a single integer used for indicating the solution
 * by the solve board kernel.
 */
__global__ void solveBoardKernel(char* all_boards, int num_trials, int board_size,
    char* boards_device, int* statuses, int* solution_idx){

    dim3 gridDim(NUM_BLOCKS);
    dim3 blockDim(board_size, board_size);
    int status = 1;
    int shared_memory_req = sizeof(int) * board_size * 3;

    for(int i = 0; i < num_trials; i++){
       
        char* board = all_boards + (i * board_size * board_size);
        if(board_size > 9){
            memset(statuses, 0, sizeof(int) * NUM_BLOCKS);
        }
        memcpy(boards_device, board, board_size * board_size);
        memcpy(statuses, &status, sizeof(int));

        *solution_idx = 1;

        while(*solution_idx == 1){
             
            solveBoard<<<gridDim, blockDim, shared_memory_req>>>(
                boards_device, statuses, solution_idx
            );
            
            cudaDeviceSynchronize();
        }
        
        memcpy(board, boards_device + *solution_idx, board_size * board_size);
    }
}

double solveBoardHost(std::vector<std::vector<char>> boards){

    int num_trials = boards.size();
    int board_size = boards[0].size();

    char* all_boards;
    char* boards_device;
    int* statuses;
    int* solution_idx_device;

    // Allocate and initialize global memory
    cudaMalloc(&all_boards, sizeof(char) * board_size * num_trials);
    cudaMalloc(&boards_device, sizeof(char) * board_size * NUM_BLOCKS);
    cudaMalloc(&statuses, sizeof(int) * NUM_BLOCKS);
    cudaMalloc(&solution_idx_device, sizeof(int));

    // Space to copy solutions back to host.
    std::vector<std::vector<char>> solutions(num_trials);

    // Copy all boards to host to amortize memory operations
    for(int i = 0; i < num_trials; ++i){
        solutions[i].resize(board_size);

        char* board = all_boards + (i * board_size);
        cudaMemcpy(board, boards[i].data(), board_size, cudaMemcpyHostToDevice);
    }

    const auto compute_start = std::chrono::steady_clock::now();

    solveBoardKernel<<<1, 1>>>(
        all_boards, num_trials, sqrt(board_size), boards_device,
        statuses, solution_idx_device
    );

    // Necessary to ensure solution is actually found, rather than just timing the kernel launch.
    cudaDeviceSynchronize();
    
    // Copy solutions back to host.
    for(int i = 0; i < num_trials; ++i){
        char* board = all_boards + (i * board_size);
        cudaMemcpy(solutions[i].data(), board, board_size, cudaMemcpyDeviceToHost);
    }

    const auto compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();

    cudaDeviceSynchronize();

    // Check correctness
    for(int i = 0; i < num_trials; ++i){
        verifySolve(boards[i], solutions[i]);
    }

    return compute_time;
}
