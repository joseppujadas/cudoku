#include <chrono>
#include <cmath>
#include <vector>

#include "util.h"

std::vector<char> solve(std::vector<char> board){

    int board_size = sqrt(board.size());
    int inner_board_size = sqrt(board_size);

    bool progress = true;
    bool done = false;
    bool error = false;

    int min_possible_set = 0;
    int min_possible_ct = board_size;
    int min_possible_idx = board_size * board_size;

    // Loop as long as deterministic progress can be made.
    while(progress && !done && !error){
        progress = false;
        done = true;
        error = false;

        std::vector<int> row_possibles(board_size);
        std::vector<int> col_possibles(board_size);
        std::vector<int> inner_possibles(board_size);

        // Calculate possible values for each row and column once, then each cell checks later.
        // 0 in bitmask = still possible, 1 = not possible.
        for(int i = 0; i < board_size; ++i){
            for(int j = 0; j < board_size; ++j){

                char val = board[i * board_size + j];
                if(val){
                    int mask = 1 << (val - 1);
                    int inner_row = i / inner_board_size;
                    int inner_col = j / inner_board_size;
                    int inner_idx = inner_row * inner_board_size + inner_col;

                    row_possibles[i] |= mask;
                    col_possibles[j] |= mask;
                    inner_possibles[inner_idx] |= mask;
                }
            }
        }
        if(error)
            return {};

        // Check every cell. Make deterministic updates if possible, return if no
        // values left anywhere (i.e. wrong guess somewhere).
        for(int i = 0; i < board_size; ++i){
            for(int j = 0; j < board_size; ++j){

                char val = board[i * board_size + j];
            
                if(!val){
                    done = false;

                    int inner_row = i / inner_board_size;
                    int inner_col = j / inner_board_size;
                    int inner_idx = inner_row * inner_board_size + inner_col;

                    int possibles = row_possibles[i] | col_possibles[j] | inner_possibles[inner_idx];

                    int last_possible = 0;
                    int possible_ct = 0;

                    for(int possible = 1; possible < board_size + 1; ++possible){
                        if(!(possibles & 1)){
                            last_possible = possible;
                            possible_ct += 1;
                        }
                        possibles >>= 1;
                    }
                    // No possible values --> this solution is wrong somewhere.
                    if(possible_ct == 0){
                        error = true;
                    }

                    // Exactly one possible value --> deterministic update.
                    else if(possible_ct == 1){
                        board[i * board_size + j] = last_possible;      

                        int mask = 1 << (last_possible-1);

                        row_possibles[i] |= mask;
                        col_possibles[j] |= mask;   
                        inner_possibles[inner_idx] |= mask;               
                        progress = true;
                    }

                    // > 1 possibility --> update min possibility vars if necesssary.
                    else if(possible_ct < min_possible_ct){
                        #pragma omp critical
                        {
                            min_possible_ct = possible_ct;
                            min_possible_set = row_possibles[i] | col_possibles[j] | inner_possibles[inner_idx];
                            min_possible_idx = i * board_size + j;
                        }
                    }
                }
            }
        }
    }
    

    if(error)
        return {};

    if(done)
        return board;

    // After no more deterministic progress, fork on the smallest possibility set.    
    for(int possible = 1; possible < board_size + 1; ++possible){
        if(!(min_possible_set & 1)){
            board[min_possible_idx] = possible;

            // Fork with updated vars
            std::vector<char> fork_result = solve(board);
            if(fork_result.size()){
                return fork_result;
            }
        }
        min_possible_set >>= 1;
    }
    
    return {};
}

double solveBoardsSeq(std::vector<std::vector<char>> boards){

    double total_time = 0;
    for(const auto& board : boards){

        const auto compute_start = std::chrono::steady_clock::now();

        std::vector<char> solution = solve(board);

        const auto compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();

        if(!verifySolve(board, solution)){
            printBoard(solution);
        }
        
        total_time += compute_time;
    }

    return total_time;
}