#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>

#include <algorithm>
#include <fstream>
#include <cassert>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>

#include <unistd.h>
#include <omp.h>

std::vector<char> solveHelp(std::vector<char> board, std::vector<int> row_possibles, 
                            std::vector<int> col_possibles, std::vector<int> inner_possibles){
    int board_size = sqrt(board.size());
    int inner_board_size = sqrt(board_size);

    bool progress = true;
    bool done = false;
    int min_possible_set = 0;
    int min_possible_ct = board_size;
    int min_possible_row = 0;
    int min_possible_col = 0;

    // Loop as long as deterministic progress can be made.
    while(progress && !done){
        progress = false;
        done = true;
        
        // Check every cell. Make deterministic updates if possible, return if no
        // values left anywhere (i.e. wrong guess somewhere).
        for(int i = 0; i < board_size; ++i){
            for(int j = 0; j < board_size; ++j){
                char val = board[i * board_size + j];
                if(!val){
                    done = false;
                    int possibles = row_possibles[i] | col_possibles[j];

                    int inner_row = i / inner_board_size;
                    int inner_col = j / inner_board_size;
                    possibles |= inner_possibles[inner_row * inner_board_size + inner_col];

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
                    if(possible_ct == 0)
                        return {};
                    
                    // Exactly one possible value --> deterministic update.
                    else if(possible_ct == 1){
                        board[i * board_size + j] = last_possible;

                        int mask = 1 << (last_possible - 1);
                        row_possibles[i] |= mask;
                        col_possibles[j] |= mask;
                        inner_possibles[inner_row * inner_board_size + inner_col] |= mask;

                        progress = true;
                    }

                    // > 1 possibility --> update min possibility vars if necesssary.
                    else if(possible_ct < min_possible_ct){
                        min_possible_ct = possible_ct;
                        min_possible_set = row_possibles[i] | col_possibles[j] | inner_possibles[inner_row * inner_board_size + inner_col];
                        min_possible_row = i;
                        min_possible_col = j;
                    }
                }
            }
        }
    }

    if(done)
        return board;

    int inner_row = min_possible_row / inner_board_size;
    int inner_col = min_possible_col / inner_board_size;

    // After no more deterministic progress, fork on the smallest possibility set.
    for(int possible = 1; possible < board_size + 1; ++possible){
        if(!(min_possible_set & 1)){
            int mask = (1 << (possible - 1));

            board[min_possible_row * board_size + min_possible_col] = possible;
            row_possibles[min_possible_row] |= mask;
            col_possibles[min_possible_col] |= mask;
            inner_possibles[inner_row * inner_board_size + inner_col] |= mask;

            // Fork with updated vars
            std::vector<char> fork_result = solveHelp(board, row_possibles, col_possibles, inner_possibles);
            if(fork_result.size())
                return fork_result;

            // Reset if the fork failed.
            board[min_possible_row * board_size + min_possible_col] = 0;
            row_possibles[min_possible_row] ^= mask;
            col_possibles[min_possible_col] ^= mask;
            inner_possibles[inner_row * inner_board_size + inner_col] ^= mask;
        }
        min_possible_set >>= 1;
    }
    return {};
}

std::vector<char> solve(std::vector<char>& board){
    int board_size = sqrt(board.size());
    int inner_board_size = sqrt(board_size);

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
                row_possibles[i] |= mask;
                col_possibles[j] |= mask;
                
                int inner_row = i / inner_board_size;
                int inner_col = j / inner_board_size;
                inner_possibles[inner_row * inner_board_size + inner_col] |= mask;
            }
        }
    }
    return solveHelp(board, row_possibles, col_possibles, inner_possibles);
}

void printBoard(std::vector<char> board){
    int board_size = sqrt(board.size());

    for(int i = 0; i < board_size * board_size; ++i){
        if(i % board_size == 0) printf("\n");
        printf("%d ", board[i]);
    }
    printf("\n");
}

bool verifySolve(std::vector<char> original, std::vector<char> solution){
    int board_size = sqrt(original.size());
    int inner_board_size = sqrt(board_size);

    std::vector<int> row_possibles(board_size);
    std::vector<int> col_possibles(board_size);
    std::vector<int> inner_possibles(board_size);

    for(int i = 0; i < board_size; ++i){
        for(int j = 0; j < board_size; ++j){
            char val = solution[i * board_size + j];
            char orig_val = original[i * board_size + j];

            // Solution should match all given values.
            if(orig_val && orig_val != val) return false;

            if(val){
                int mask = 1 << (val - 1);
                row_possibles[i] |= mask;
                col_possibles[j] |= mask;
                
                int inner_row = i / inner_board_size;
                int inner_col = j / inner_board_size;
                inner_possibles[inner_row * inner_board_size + inner_col] |= mask;
            }
        }
    }
    int target = (1 << board_size) - 1;
    for(int i = 0; i < board_size; ++i){
        if(row_possibles[i] != target) return false;
        if(col_possibles[i] != target) return false;
        if(inner_possibles[i] != target) return false;
    }

    return true;
}

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -f  --file  <FILENAME>     Path to the input file\n");
    printf("  -s  --size  <INT>          The size of one side of the input board\n");
    printf("  -c  --cuda                 Whether to use the CUDA version (CPU by default)\n");
    printf("  -?  --help                 This message\n");
}

int main(int argc, char** argv){

    int opt;
    static struct option options[] = {
        {"help",     0, 0,  '?'},
        {"file",     1, 0,  'f'},
        {"size",     1, 0,  's'},
        {"cuda",     1, 0,  'c'},
        {0 ,0, 0, 0}
    };

    int board_size;
    bool use_cuda = false;
    std::string board_filename;

    while ((opt = getopt_long(argc, argv, "f:s:c?", options, NULL)) != EOF) {
        switch (opt) {
            case 'f':
                board_filename = optarg;
                break;
            case 's':
                board_size = atoi(optarg);
                break;
            case 'c':
                use_cuda = true;
                break;
            case '?':
            default:
                usage(argv[0]);
                return 1;
        }
    }

    // Read board from input file
    std::ifstream fin(board_filename);
    std::vector<char> first_board(board_size * board_size);
    int tmp;

    for(int i = 0; i < board_size * board_size; ++i){
        fin >> tmp;
        first_board[i] = (char)tmp;
    }

    printBoard(first_board);
    std::vector<char> solution = solve(first_board);
    printBoard(solution);

    if(verifySolve(first_board, solution)){
        printf("Correctness passed!\n");
    }
    else{
        printf("Correctness failed\n");
    }
    return 1;
}