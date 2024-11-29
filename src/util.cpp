#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>

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
            if(orig_val && orig_val != val){
                printf("Solution did not match template at %d, %d\n", i, j);
                return false;
            }

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
        if(row_possibles[i] != target){
            printf("Correctness failed on row %d\n", i);
            return false;
        } 
        if(col_possibles[i] != target){
            printf("Correctness failed on column %d\n", i);
            return false;
        } 
        if(inner_possibles[i] != target){
            printf("Correctness failed on inner grid %d\n", i);
            return false;
        } 
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