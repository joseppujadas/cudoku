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

    // Generate possibility bitmasks for every row, column, and subgrid.
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
    
    // Bit mask of 111111... for the first board_size bits.
    int target = (1 << board_size) - 1;

    // Every bit mask should be equal to all ones for the first board_size bits.
    // This indicates every value is present, since if any value was missing it
    // indicates 0s or duplicate values.
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