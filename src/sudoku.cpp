#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

#include <getopt.h>
#include <omp.h>

#include "cudoku.h"
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
    // std::vector<char> sol = {};
    
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

std::vector<char> solveOMP(std::vector<char> board, int depth=1){
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

                    // Conflicting deterministic update check
                    if((row_possibles[i] & mask) 
                    or (col_possibles[j] & mask) 
                    or (inner_possibles[inner_idx] & mask)){
                        error = true;
                    }

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

        // #pragma omp taskloop collapse(2) default(shared) untied
        #pragma omp taskloop collapse(2) default(shared)
        for(int i = 0; i < board_size; ++i){
            for(int j = 0; j < board_size; ++j){
                // printf("%d\n", omp_get_thread_num());
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
    if(depth == 1){
        std::vector<char> sol = {};
        #pragma omp taskloop default(shared) firstprivate(board) grainsize(1)
        for(int possible = 1; possible < board_size + 1; ++possible){
            if(!(min_possible_set & (1 << (possible-1)))){
                board[min_possible_idx] = possible;

                // Fork with updated vars
                std::vector<char> fork_result = solveOMP(board, 2);
                if(fork_result.size()){
                    sol = fork_result;
                    #pragma omp cancel taskgroup
                }
            }
        }
        
        return sol;
    }

    else{
        for(int possible = 1; possible < board_size + 1; ++possible){
            if(!(min_possible_set & 1)){
                board[min_possible_idx] = possible;

                // Fork with updated vars
                std::vector<char> fork_result = solveOMP(board, 2);
                if(fork_result.size()){
                    return fork_result;
                }
            }
            min_possible_set >>= 1;
        }
        

        return {};
    }
    
    
    
}

double solveBoardsCPU(std::vector<std::vector<char>> boards){

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

double solveBoardsOMP(std::vector<std::vector<char>> boards){

    double total_time = 0;
    

    for(const auto& board : boards){

        const auto compute_start = std::chrono::steady_clock::now();
        
        std::vector<char> solution = solveOMP(board);

        const auto compute_time = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - compute_start).count();
        
        if(!verifySolve(board, solution)){
            printBoard(solution);
        }
        total_time += compute_time;
    }

    return total_time;
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
    int num_threads = 2;
    std::string board_filename;

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
    printf("Sequential: %lf\n", solveBoardsCPU(boards));
    printf("OpenMP: %lf\n", solveBoardsOMP(boards));
    printf("CUDA: %lf\n", solveBoardHost(boards));

    return 1;
}
