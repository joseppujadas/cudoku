#include <vector>

/**
 * @brief Solves all the given boards using CUDA.
 * 
 * @param boards The Sudoku boards to solve. 
 * @return double The total time taken to solve all the boards.
 */
double solveBoardHost(std::vector<std::vector<char>> boards);