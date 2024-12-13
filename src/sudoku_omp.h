#include <vector>

/**
 * @brief Solves a single sudoku board and returns the solved board.
 The board size must be of size nxn, where n is a perfect square, and
 must contain only values between 0 and n inclusive.
 *
 * @param board The unsolved board.
 * @param depth The recursion depth to indicate whether to use threads
 for subsequent forks. 1 initially, then > 1 afterwards to not fork.
 * @return The solved board.
 */
std::vector<char> solveOMP(std::vector<char> board, int depth);


/**
 * @brief Solves all the given sudoku boards and returns the 
 computation time using OpenMP. 
 * 
 * @param boards The boards to solve.
 * @return double The time taken to solve all the boards.
 */
double solveBoardsOMP(std::vector<std::vector<char>> boards);