#include <vector>

/**
 * @brief Solves a single sudoku board and returns the solved board.
 The board size must be of size nxn, where n is a perfect square, and
 must contain only values between 0 and n inclusive.
 *
 * @param board The unsolved board.
 * @return The solved board.
 */
std::vector<char> solve(std::vector<char> board);


/**
 * @brief Solves all the given sudoku boards and returns the 
 computation time using ordinary CPU computation.
 * 
 * @param boards The boards to solve.
 * @return double The time taken to solve all the boards.
 */
double solveBoardsSeq(std::vector<std::vector<char>> boards);