#include <vector>

/**
 * @brief Prints a board in a human-readable grid layout. Assuming the board
 * is nxn, prints a newline character every n values.
 * 
 * @param board The board data to print.
 */
void printBoard(std::vector<char> board);

/**
 * @brief Verifies a solution to a board by checking all possible values
 * occur in each row, column, and subgrid. Also checks that template cells
 * match the solution where values are given. Prints where and how the board
 * is not correct if an error occurs.
 * 
 * @param original The template board.
 * @param solution The solved board.
 * @return true If The solution is correct.
 * @return false Otherwise.
 */
bool verifySolve(std::vector<char> original, std::vector<char> solution);