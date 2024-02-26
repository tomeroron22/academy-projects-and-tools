#ifndef SYMNMF_H_
#define SYMNMF_H_

/* Function: symnmf
   Performs symmetric non-negative matrix factorization (symNMF).
   Arguments: H (2D array) - Matrix H.
              W (2D array) - Matrix W.
              k (int) - Rank of factorization.
              N1 (int) - Number of data points.
   Returns: Pointer to the computed matrix H.
   Memory allocation failure returns NULL.
*/
double **symnmf_run(double **H, double**W, int k, int N1);

/* Function: symshell
   Performs spectral clustering and matrix factorization based on the specified goal.
   Arguments: goal (string) - Desired operation ("sym", "ddg", "norm", "symnmf").
              path (string) - Path to input data file.
              n (int) - The number of data points.
   Returns: Pointer to the computed matrix (symmetry, diagonal degree, or normalized similarity).
   Memory allocation failure or invalid goal returns NULL.
*/
double **symshell(const char *goal, double **vectors, int n, int d1);

#endif
