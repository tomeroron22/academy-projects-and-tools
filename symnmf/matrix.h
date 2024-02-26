#ifndef MATRIX_H_
#define MATRIX_H_


/* Function: create_matrix
   Creates and returns a 2-dimensional matrix (double type).
   Arguments: rows (integer), cols (integer)
   Returns: Pointer to the created matrix
   If memory allocation fails, returns NULL
*/
double **create_matrix(int rows,int cols);


/* Function: delete_matrix
   Deletes a 2-dimensional matrix.
   Arguments: matrix (pointer to a pointer to double), rows (integer)
   Returns: None
*/
void delete_matrix(double ** matrix, int rows);

/* Function: add
   Adds two 2-dimensional matrices element-wise.
   Arguments: mat1 (pointer to a pointer to double), mat2 (pointer to a pointer to double),
              rows (integer), cols (integer)
   Returns: Pointer to the resulting matrix
   If memory allocation fails, returns NULL
*/
double ** add(double **mat1, double **mat2, int rows, int cols);

/* Function: sub
   Subtracts two 2-dimensional matrices element-wise (mat1 - mat2).
   Arguments: mat1 (pointer to a pointer to double), mat2 (pointer to a pointer to double),
              rows (integer), cols (integer)
   Returns: Pointer to the resulting matrix
   If memory allocation fails, returns NULL
*/
double ** sub(double **mat1, double **mat2, int rows, int cols);

/* Function: transpose
   Computes and returns the transpose of a 2-dimensional matrix.
   Arguments: matrix (pointer to a pointer to double), rows (integer), cols (integer)
   Returns: Pointer to the transposed matrix
   If memory allocation fails, returns NULL
*/
double ** transpose(double **matrix, int rows,int cols);

/* Function: mult
   Multiplies two 2-dimensional matrices.
   Arguments: mat1 (pointer to a pointer to double), mat2 (pointer to a pointer to double),
              rows1 (integer), cols1 (integer), cols2 (integer)
   Returns: Pointer to the resulting matrix
   If memory allocation fails, returns NULL
*/
double ** mult(double **mat1, double **mat2, int rows1, int cols1, int cols2);

/* Function: hadamard_mult
   Performs element-wise multiplication of two 2-dimensional matrices.
   Arguments: mat1 (pointer to a pointer to double), mat2 (pointer to a pointer to double),
              rows (integer), cols (integer)
   Returns: Pointer to the resulting matrix
   If memory allocation fails, returns NULL
*/
double ** hadamard_mult(double **mat1, double **mat2, int rows, int cols);

/* Function: hadamard_div
   Performs element-wise division of two 2-dimensional matrices.
   Arguments: mat1 (pointer to a pointer to double), mat2 (pointer to a pointer to double),
              rows (integer), cols (integer)
   Returns: Pointer to the resulting matrix
   If memory allocation fails, returns NULL
*/
double ** hadamard_div(double **mat1, double **mat2, int rows, int cols);

/* Function: elem_pow
   Computes element-wise power of a 2-dimensional matrix.
   Arguments: matrix (pointer to a pointer to double), rows (integer), cols (integer),
              power (double)
   Returns: Pointer to the resulting matrix
   If memory allocation fails, returns NULL
*/
double ** elem_pow(double **matrix, int rows, int cols, double power);

/* Function: frob_norm2
   Computes the Frobenius norm squared between two 2-dimensional matrices.
   Arguments: matrix1 (pointer to a pointer to double), matrix2 (pointer to a pointer to double),
              rows (integer), cols (integer)
   Returns: Frobenius norm squared (double)
*/
double frob_norm2(double **matrix1, double**matrix2, int rows,int cols);

/* Function: euc_norm2
   Computes the squared Euclidean norm between two vectors.
   Arguments: vec1 (pointer to double), vec2 (pointer to double), dim (integer)
   Returns: Squared Euclidean norm (double)
*/
double euc_norm2(double *vec1, double* vec2, int dim);

/* Function: print_matrix
   Prints a 2-dimensional matrix in the required format.
   Arguments: matrix (pointer to a pointer to double), rows (integer), cols (integer)
   Returns: None
*/
void print_matrix(double **matrix, int rows,int cols);
#endif 
