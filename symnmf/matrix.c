#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "matrix.h"

/* The function creates and returns a 2-dimentional matrix (double type) where #rows = rows, #columns = cols.
   All entries are set to 0 (by using calloc).
   If can't allocate memory - returns NULL*/
double **create_matrix(int rows,int cols){
    int r;
    double **mat = (double **)calloc(rows, sizeof(double *));
    if(mat == NULL){return NULL;}

    for(r=0; r<rows; r++){
        mat[r] = (double*)calloc(cols, sizeof(double));
        if(mat[r] == NULL){
            delete_matrix(mat, r);
            return NULL;
        }
    }
    return mat;
}

/* The function deletes a 2-dimentional matrix with #rows = rows*/
void delete_matrix(double **matrix, int rows){
    int r;
    for(r=0; r<rows; r++){
        free(matrix[r]);
    }
    free(matrix);
}

/* The function recives two 2-dimentional matrices where #rows = rows, #columns = cols, for both.
   It creates and returns a 2-dimentional matrix (double type) where each entry is set to the sum of
   the corresponding entries of the input matrices.
   If can't allocate memory for the new matrix - returns NULL*/
double ** add(double **mat1, double **mat2, int rows, int cols){
    int r, c;
    double **mat = create_matrix(rows,cols);
    if(mat == NULL){return NULL;}

    for(r=0; r<rows; r++){
        for(c=0; c<cols; c++){
            mat[r][c] = mat1[r][c] + mat2[r][c];
        }
    }
    return mat;
}

/* The function recives two 2-dimentional matrices where #rows = rows, #columns = cols, for both.
   It creates and returns a 2-dimentional matrix (double type) where each entry is set to the subtraction of
   the corresponding entries of the input matrices (mat1 - mat2).
   If can't allocate memory for the new matrix - returns NULL*/
double ** sub(double **mat1, double **mat2, int rows, int cols){
    int r, c;
    double **mat = create_matrix(rows,cols);
    if(mat == NULL){return NULL;}

    for(r=0; r<rows; r++){
        for(c=0; c<cols; c++){
            mat[r][c] = mat1[r][c] - mat2[r][c];
        }
    }
    return mat;
}

/* The function recives two 2-dimentional matrices - mat1= rows1 X cols1, mat2= cols1 X cols2.
   It creates and returns a 2-dimentional matrix (double type) where each entry is set to the muiltiplication
   of the two matrices (not element wise). New matrix is of size rows1 X cols2.
   If can't allocate memory for the new matrix - returns NULL*/
double ** mult(double **mat1, double **mat2, int rows1, int cols1, int cols2){
    int i, j, k;
    double **mat = create_matrix(rows1,cols2);
    if(mat == NULL){return NULL;}

    for(i=0; i<rows1; i++) {
        for (j=0; j<cols2; j++) {
            for (k=0; k<cols1; k++) {
                mat[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return mat;
}

/* The function recives two 2-dimentional matrices where #rows = rows, #columns = cols, for both.
   It creates and returns a 2-dimentional matrix (double type) where each entry is set to the muiltiplication
   of the corresponding entries of the input matrices (element wise).
   If can't allocate memory for the new matrix - returns NULL*/
double ** hadamard_mult(double **mat1, double **mat2, int rows, int cols){
    int r, c;
    double **mat = create_matrix(rows,cols);
    if(mat == NULL){return NULL;}

    for(r=0; r<rows; r++){
        for(c=0; c<cols; c++){
            mat[r][c] = mat1[r][c] * mat2[r][c];
        }
    }
    return mat;
}

/* The function recives two 2-dimentional matrices where #rows = rows, #columns = cols, for both.
   It computes and returns frobenius norm of the subtration between mat1 to mat2*/
double frob_norm2(double **mat1, double**mat2, int rows, int cols){
    double sum=0;
    int r, c;
    for(r=0; r<rows; r++){
        for(c=0; c<cols; c++){
            sum += pow((mat1[r][c] - mat2[r][c]), 2);
        }
    }
    return sum;
}

/* The function recives two vectors (dimention = dim).
   It computes and returns euclidean norm of the subtration between vec1 to vec2*/
double euc_norm2(double *vec1, double *vec2, int dim){
    double sum=0;
    int idx;
    for(idx=0; idx<dim; idx++){
        sum += pow((vec1[idx]-vec2[idx]),2);
    }
    return sum;
}

/* The function recives a 2-dimentional matrix where #rows = rows, #columns = cols.
   It creates and returns the transpose matrix (#rows = cols, #columns = rows).
   If can't allocate memory for the new matrix - returns NULL*/
double ** transpose(double **matrix, int rows,int cols){
    int r, c;
    double **mat = create_matrix(cols,rows);
    if(mat == NULL){return NULL;}

    for(r=0; r<cols; r++){
        for(c=0; c<rows; c++){
            mat[r][c] = matrix[c][r];
        }
    }
    return mat;
}

/* The function recives two 2-dimentional matrices where #rows = rows, #columns = cols, for both.
   It creates and returns a 2-dimentional matrix (double type) where each entry is set to the division
   between the corresponding entries of the input matrices (element wise).
   If can't allocate memory for the new matrix - returns NULL*/
double ** hadamard_div(double **mat1, double **mat2, int rows, int cols){
    int r, c;
    double **mat = create_matrix(rows,cols);
    if(mat == NULL){return NULL;}

    for(r=0; r<rows; r++){
        for(c=0; c<cols; c++){
            mat[r][c] = mat1[r][c] / mat2[r][c];
        }
    }
    return mat;
}

/* The function recives a 2-dimentional matrix where #rows = rows, #columns = cols.
   It creates and returns a 2-dimentional matrix (double type) where each entry is set to the power of the
   corresponding entry of the input matrix, to the power of "power" input.
   If can't allocate memory for the new matrix - returns NULL*/
double **elem_pow(double **matrix, int rows, int cols, double power){
    int r, c;
    double **mat = create_matrix(rows,cols);
    if(mat == NULL){return NULL;}

    for(r=0; r<rows; r++){
        for(c=0; c<cols; c++) {
            if (matrix[r][c] != 0) {
                mat[r][c] = pow(matrix[r][c], power);
            }
        }
    }
    return mat;
}

/* The function recives a 2-dimentional matrix where #rows = rows, #columns = cols.
   It prints the marix in the format required in the guidelines*/
void print_matrix(double **matrix, int rows, int cols){
    int r,c;
    for(r=0;r<rows;r++){
        for(c=0;c<cols;c++){
            printf("%.4f",matrix[r][c]);
            if(c<cols-1){
                printf(",");
            }
        }
        printf("\n");
    }
}