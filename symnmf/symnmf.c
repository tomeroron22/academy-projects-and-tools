#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "matrix.h"
#include "symnmf.h"

#define EPS 0.0001
#define GENERAL_ERROR "An Error Has Occurred\n"

/* Global variables */
int d = 0, N = 0, k = 0;




/* Function: read_input
   Reads input data from a file and stores it in a 2D array.
   Arguments: path (string) - Path to the input file.
   Returns: Pointer to a 2D array containing input vectors.
   Memory allocation failure or file read failure returns NULL.
*/
double **read_input(const char *path){
    int maxsize=100, idx;
    FILE *ifp = NULL;
    double **vectors;
    double *vecrow;

    /*Allocate memory for the temporary row vector*/ 
    vecrow = (double *)calloc(maxsize, sizeof(double));
    if(vecrow == NULL){return NULL;} 

    ifp = fopen(path, "r");
    if(ifp == NULL){
        free(vecrow);
        return NULL;
    }
    /* read first row to determine d, element at a time*/
    while(fscanf(ifp, "%lf", &(vecrow[d])) == 1) {
        d++;

        /* Check if the line ends */
        if (fgetc(ifp) == '\n') {
            N++;

            /* Realloc in order to create vector 0f size d */
            vecrow = (double *) realloc(vecrow, d*sizeof(double));
            if(vecrow == NULL){
                fclose(ifp);
                return NULL;
            }
            break;
        }

        /* Resize the temporary row vector if it reaches maxsize */
        if(d==maxsize){
            maxsize *=2;
            vecrow = (double *)realloc(vecrow, maxsize*sizeof(double));
            if(vecrow == NULL){
                fclose(ifp);
                return NULL;
            }
        }
    }

    /* Allocate memory array for temporary matrix*/
    vectors = (double**)calloc(maxsize, sizeof(double *));
    if(vectors== NULL){
        free(vecrow);
        fclose(ifp);
        return NULL;
    }
    vectors[0] = vecrow;

    vecrow = (double*)calloc(d, sizeof(double));
    if(vecrow== NULL){
        delete_matrix(vectors, N);
        fclose(ifp);
        return NULL;
    }
    idx = 0;
    while(fscanf(ifp, "%lf", &(vecrow[idx])) == 1) {
        idx++;
        fgetc(ifp);

        /* When a row is complete, add it to the array of vectors */
        if (idx == d){
            idx = 0;
            vectors[N] = vecrow;
            vecrow = (double*)calloc(d, sizeof(double));
            if(vecrow == NULL){
                delete_matrix(vectors, N);
                fclose(ifp);
                return NULL;
            }
            N++;

            /* Resize the array of vectors if necessary */
            if (N == maxsize){
                maxsize *= 2;
                vectors = (double **) realloc(vectors ,maxsize*sizeof(double *));
                if(vectors== NULL){
                    delete_matrix(vectors, N);
                    fclose(ifp);
                    return NULL;
                }
            }
        }
    }

    fclose(ifp);
    /* Resize the vectors to have N rows*/
    vectors = (double **)realloc(vectors ,N*sizeof(double *));
    if(vectors== NULL){
        delete_matrix(vectors, N);
        return NULL;
    }
    free(vecrow);
    return vectors;
}

/* Function: sym
   Computes the similarity matrix based on input vectors.
   Arguments: vectors (2D array) - Input vectors.
   Returns: Pointer to the computed similarity matrix.
   Memory allocation failure returns NULL.
*/
double **sym(double **vectors){
    int r,c;
    double *vec1, *vec2;
    double ** sym_mat = create_matrix(N,N);
    if(sym_mat == NULL){return NULL;}

    for(r=0; r<N; r++){
        vec1 = vectors[r];
        for(c=0; c<N; c++){
            vec2 = vectors[c];

            if(r==c) {sym_mat[r][c] = 0;}
            else {sym_mat[r][c] = exp(-0.5 * euc_norm2(vec1, vec2, d));}
        }
    }
    return sym_mat;
}

/* Function: ddg
   Computes the diagonal degree matrix based on a similarity matrix.
   Arguments: sym_mat (2D array) - Symmetry matrix.
   Returns: Pointer to the computed diagonal degree matrix.
   Memory allocation failure returns NULL.
*/
double **ddg(double **sym_mat){
    int r,c;
    double ** ddg_mat = create_matrix(N,N);
    if(ddg_mat == NULL){
        return NULL;
    }

    for(r=0; r<N; r++){
        for(c=0; c<N; c++){
             ddg_mat[r][r] += sym_mat[r][c];
        }
    }
    return ddg_mat;
}

/* Function: norm
   Computes the normalized similarity matrix based on a diagonal degree matrix and a symmetry matrix.
   Arguments: ddg_mat (2D array) - Diagonal degree matrix.
              sym_mat (2D array) - Symmetry matrix.
   Returns: Pointer to the computed normalized similarity matrix.
   Memory allocation failure returns NULL.
*/
double **norm(double **ddg_mat, double **sym_mat){
    double **half_norm_mat, **d_half, **norm_mat;
    
    /* Calculate D^-0.5 */
    d_half = elem_pow(ddg_mat, N,N, -0.5);
    if(d_half == NULL){return NULL;}

    /* Calculate  D^-0.5 * A */
    half_norm_mat = mult(d_half, sym_mat, N, N, N);
    if(half_norm_mat == NULL){
        delete_matrix(d_half, N);
        return NULL;
    }
    /* Calculate  (D^-0.5 * A)* D^-0.5 */
    norm_mat = mult(half_norm_mat, d_half, N, N, N);
    if(norm_mat == NULL){
        delete_matrix(half_norm_mat, N);
        delete_matrix(d_half, N);
        return NULL;
    }

    /* free allocated memory */
    delete_matrix(half_norm_mat, N);
    delete_matrix(d_half, N);

    return norm_mat;
}

/* Function: symnmf_run
   Performs symmetric non-negative matrix factorization (symNMF).
   Arguments: H (2D array) - Matrix H.
              W (2D array) - Matrix W.
              k (int) - Rank of factorization.
              N1 (int) - Number of data points.
   Returns: Pointer to the computed matrix H.
   Memory allocation failure returns NULL.
*/
double **symnmf_run(double **H, double**W, int k, int N1){
    int iter = 0, c, r;
    double diff = DBL_MAX;
    double **H1, **WH, **HHtH, **HHt, **Ht, **WH_div_HHtH;
    N=N1;


    while (iter < 300 && diff > EPS){
        /* Calculate WH = W * H */
        WH = mult(W, H, N, N, k);
        
        if(WH == NULL){
            /* Clean up and return on memory allocation failure */
            delete_matrix(W, N);
            delete_matrix(H, N);
            return NULL;
        }

        /* Calculate Ht = transpose(H) */
        Ht = transpose(H, N, k);
        if(Ht == NULL){
            /* Clean up and return on memory allocation failure */
            delete_matrix(W, N);
            delete_matrix(H, N);
            delete_matrix(WH, N);
            return NULL;
        }

        /* Calculate HHt = H * Ht */
        HHt = mult(H, Ht, N, k, N);
        delete_matrix(Ht, k);
        if(HHt == NULL){
            /* Clean up and return on memory allocation failure */
            delete_matrix(W, N);
            delete_matrix(H, N);
            delete_matrix(WH, N); 
            return NULL;
        }
        
        /* Calculate HHtH = HHt * H */
        HHtH = mult(HHt, H, N, N, k);
        delete_matrix(HHt, N);
        if(HHtH == NULL){
            /* Clean up and return on memory allocation failure */
            delete_matrix(W, N);
            delete_matrix(H, N);
            delete_matrix(WH, N);
            return NULL;
        }

        /* Calculate WH_div_HHtH = hadamard division product of WH and HHtH */
        WH_div_HHtH = hadamard_div(WH, HHtH, N, k);
        delete_matrix(WH, N);
        delete_matrix(HHtH, N);
        if(WH_div_HHtH == NULL){
            /* Clean up and return on memory allocation failure */
            delete_matrix(W, N);
            delete_matrix(H, N);
            return NULL;
        }
        
        for(r=0; r<N; r++){
            for(c=0; c<k; c++){
                WH_div_HHtH[r][c] = 0.5 + (0.5 * WH_div_HHtH[r][c]);
            }
        }

        /* Calculate H1 = hadamard multiplication of H and WH_div_HHtH */
        H1 = hadamard_mult(H,WH_div_HHtH, N, k);
        if(H1 ==NULL){
        /* Calculate H1 = hadamard multiplication of H and WH_div_HHtH */
            delete_matrix(W, N);
            delete_matrix(H, N);
            delete_matrix( WH_div_HHtH, N);
            return NULL;
        }

        delete_matrix(WH_div_HHtH, N);
        /* Calculate Frobenius norm between Hi and Hi+1 */
        diff = frob_norm2(H1, H, N,k);
        
        /* Copy values from H1 to H */
        for(r=0;r<N;r++){
            for(c=0;c<k;c++){
                H[r][c] = H1[r][c];
            }
        }
        /* Clean up H1 */
        delete_matrix(H1, N);
        iter++;
    }
    /* Clean up W and return the updated H */
    delete_matrix(W, N);
    return H;

}

/* Function: symshell
   Performs spectral clustering and matrix factorization based on the specified goal.
   Arguments: goal (string) - Desired operation ("sym", "ddg", "norm", "symnmf").
              path (string) - Path to input data file.
              n (int) - The number of data point.
              d1 (int) - The dimensions of eacj point 
   Returns: Pointer to the computed matrix (symmetry, diagonal degree, or normalized similarity).
   Memory allocation failure or invalid goal returns NULL.
*/
double **symshell(const char *goal, double **vectors, int n, int d1){
    
    double **sym_mat, **ddg_mat, **norm_mat;
    N = n;
    d = d1;
    /* Compute the symmetry matrix */
    sym_mat = sym(vectors);
    delete_matrix(vectors, N);
    if(sym_mat == NULL){return NULL;}

    if(strcmp(goal, "sym")==0){return sym_mat;}  /* Check if goal == sym*/
    
   
    /* Compute the diagonal degree matrix */
    ddg_mat = ddg(sym_mat);
    
    if(ddg_mat == NULL){
        delete_matrix(sym_mat, N);
        return NULL;
    }

    if(strcmp(goal, "ddg")==0){
        delete_matrix(sym_mat, N);
        return ddg_mat;
    }
    
    /* If here - goal != sym, ddg --> goal == norm or symnmf 
       Calculate normalized similarity matrix */
    norm_mat = norm(ddg_mat, sym_mat);
    delete_matrix(sym_mat, N);
    delete_matrix(ddg_mat, N);
    if(norm_mat == NULL){return NULL;}

    return norm_mat;
}


/* MAIN FUNCTION*/
int main(int argc, char *argv[]){
    
    double **res, **vectors;
    if(argc !=3){ 
        printf(GENERAL_ERROR);
        return 1;
    }
    /* Read input data from the specified path */
    vectors = read_input(argv[2]);
    if (vectors == NULL){
        printf(GENERAL_ERROR);
        return 1;}
   
    res = symshell(argv[1], vectors, N, d);
    if(res == NULL){
        printf(GENERAL_ERROR);
        return 1;
    }
    print_matrix(res, N, N);
    delete_matrix(res, N);
    return 0;
}