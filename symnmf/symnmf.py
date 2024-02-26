import numpy as np
import mysymnmf
import sys

err = "An Error Has Occurred"
np.random.seed(0)

def sym(X):
    """
    :param X: 2d list (matrix)  contains N vectors, each of dimension d
    :type X: list
    The function returns the sym matrix (N*N) 
    """
    res = mysymnmf.sym(X)
    if res is None:
        sys.exit(err)
    return res

def ddg(X):
    """
    :param X: 2d list (matrix)  contains N vectors, each of dimension d
    :type X: list
    The function returns the ddg matrix (N*N) 
    """
    res = mysymnmf.ddg(X)
    if res is None:
        sys.exit(err)
    return res

def norm(X):
    """
    :param X: 2d list (matrix)  contains N vectors, each of dimension d
    :type X: list
    The function returns the norm matrix (N*N) 
    """
    res = mysymnmf.norm(X)
    if res == None:
        sys.exit(err)
    return res

def symnmf(X, k):
    """
    :param X: 2d list (matrix)  contains N vectors, each of dimension d
    :type X: list
    :param k: number of clusters
    :type k: int
    The function returns the symnmf matrix (N*k) 
    """
    W = norm(X)
    N = len(W)
    m = float(np.mean(W))
    # Create H0
    H = np.random.uniform(0, 2*np.sqrt(m/k), (N,k)).tolist()
    # Compute H matrix using SymNMF
    res = mysymnmf.symnmf(H, W)
    if res == None:
        sys.exit(err)
    return res

def main(argv):
    """
    Main function to perform symmetric non-negative matrix factorization (SymNMF) and print the results.
    :param argv: Command-line arguments containing [script_name, k, goal, file_path].
    :type argv: list
    """
    k = int(argv[1])
    goal = argv[2]
    X = np.loadtxt(argv[3], delimiter=",").tolist()
    # Call matching function to the goal
    funcs ={
        "sym": sym,
        "ddg": ddg,
        "norm": norm,
    }

    if goal != "symnmf":
        res = funcs[goal](X)
    else:
        res = symnmf(X, k)
    
    # Print the result
    print_answer(res)
        

def print_answer(matrix):
    """
    :param matrix: a matrix to print
    :type matrix: list
    The function prints the input matrix in the required format 
    """
    for row in matrix:
        for i in range(len(row)):
            if i < len(row)-1:
                print(f"{row[i]:.4f}",end=",")
            else:
                print(f"{row[i]:.4f}")


if __name__ == "__main__":
    main(sys.argv)