#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"
#include "matrix.h"


/* Function: lst_to_arr
   converts python 2d list into C 2d dynamic array
   Arguments: lst(pointer to python list)
   Returns: pointer to 2d dynamic array
*/
double **lst_to_arr(PyObject *lst){
    int rows,cols, r,c;
    PyObject *row, *item;
    double** res, num;

    rows = PyList_Size(lst);
    row = PyList_GetItem(lst, 0);
    cols = PyList_Size(row);

    res = create_matrix(rows, cols);
    if(res == NULL){return NULL;}

    for(r=0; r<rows; r++){
        row = PyList_GetItem(lst, r);
        
        for(c=0; c<cols; c++){
            item = PyList_GetItem(row, c);
            num = PyFloat_AsDouble(item);
            res[r][c] = num;
        }
    }
    return res;
}

/* Function: arr_to_lst
   converts C 2d dynamic array into python 2d list
   Arguments: arr(double **) pointer to 2d dynamic array, res(PyObject *)a 2d python list
   Returns:  res(pointer to python list)
*/
PyObject *arr_to_lst(double **arr, PyObject *res){
    int r,c,rows;
    PyObject *num, *row;
    rows = PyList_Size(res);

    for(r=0; r<rows; r++){
        row = PyList_GetItem(res,r);
        for(c=0; c<PyList_Size(row); c++){
            num = PyFloat_FromDouble(arr[r][c]);
            PyList_SetItem(row, c, num);
        }
    }
    delete_matrix(arr, rows);
    return res;
}

/* Function: new_lst
   creates python 2d list
   Arguments: rows(int)- number of rows, cols(int)- number of items in each sub-list 
   Returns:  res(pointer to python list)
*/
PyObject *new_lst(int rows, int cols){
    int r,c;
    PyObject *row, *res;

    res = PyList_New(rows);
    for(r=0; r<rows; r++){
        row = PyList_New(cols);
        for(c=0;c<cols;c++){
            PyList_SetItem(row, c, PyFloat_FromDouble(0));
        }
        PyList_SetItem(res, r, row);
    }
    return res;
}

/* Function: sym
   Calculates distances matrix between given vectors in file
   Arguments: vecs (Python list of lists)- representing list of datapoints
   Returns: List of Lists representing the resulting matrix
*/
static PyObject* sym(PyObject *self, PyObject *args){
    double **res, **vectors;
    int  rows, cols;
    PyObject *result, *vecs;
    
    /* Parse function arguments */
    if(!PyArg_ParseTuple(args, "O", &vecs)) {
        return NULL;
    }

    vectors = lst_to_arr(vecs);
    if(vectors == NULL){return NULL;}

    rows = PyList_Size(vecs);
    cols = PyList_Size(PyList_GetItem(vecs,0));

    /* call the shell function in C*/
    res = symshell("sym", vectors, rows, cols);

    if(res == NULL){return NULL;}
    /* Convert result to Python list */
    result = arr_to_lst(res, new_lst(rows, rows));

    return result;
}

/* Function: ddg
   Calculates ddg matrix given vectors file
   Arguments: lst (Python list of lists)
   Returns: List of Lists representing the resulting matrix
*/
static PyObject* ddg(PyObject *self, PyObject *args){
    double **res, ** arr;
    int rows, cols;
    PyObject *lst;

    /* Parse function arguments */
    if(!PyArg_ParseTuple(args, "O", &lst)) {
        return NULL;
    }
    rows = PyList_Size(lst);
    cols = PyList_Size(PyList_GetItem(lst, 0));
    arr = lst_to_arr(lst);

    /* call the shell function in C*/
    res = symshell("ddg", arr, rows, cols);

    if(res == NULL){
        return NULL;
    }
    return arr_to_lst(res, new_lst(rows, rows));
}

/* Function: norm
   Calculates normalized distances matrix between given vectors
   Arguments: lst (Python list of lists)
   Returns: List of Lists representing the resulting matrix
*/
static PyObject* norm(PyObject *self, PyObject *args)
{
    double **res, ** arr;
    int rows, cols;
    PyObject *lst, *res_as_lst;

    /* Parse function arguments */
    if(!PyArg_ParseTuple(args, "O", &lst)) {
        return NULL;
    }
    rows = PyList_Size(lst);
    cols = PyList_Size(PyList_GetItem(lst, 0));
    
    arr = lst_to_arr(lst);

    /* call the shell function in C*/
    res = symshell("norm", arr, rows, cols);
    if(res == NULL){
        return NULL;
    }
    res_as_lst = arr_to_lst(res, new_lst(rows, rows));
    return res_as_lst;
}

/* Function: symnmf
   Compute matrix H for SymNMF algorithm.
   Arguments: matrix H0 (list of lists), matrix W (list of lists), k (int), N (int)
   Returns: List of Lists representing the computed matrix H
*/
static PyObject* symnmf(PyObject *self, PyObject *args){
    double **H, **W;
    PyObject *Hpy, *Wpy;
    int N,k;

    /* Parse function arguments */
    if(!PyArg_ParseTuple(args, "OO", &Hpy, &Wpy)) {
        return NULL;
    }

    N = PyList_Size(Hpy);
    k = PyList_Size(PyList_GetItem(Hpy, 0));


    /* Convert Python lists to C arrays */
    H = lst_to_arr(Hpy);
    if(H==NULL){return NULL;}

    W = lst_to_arr(Wpy);
    if(W==NULL){
        delete_matrix(H,N);
        return NULL;
    }

    /* Call symnmf_run function */
    if(symnmf_run(H, W, k, N) == NULL){return NULL;}

    /* Convert computed matrix H back to Python list */
    Hpy = arr_to_lst(H, new_lst(N,k));    
    return Hpy;
}


/* Define the methods for the module */
static PyMethodDef symnmfMethods[] = {
    {"sym",                   
      (PyCFunction) sym, 
      METH_VARARGS,         
      PyDoc_STR("Calculates distances matrix between given vectors in file. Arguments: X")},
      {"ddg",                   
      (PyCFunction) ddg, 
      METH_VARARGS,         
      PyDoc_STR("Calculates ddg matrix given vectors in file. Arguments: X")},
      {"norm",                   
      (PyCFunction) norm, 
      METH_VARARGS,         
      PyDoc_STR("Calculates normalized distances matrix between given vectors. Arguments: X")},
      {"symnmf",
      (PyCFunction) symnmf,
      METH_VARARGS,
      PyDoc_STR("Computes matrix H for SymNMF algorithm. Arguments: matrix H0, matrix W")}, 
    {NULL, NULL, 0, NULL}     
};


/* Define the module */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT, 
    "mysymnmf", /* Module name */
    NULL, /* Module documentation (unused in this case) */
    -1,   /* Module state size (unused in this case) */
    symnmfMethods /* Method table */
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_mysymnmf(void)
{
    PyObject *m;
    m = PyModule_Create(&symnmfmodule);
    if (!m) {
        return NULL;
    }
    return m;
}