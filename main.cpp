
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "MarketIO.h"

#include <cuda.h>
#include "helper_cusolver.h"

#include <cusolverSp.h>
#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include "mmio.h"
#include <helper_cuda.h>

#include "MatSparse.h";

#define METIS_LIBS 1

#ifdef METIS_LIBS
#include "Eigen/MetisSupport"
#endif

char *filename = "MatrixMarket_MHM_subproblem18k.txt";

template <typename T_ELEM>
int loadMMSparseMatrix(
    char *filename,
    char elem_type,
    bool csrFormat,
    int *m,
    int *n,
    int *nnz,
    T_ELEM **aVal,
    int **aRowInd,
    int **aColInd,
    int extendSymMatrix);

void testEigen(int m, int n, int nnz, std::vector<int>& rows, std::vector<int>& cols,
		std::vector<double>& values, double* matB){

	double start, stop, time_to_solve, time_to_build;
    double tol=1e-9;
    Eigen::SparseMatrix<double> A;

    std::vector< Eigen::Triplet<double> > trips;
    trips.reserve(m * n);

    for (int k = 0; k < nnz; k++){
    	double _val = values[k];
    	int i = rows[k];
    	int j = cols[k];

    	if (fabs(_val) > tol){
    		trips.push_back(Eigen::Triplet<double>(i-1,j-1,_val));
        }
    }



    //NOTE: setFromTriples() accumulates contributions to the same (i,j)!
    A.resize(m, n);
    start = second();
    A.setFromTriplets(trips.begin(), trips.end());
    stop = second();
    time_to_build = stop - start;

	Eigen::SparseLU< Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int> > solverLU;



    Eigen::VectorXd b; b.resize(m);
    for (int i = 0; i < m; i++ ) b(i) = matB[i];

	printf("\nProcessing in Eigen using LU...\n");
	start = second();
	solverLU.compute(A);
	Eigen::VectorXd X = solverLU.solve(b);
	stop = second();
	time_to_solve = stop - start;

    Eigen::VectorXd ax = A * X;
    Eigen::VectorXd bMinusAx = b - ax;

	double h_r[m];
    for (int i=0; i<m; i++) h_r[i]=bMinusAx(i);

    double r_inf = vec_norminf(m, h_r);

    printf("(Eigen) |b - A*x| = %E \n", r_inf);
    printf("(Eigen) Time to build(sec): %f\n", time_to_build);
    printf("(Eigen) Time (sec): %f\n", time_to_solve);
}

void testCusolver(int rows, int cols, int nnz, int *row_ptr, int *col_index, double *values,
		double *valuesB){
    // --- Initialize cuSPARSE
 	cusolverSpHandle_t cusolver_handle = NULL; checkCudaErrors(cusolverSpCreate(&cusolver_handle));
 	cusparseHandle_t cusparse_handle = NULL; checkCudaErrors(cusparseCreate(&cusparse_handle));
 	cudaStream_t cudaStream = NULL; checkCudaErrors(cudaStreamCreate(&cudaStream));
 	checkCudaErrors(cusolverSpSetStream(cusolver_handle, cudaStream));
 	checkCudaErrors(cusparseSetStream(cusparse_handle, cudaStream));


    cusparseMatDescr_t descrA;      checkCudaErrors(cusparseCreateMatDescr(&descrA));
    checkCudaErrors(cusparseSetMatType     (descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));

    double start, stop, time_to_solve;
    start = second();

    // --- Device side dense matrix
    printf("\nAlloc GPU memory...\n");
    double *d_A;            checkCudaErrors(cudaMalloc(&d_A, nnz * sizeof(double)));
    int *d_A_RowIndices;    checkCudaErrors(cudaMalloc(&d_A_RowIndices, (rows + 1) * sizeof(int)));
    int *d_A_ColIndices;    checkCudaErrors(cudaMalloc(&d_A_ColIndices, nnz * sizeof(int)));
    double *d_x;        checkCudaErrors(cudaMalloc(&d_x, rows * sizeof(double)));

    checkCudaErrors(cudaMemcpy(d_A, values, nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A_RowIndices, row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_A_ColIndices, col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));

    double *d_b; checkCudaErrors(cudaMalloc(&d_b, rows * sizeof(double)));
    checkCudaErrors(cudaMemcpy(d_b, valuesB, rows * sizeof(double), cudaMemcpyHostToDevice));

    double *h_x = (double *)malloc(rows * sizeof(double));

    double tol = 1.e-12;
    int reorder = 0;
    int singularity = 0;
	printf("\nProcessing in GPU using cusolver QR...\n");


    //checkCudaErrors(cusolverSpDcsrlsvluHost(cusolver_handle, Nrows, nnz, descrA, sparse.Values(),
    	//	sparse.RowPtr(), sparse.ColIdx(), mB.values, tol, reorder, h_x, &singularity));
    checkCudaErrors(cusolverSpDcsrlsvqr(cusolver_handle, rows, nnz, descrA, d_A, d_A_RowIndices,
           		d_A_ColIndices, d_b, tol, reorder, d_x, &singularity));
    checkCudaErrors(cudaDeviceSynchronize());
    stop = second();
    time_to_solve = stop - start;
    checkCudaErrors(cudaMemcpy(h_x, d_x, rows * sizeof(double), cudaMemcpyDeviceToHost));

    double minusOne = -1.0;
    double one = 1.0;
    double *d_r; checkCudaErrors(cudaMalloc((void **)&d_r, sizeof(double)*rows));
    checkCudaErrors(cudaMemcpy(d_r, d_b, sizeof(double)*rows, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cusparseDcsrmv(cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            rows,
            cols,
            nnz,
            &minusOne,
            descrA,
            d_A,
            d_A_RowIndices,
            d_A_ColIndices,
            d_x,
            &one,
            d_r));
    double *h_r; h_r = (double*) malloc(rows * sizeof(double));
    checkCudaErrors(cudaMemcpy(h_r, d_r, sizeof(double)*rows, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_r, d_r, rows * sizeof(double), cudaMemcpyDeviceToHost));

    double r_inf = vec_norminf(rows, h_r);

    printf("(GPU - cuSolver) Time (sec): %f\n", time_to_solve);
    printf("(Eigen) |b - A*x| = %E \n", r_inf);

    checkCudaErrors(cusparseDestroy(cusparse_handle));
    checkCudaErrors(cusolverSpDestroy(cusolver_handle));
    checkCudaErrors(cudaStreamDestroy(cudaStream));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_r));

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_A_RowIndices));
    checkCudaErrors(cudaFree(d_A_ColIndices));

    free(h_x);
    free(h_r);
}

void testCuda(int m, int n, int nnz, std::vector<int>& rows, std::vector<int>& cols,
		std::vector<double>& values, double* matB){

    double tol=1e-9;
    double start, stop, time_to_build, time_to_solve;

    int cudaDevice = 0;

    checkCudaErrors(cudaSetDevice(cudaDevice));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDevice);
    printf("Device Number: %d\n", cudaDevice);
    printf("  Device name: %s\n", prop.name);
    checkCudaErrors(cudaDeviceReset());

	 size_t mem_tot = 0;
	 size_t mem_free = 0;

	 cudaMemGetInfo(&mem_free, & mem_tot);
	 printf("\nFree memory: %d", mem_free);

	MatSparse matA;
    matA.setSize(m, n);

    std::vector<int> I, J;
    std::vector<double> V;

    for (int k = 0; k < nnz; k++){
    	double _val = values[k];
    	int i = rows[k];
    	int j = cols[k];

    	if (fabs(_val) > tol){
        	I.push_back(i-1);
        	J.push_back(j-1);
        	V.push_back(_val);
        }
    }

    start = second();
    matA.fromTruples(I, J, V);
    stop = second();
    time_to_build = stop - start;
    std::cerr << "Time to Build in GPU (second): " << time_to_build << std::endl;


    // ******************************** GPU SOLVER ******************************** //

    // --- Initialize cuSPARSE
     	cusolverSpHandle_t cusolver_handle = NULL; checkCudaErrors(cusolverSpCreate(&cusolver_handle));
     	cusparseHandle_t cusparse_handle = NULL; checkCudaErrors(cusparseCreate(&cusparse_handle));
     	cudaStream_t cudaStream = NULL; checkCudaErrors(cudaStreamCreate(&cudaStream));
     	checkCudaErrors(cusolverSpSetStream(cusolver_handle, cudaStream));
     	checkCudaErrors(cusparseSetStream(cusparse_handle, cudaStream));


        cusparseMatDescr_t descrA;      checkCudaErrors(cusparseCreateMatDescr(&descrA));
        checkCudaErrors(cusparseSetMatType     (descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
        checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));


        printf("\nAlloc GPU memory...\n");
        double *d_A;            checkCudaErrors(cudaMalloc(&d_A, nnz * sizeof(double)));
        int *d_A_RowIndices;    checkCudaErrors(cudaMalloc(&d_A_RowIndices, (m + 1) * sizeof(int)));
        int *d_A_ColIndices;    checkCudaErrors(cudaMalloc(&d_A_ColIndices, nnz * sizeof(int)));
        double *d_x;        checkCudaErrors(cudaMalloc(&d_x, m * sizeof(double)));
        double *d_b; checkCudaErrors(cudaMalloc(&d_b, m * sizeof(double)));
        printf("\nError: %s", cudaGetErrorString(cudaGetLastError()));

        printf("\nCopying data...\n");
        checkCudaErrors(cudaMemcpy(d_A, matA.valuesPtr(), nnz * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_A_RowIndices, matA.RowPtr(), (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_A_ColIndices, matA.ColIdxPtr(), nnz * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_b, matB, m * sizeof(double), cudaMemcpyHostToDevice));

        double *h_x = (double *)malloc(m * sizeof(double));

        printf("\nError: %s", cudaGetErrorString(cudaGetLastError()));
        cudaMemGetInfo(&mem_free, &mem_tot);
        printf("\nFree memory: %d", mem_free);

        int reorder = 0;
        int singularity = 0;
        start = second();
        //checkCudaErrors(cusolverSpDcsrlsvluHost(cusolver_handle, Nrows, nnz, descrA, sparse.Values(),
        	//	sparse.RowPtr(), sparse.ColIdx(), mB.values, tol, reorder, h_x, &singularity));
        checkCudaErrors(cusolverSpDcsrlsvqr(cusolver_handle, m, nnz, descrA, d_A, d_A_RowIndices,
               		d_A_ColIndices, d_b, tol, reorder, d_x, &singularity));
        checkCudaErrors(cudaDeviceSynchronize());
        stop = second();
        time_to_solve = stop - start;


        checkCudaErrors(cudaMemcpy(h_x, d_x, m * sizeof(double), cudaMemcpyDeviceToHost));

//        for (int k=0; k<mA.getNumRows(); k++) solution[k] = h_x[k];


        checkCudaErrors(cusparseDestroy(cusparse_handle));
        checkCudaErrors(cusolverSpDestroy(cusolver_handle));
        checkCudaErrors(cudaStreamDestroy(cudaStream));
        checkCudaErrors(cudaFree(d_b));
        checkCudaErrors(cudaFree(d_x));

        checkCudaErrors(cudaFree(d_A));
        checkCudaErrors(cudaFree(d_A_RowIndices));
        checkCudaErrors(cudaFree(d_A_ColIndices));

        free(h_x);

        std::cerr << "Time to Build in GPU (second): " << time_to_build << std::endl;
        std::cerr << "Time to Solve in GPU (second): " << time_to_solve << std::endl;
        std::cerr << "done!";

    // ****************************************************************************** //
}

int main(int argc, char **argv)
{
	int error;
	int nRows, nCols, nnz;
	int *row, *col;
	double *val, *matB;
	MM_typecode matcode;

	// Read the matrix
	printf("Reading File...\n");
    error = mm_read_mtx_crd(filename, &nRows, &nCols, &nnz, &row, &col, &val, &matcode);

    if (error)
    	exit(-1);

	printf("\nsparse matrix A is %d x %d with %d nonzeros\n", nRows, nCols, nnz);
	printf("Memory usage: %d\n", sizeof(double) * nnz);



    // Create Tuples
    printf("Creating tuples...\n");
    std::vector<int> I, J;
    std::vector<double> V;

    for (int k = 0; k < nnz; k++){
    	double _val = val[k];
    	int i = row[k];
    	int j = col[k];

    	I.push_back(i);
        J.push_back(j);
        V.push_back(_val);
    }

    matB = new double[nRows];

    for(int k = 0 ; k < nRows ; k++)
    {
        matB[k] = 1.0;
    }

    //printf("Solving in Eigen...\n");
    testEigen(nRows, nCols, nnz, I, J, V, matB);

    printf("Solving in GPU...\n");
    testCuda(nRows, nCols, nnz, I, J, V, matB);

    free(row);
    free(col);
    free(val);
    free(matB);

	/*if (loadMMSparseMatrix<double>(filename, 'd', true , &rowsA, &colsA,
	           &nnzA, &h_csrValA, &h_csrRowPtrA, &h_csrColIndA, true)){
		exit (-1);
	}

	baseA = h_csrRowPtrA[0]; // baseA = {0,1}
	printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA, nnzA, baseA);

	h_b = (double*)malloc(sizeof(double)*rowsA);
    printf("step 2: set right hand side vector (b) to 1\n");
    for(int row = 0 ; row < rowsA ; row++)
    {
        h_b[row] = 1.0;
    }

    testEigen(rowsA, colsA, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, h_b);
    testCusolver(rowsA, colsA, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, h_b);

    h_x = (double*)malloc(sizeof(double)*rowsA);

	if (h_csrRowPtrA) free(h_csrRowPtrA);
	if (h_csrColIndA) free(h_csrColIndA);
	if (h_csrValA) free(h_csrValA);
	if (h_b) free(h_b);
	if (h_x) free(h_x);*/

	return 0;
}
