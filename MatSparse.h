/*
 * MatSparse.h
 *
 *  Created on: Apr 26, 2016
 *      Author: jricardo
 */

#ifndef MATSPARSE_H_
#define MATSPARSE_H_

#define ORDERING_IN_GPU

#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <vector>
#include <cusp/array1d.h>
#include <cusp/coo_matrix.h>
#include <cusp/format.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>


class MatSparse {
private:
    thrust::host_vector<double> valuesCSR;
    thrust::host_vector<int> colCSR;
    thrust::host_vector<int> rowCSR;
    int numRows;
    int numCols;
    int num_elements;

public:

    int getNumRows(){ return numRows; }
    int getNumCols(){ return numCols; }

    double *valuesPtr(){
    	return thrust::raw_pointer_cast(&valuesCSR[0]);
    }

    int *ColIdxPtr(){
    	return thrust::raw_pointer_cast(&colCSR[0]);
    }

    int* RowPtr(){ return thrust::raw_pointer_cast(&rowCSR[0]); }

    int getNonZeroValues(){ return num_elements; }

    MatSparse (){
        numRows = -1;
        numCols = -1;
    }

    void setSize(int rows, int cols){ numRows = rows; numCols = cols; }

    MatSparse (int rows, int cols){
        numRows = rows;
        numCols = cols;
    }

    void fromTruples(std::vector<int>& I, std::vector<int>& J, std::vector<double>& V);

};

#endif /* MATSPARSE_H_ */
