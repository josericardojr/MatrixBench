/*
 * MatSparse.cpp
 *
 *  Created on: Apr 26, 2016
 *      Author: jricardo
 */

#include "MatSparse.h"

 void MatSparse::fromTruples(std::vector<int>& I, std::vector<int>& J, std::vector<double>& V){

#ifdef ORDERING_IN_GPU
	 printf("\nOrdering in GPU");
	 size_t mem_tot = 0;
	 size_t mem_free = 0;

	 cudaMemGetInfo(&mem_free, & mem_tot);
	 printf("\nFree memory: %d", mem_free);

	thrust::device_vector<int> d_h_I(I.begin(), I.end());
	thrust::device_vector<int> d_h_J(J.begin(), J.end());
	thrust::device_vector<double> d_h_V(V.begin(), V.end());
#else
	printf("\nOrdering in CPU");
 	thrust::host_vector<int> d_h_I(I.begin(), I.end());
	thrust::host_vector<int> d_h_J(J.begin(), J.end());
	thrust::host_vector<double> d_h_V(V.begin(), V.end());
#endif

	try {
	printf("\nSorting by key...");
	 cudaMemGetInfo(&mem_free, & mem_tot);
	 printf("\nFree memory: %d", mem_free);

	thrust::stable_sort_by_key(d_h_J.begin(), d_h_J.end(), thrust::make_zip_iterator(
			thrust::make_tuple(d_h_I.begin(), d_h_V.begin())));
	thrust::stable_sort_by_key(d_h_I.begin(), d_h_I.end(), thrust::make_zip_iterator(
			thrust::make_tuple(d_h_J.begin(), d_h_V.begin())));

	printf("\nInner product...");
	 cudaMemGetInfo(&mem_free, & mem_tot);
	 printf("\nFree memory: %d", mem_free);

	num_elements = thrust::inner_product(thrust::make_zip_iterator(thrust::make_tuple(d_h_I.begin(), d_h_J.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_h_I.end (), d_h_J.end())) - 1,
			thrust::make_zip_iterator(thrust::make_tuple(d_h_I.begin(), d_h_J.begin())) + 1,
			int(0),
			thrust::plus<int>(),
			thrust::not_equal_to< thrust::tuple<int,int> >()) + 1;
	} catch(std::bad_alloc &e)
	{
	    std::cout << e.what() << std::endl;
	    cudaGetLastError();
	    cudaDeviceReset();
	    cudaSetDevice(0);
	    exit(-3);
	}


#ifdef ORDERING_IN_GPU
	try{
	printf("\nAlloc Memory...");
	 cudaMemGetInfo(&mem_free, & mem_tot);
	 printf("\nFree memory: %d", mem_free);
	cusp::array1d<int, cusp::device_memory> d_rowCOO(num_elements);
    thrust::device_vector<double> d_valuesCSR(num_elements);
    thrust::device_vector<int> d_colCSR(num_elements);
    thrust::device_vector<int> d_rowCSR(numRows + 1);

	printf("\nReducing by key...");
	 cudaMemGetInfo(&mem_free, & mem_tot);
	 printf("\nFree memory: %d", mem_free);
	thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(d_h_I.begin(), d_h_J.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_h_I.end(), d_h_J.end())),
			d_h_V.begin(),
			thrust::make_zip_iterator(thrust::make_tuple(d_rowCOO.begin(), d_colCSR.begin())),
			d_valuesCSR.begin(),
			thrust::equal_to< thrust::tuple<int,int> >(),
			thrust::plus<double>());

	printf("\nCopying data...");
	 cudaMemGetInfo(&mem_free, & mem_tot);
	 printf("\nFree memory: %d", mem_free);
	rowCSR.resize(numRows + 1);
	cusp::array1d<int, cusp::device_memory> d_rowOffset(numRows + 1);
	cusp::detail::indices_to_offsets(d_rowCOO,d_rowOffset);
	thrust::copy(d_rowOffset.begin(), d_rowOffset.end(), rowCSR.begin());

	valuesCSR = d_valuesCSR;
	colCSR = d_colCSR;
	} catch(std::bad_alloc &e)
	{
	    std::cout << e.what() << std::endl;
	    cudaGetLastError();
	    cudaDeviceReset();
	    cudaSetDevice(0);
	    exit(-3);
	}
#else
	cusp::array1d<int, cusp::host_memory> h_rowCOO(num_elements);
	valuesCSR.resize(num_elements);
	colCSR.resize(num_elements);
	rowCSR.resize(numRows + 1);

	thrust::reduce_by_key(thrust::make_zip_iterator(thrust::make_tuple(d_h_I.begin(), d_h_J.begin())),
			thrust::make_zip_iterator(thrust::make_tuple(d_h_I.end(), d_h_J.end())),
			d_h_V.begin(),
			thrust::make_zip_iterator(thrust::make_tuple(h_rowCOO.begin(), colCSR.begin())),
			valuesCSR.begin(),
			thrust::equal_to< thrust::tuple<int,int> >(),
			thrust::plus<double>());

	cusp::array1d<int, cusp::host_memory> rowOffset(numRows + 1);
	cusp::detail::indices_to_offsets(h_rowCOO, rowOffset);
	thrust::copy(rowOffset.begin(), rowOffset.end(), rowCSR.begin());
#endif
 }
