﻿#include <stdio.h>
#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include "util.h"


__global__ void redKernel(double* u, double* un, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int index = i * n + j;

	if ((i + j) % 2 == 0) {
		if (i > 0 && i < m - 1 && j > 0 && j < n - 1) {
			un[index] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j]
				+ u[i * n + (j - 1)] + u[i * n + (j + 1)]);
		}
	}
}
__global__ void blackKernel(double* un, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int index = i * n + j;

	if ((i + j) % 2 != 0) {
		if (i > 0 && i < m - 1 && j > 0 && j < n - 1) {
			un[index] = 0.25 * (un[(i - 1) * n + j] + un[(i + 1) * n + j]
				+ un[i * n + (j - 1)] + un[i * n + (j + 1)]);
		}
	}
}



extern thrust::device_vector<double>* jacubi_redblack(thrust::device_vector<double>& u, int n, int m, ConvergenceCriteria cc)
{
	int size = n * m;
	thrust::device_vector<double> un(u);

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((m - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1);

	double error;
	int iterations = 0;
	while (true) {
		iterations++;

		redKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(u.data()), thrust::raw_pointer_cast(un.data()), n, m);
		checkForError();

		blackKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(un.data()), n, m);
		checkForError();

		auto begin = thrust::make_zip_iterator(thrust::make_tuple(u.begin(), un.begin()));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(u.end(), un.end()));

		error = thrust::transform_reduce(begin, end, abs_difference(), 0.0, thrust::maximum<double>());
		swap(u, un);
		if (cc.hasConverged(error, iterations))
			break;
	}
	return &u;
}

