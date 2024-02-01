#include <stdio.h>
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

__device__ static int indexof(int i, int j, int k, int n, int m, int p) {
	return k * n * m + i * n + j;
}

__global__ void redKernel(double* u, double* un, int n, int m, int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int index = k * n * m + i * n + j;

	if ((i + j + k) % 2 == 0) {
		if (i > 0 && i < m - 1 && j > 0 && j < n - 1 && k > 0 && k < p - 1) {
			un[index] = 1.0 / 6.0 * (u[indexof(i - 1, j, k, n, m, p)] + u[indexof(i + 1, j, k, n, m, p)]
				+ u[indexof(i, j - 1, k, n, m, p)] + u[indexof(i, j + 1, k, n, m, p)]
				+ u[indexof(i, j, k - 1, n, m, p)] + u[indexof(i, j, k + 1, n, m, p)]);
		}
	}
}
__global__ void blackKernel(double* un, int n, int m, int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int index = k * n * m + i * n + j;

	if ((i + j + k) % 2 != 0) {
		if (i > 0 && i < m - 1 && j > 0 && j < n - 1 && k > 0 && k < p - 1) {
			un[index] = 1.0 / 6.0 * (un[indexof(i - 1, j, k, n, m, p)] + un[indexof(i + 1, j, k, n, m, p)]
				+ un[indexof(i, j - 1, k, n, m, p)] + un[indexof(i, j + 1, k, n, m, p)]
				+ un[indexof(i, j, k - 1, n, m, p)] + un[indexof(i, j, k + 1, n, m, p)]);
		}
	}
}



extern Result jacubi_redblack(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc)
{
	thrust::device_vector<double> un(u);

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((m - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1, (p - 1) / blockDim.z + 1);

	double error;
	int iterations = 0;
	while (true) {
		iterations++;

		redKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(u.data()), thrust::raw_pointer_cast(un.data()), n, m, p);
		checkForError();

		blackKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(un.data()), n, m, p);
		checkForError();

		auto begin = thrust::make_zip_iterator(thrust::make_tuple(u.begin(), un.begin()));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(u.end(), un.end()));

		error = thrust::transform_reduce(begin, end, abs_difference(), 0.0, thrust::maximum<double>());
		swap(u, un);
		if (cc.hasConverged(error, iterations))
			break;
	}
	return Result(&u, error, iterations);
}

