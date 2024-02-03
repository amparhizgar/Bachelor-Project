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

__global__ static void redKernel(double* u, double* un, int n, int m, int p, double lambda) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int index = k * n * m + i * n + j;

	if ((i + j + k) % 2 == 0) {
		if (i > 0 && i < m - 1 && j > 0 && j < n - 1 && k > 0 && k < p - 1) {
			un[index] = (1 - lambda) * u[index] + lambda / 6.0 * (u[indexof(i - 1, j, k, n, m, p)] + u[indexof(i + 1, j, k, n, m, p)]
				+ u[indexof(i, j - 1, k, n, m, p)] + u[indexof(i, j + 1, k, n, m, p)]
				+ u[indexof(i, j, k - 1, n, m, p)] + u[indexof(i, j, k + 1, n, m, p)]);
		}
	}
}

__global__ static void blackKernel(double* u, double* un, int n, int m, int p, double lambda) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int index = k * n * m + i * n + j;

	if ((i + j + k) % 2 != 0) {
		if (i > 0 && i < m - 1 && j > 0 && j < n - 1 && k > 0 && k < p - 1) {
			un[index] = (1 - lambda) * u[index] + lambda / 6.0 * (un[indexof(i - 1, j, k, n, m, p)] + un[indexof(i + 1, j, k, n, m, p)]
				+ un[indexof(i, j - 1, k, n, m, p)] + un[indexof(i, j + 1, k, n, m, p)]
				+ un[indexof(i, j, k - 1, n, m, p)] + un[indexof(i, j, k + 1, n, m, p)]);
		}
	}
}

__global__ static void errorKernel(double* u, double* un, double* error, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		error[index] = fabs(u[index] - un[index]);
}

extern Result sor(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc)
{
	int size = n * m * p;
	thrust::device_vector<double> un(u);
	thrust::device_vector<double> error_temp(size);

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((m - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1, (p - 1) / blockDim.z + 1);

	double lambda = 2 / (1 + sin(pi / (n + 1)));

	double error;
	int iterations = 0;
	while (true) {
		iterations++;
		redKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(u.data()), thrust::raw_pointer_cast(un.data()), n, m, p, lambda);
		blackKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(u.data()), thrust::raw_pointer_cast(un.data()), n, m, p, lambda);

		errorKernel << <(size - 1) / 128 + 1, 128 >> > (thrust::raw_pointer_cast(u.data()), thrust::raw_pointer_cast(un.data()),
			thrust::raw_pointer_cast(error_temp.data()), size);
		checkForError();
		error = thrust::reduce(error_temp.begin(), error_temp.end(), 0.0, thrust::maximum<double>());
		swap(u, un);
		if (cc.hasConverged(error, iterations))
			break;
	}

	return Result(&u, error, iterations);
}

