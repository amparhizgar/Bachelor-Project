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



#define BLOCK_SIZE 16

void print2DArray(const double* array, int n,int m) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			std::cout << std::fixed << array[i * n + j] << "   ";
		}
		std::cout << std::endl;
	}
}

void print2DArray(const double* array, int n) {
	print2DArray(array, n, n);
}


__global__ void redKernel(double* u, double* un, int n, double tol) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n && j < n) {
		int index = i * n + j;

		if ((i + j) % 2 == 0) {
			if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
				un[index] = 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j]
					+ u[i * n + (j - 1)] + u[i * n + (j + 1)]);
			}
		}
	}
}
__global__ void blackKernel(double* un, int n, double tol) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n && j < n) {
		int index = i * n + j;

		if ((i + j) % 2 != 0) {
			if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
				un[index] = 0.25 * (un[(i - 1) * n + j] + un[(i + 1) * n + j]
					+ un[i * n + (j - 1)] + un[i * n + (j + 1)]);
			}
		}
	}
}


struct abs_difference
{
	__host__ __device__
		double operator()(const thrust::tuple<double, double>& t) const {
		return fabs(thrust::get<0>(t) - thrust::get<1>(t));
	}
};

extern void redblack()
{
	const int n = 13;
	int size = n * n;
	thrust::device_vector<double> u_device(size);
	thrust::fill(u_device.begin(), u_device.end(), 0.0);
	thrust::fill(u_device.begin(), u_device.begin() + n, 2.0);
	thrust::fill(u_device.begin() + n * (n-1), u_device.end(), 1.0);
	thrust::device_vector<double> un_device(u_device);
	double tol = 1e-5;

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((n - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1);

	double error = tol + 1.0;
	while (error > tol) {

		redKernel<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(u_device.data()), thrust::raw_pointer_cast(un_device.data()), n, tol);
		blackKernel<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(u_device.data()), n, tol);

		auto begin = thrust::make_zip_iterator(thrust::make_tuple(u_device.begin(), un_device.begin()));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(u_device.end(), un_device.end()));
		error = thrust::transform_reduce(begin, end, abs_difference(), 0.0, thrust::maximum<double>());
		printf("error is %f\n", error);
		swap(u_device, un_device);
		std::cout << "Current Error: " << error << std::endl;
	}

	printf("Finished\n");
	thrust::host_vector<double> result(u_device);
	print2DArray(thrust::raw_pointer_cast(result.data()), n);

}


