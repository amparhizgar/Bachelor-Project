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



__global__ static void redKernel(double* u, double* un, int n, double lambda) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n && j < n) {
		int index = i * n + j;

		if ((i + j) % 2 == 0) {
			if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
				un[index] = (1-lambda) * u[index] + lambda * 0.25 * (u[(i - 1) * n + j] + u[(i + 1) * n + j]
					+ u[i * n + (j - 1)] + u[i * n + (j + 1)]);
			}
		}
	}
}

__global__ static void blackKernel(double* u, double* un, int n, double lambda) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < n && j < n) {
		int index = i * n + j;

		if ((i + j) % 2 != 0) {
			if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
				un[index] = (1 - lambda) * u[index] + lambda * 0.25 * (un[(i - 1) * n + j] + un[(i + 1) * n + j]
					+ un[i * n + (j - 1)] + un[i * n + (j + 1)]);
			}
		}
	}
}


extern void sor()
{
	const int n = 13;
	int size = n * n;
	thrust::device_vector<double> u_device(size);
	thrust::fill(u_device.begin(), u_device.end(), 0.0);
	thrust::fill(u_device.begin(), u_device.begin() + n, 2.0);
	thrust::fill(u_device.begin() + n * (n - 1), u_device.end(), 1.0);
	thrust::device_vector<double> un_device(u_device);
	double tol = 1e-5;
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((n - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1);

	double lambda = 1.7;


	double error = tol + 1.0;
	int iterations = 0;
	while (error > tol) {
		iterations++;
		redKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(u_device.data()), thrust::raw_pointer_cast(un_device.data()), n, lambda);
		blackKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(u_device.data()), thrust::raw_pointer_cast(un_device.data()), n, lambda);
		
		auto begin = thrust::make_zip_iterator(thrust::make_tuple(u_device.begin(), un_device.begin()));
		auto end = thrust::make_zip_iterator(thrust::make_tuple(u_device.end(), un_device.end()));
		error = thrust::transform_reduce(begin, end, abs_difference(), 0.0, thrust::maximum<double>());
		printf("error is %f\n", error);
		swap(u_device, un_device);
	}

	thrust::host_vector<double> result(u_device);
	print2DArray(thrust::raw_pointer_cast(result.data()), n);
	printf("Finished SOR with lambda=%f\n", lambda);
	printf("total iterations: %d\n", iterations);
}

