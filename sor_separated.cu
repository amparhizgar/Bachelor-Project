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
#include <thrust/sequence.h>

#include "util.h"


#define BLOCK_SIZE 16


__global__ static void redKernel(double* rnew, double* r, double* b, int n, int m, double lambda) {
	const int halfn = (n - 1) / 2 + 1;
	const int is_odd = n & 1;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int realj = j * 2 + (i & 1);
	if (i > 0 && i < m - 1 && j + (i & 1) > 0 && realj< n - 1 && (!is_odd || realj < n - 2 || (i & 1))) {
		int left = i * halfn + j - 1 + (i & 1);
		int right = i * halfn + j + (i & 1);
		int top = (i - 1) * halfn + j;
		int bottom = (i + 1) * halfn + j;
		rnew[i * halfn + j] = (1 - lambda) * r[i * halfn + j] + lambda * 0.25 * (b[left] + b[right] 
			+ b[top] + b[bottom]);
	}
}

__global__ static void blackKernel(double* bnew, double* b, double* r, int n, int m, double lambda) {
	const int halfn = (n - 1) / 2 + 1;
	const int is_odd = n & 1;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int realj = j * 2 + ((i + 1) & 1);
	if (i > 0 && i < m - 1 && j + ((i + 1) & 1) > 0 && realj < n-1  && (!is_odd || realj < n - 2 || ((i + 1) & 1))) {
		int left = i * halfn + j - (i & 1);
		int right = i * halfn + j + 1 - (i & 1);
		int top = (i - 1) * halfn + j;
		int bottom = (i + 1) * halfn + j;
		bnew[i * halfn + j] = (1 - lambda) * b[i * halfn + j] + lambda * 0.25 * (r[left] + r[right] 
			+ r[top] + r[bottom]);
	}
}

double calculate_error(thrust::device_vector<double> olddata, thrust::device_vector<double> newdata) {
	auto begin = thrust::make_zip_iterator(thrust::make_tuple(olddata.begin(), newdata.begin()));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(olddata.end(), newdata.end()));
	return thrust::transform_reduce(begin, end, abs_difference(), 0.0, thrust::maximum<double>());
}


extern void sor_separated()
{
	const int n = 13;
	const int m = n;
	int size = n * m;

	const int halfn = (n - 1) / 2 + 1;
	const int is_odd = n & 1;


	thrust::host_vector<double> u(size);

	thrust::fill(u.begin(), u.end(), 0.0);
	thrust::fill(u.begin(), u.begin() + n, 2.0);
	thrust::fill(u.begin() + n * (m - 1), u.end(), 1.0);

	thrust::host_vector<double> h_red(halfn * m);
	thrust::host_vector<double> h_black(halfn * m);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < halfn; j++) {
			if (2 * j + (i & 1) < n)
				h_red[i * halfn + j] = u[i * n + 2 * j + (i & 1)];
		}
	}

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < halfn; j++) {
			if (2 * j + ((i + 1) & 1) < n)
				h_black[i * halfn + j] = u[i * n + 2 * j + ((i + 1) & 1)];
		}
	}

	thrust::device_vector<double> b(h_black);
	thrust::device_vector<double> bnew(b);
	thrust::device_vector<double> r(h_red);
	thrust::device_vector<double> rnew(r);


	double tol = 1e-5;
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((m - 1) / blockDim.x + 1, (halfn - 1) / blockDim.y + 1);

	double lambda = 1.7;


	double error = tol + 1.0;
	int iterations = 0;
	while (error > tol) {
		iterations++;
		redKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(rnew.data()), thrust::raw_pointer_cast(r.data()), thrust::raw_pointer_cast(b.data()), n, m, lambda);
		blackKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(bnew.data()), thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(rnew.data()), n, m, lambda);

		error = fmax(calculate_error(r, rnew), calculate_error(b, bnew));
		printf("error is %f\n", error);
		swap(r, rnew);
		swap(b, bnew);
	}

	thrust::host_vector<double> result_red(r);
	thrust::host_vector<double> result_black(b);
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			if ((i + j & 1) == 0) {
				u[i * n + j] = result_red[i * halfn + j / 2];
			}
			else {
				u[i * n + j] = result_black[i * halfn + j / 2];
			}
		}
	}
	print2DArray(thrust::raw_pointer_cast(u.data()), n, m);
	printf("Finished SOR with separation. lambda=%f\n", lambda);
	printf("total iterations: %d\n", iterations);
}

