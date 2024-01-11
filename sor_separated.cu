#include <stdio.h>
#include <iostream>
#include <cmath>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/iterator_facade.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include "util.h"



__global__ static void redKernel(double* rnew, double* r, double* b, int n, int m, double lambda) {
	const int halfn = (n - 1) / 2 + 1;
	const int is_odd = n & 1;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int realj = j * 2 + (i & 1);
	if (i > 0 && i < m - 1 && j + (i & 1) > 0 && realj < n - 1 && (!is_odd || realj < n - 2 || (i & 1))) {
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
	if (i > 0 && i < m - 1 && j + ((i + 1) & 1) > 0 && realj < n - 1 && (!is_odd || realj < n - 2 || ((i + 1) & 1))) {
		int left = i * halfn + j - (i & 1);
		int right = i * halfn + j + 1 - (i & 1);
		int top = (i - 1) * halfn + j;
		int bottom = (i + 1) * halfn + j;
		bnew[i * halfn + j] = (1 - lambda) * b[i * halfn + j] + lambda * 0.25 * (r[left] + r[right]
			+ r[top] + r[bottom]);
	}
}

__global__ static void initRedKernel(double* red, double* u, int n, int m) {
	const int halfn = (n - 1) / 2 + 1;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (2 * j + (i & 1) < n && i < m)
		red[i * halfn + j] = u[i * n + 2 * j + (i & 1)];
}

__global__ static void initBlackKernel(double* black, double* u, int n, int m) {
	const int halfn = (n - 1) / 2 + 1;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (2 * j + ((i + 1) & 1) < n && i < m)
		black[i * halfn + j] = u[i * n + 2 * j + ((i + 1) & 1)];
}

__global__ static void joinKernel(double* red, double* black, double* u, int n, int m) {
	const int halfn = (n - 1) / 2 + 1;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < m && j < n) {
		if ((i + j & 1) == 0) {
			u[i * n + j] = red[i * halfn + j / 2];
		}
		else {
			u[i * n + j] = black[i * halfn + j / 2];
		}
	}
}
__global__ static void errorKernel(double* r, double* rnew, double* b, double* bnew, double* error, int sizer) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < sizer) {
		error[i] = fmax(fabs(r[i] - rnew[i]), fabs(b[i] - bnew[i]));
	}
}

struct Concatenator {
	const double* vector1;
	const double* vector2;
	size_t size1;
	size_t size2;

	Concatenator(const double* v1, const double* v2, size_t s1, size_t s2)
		: vector1(v1), vector2(v2), size1(s1), size2(s2) {}

	__host__ __device__
		double operator()(int idx) const {
		if (idx < 0 || idx >= size1 + size2) {
			printf("Out-of-bounds access at index %d\n", idx);
			return 0;
		}
		return idx < size1 ? vector1[idx] : vector2[idx - size1];
	}
};

extern thrust::device_vector<double>* sor_separated(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc)
{
	const int halfn = (n - 1) / 2 + 1;

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((m - 1) / blockDim.x + 1, (halfn - 1) / blockDim.y + 1);

	thrust::device_vector<double> r(halfn * m);
	thrust::device_vector<double> b(halfn * m);

	initRedKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(r.data()), thrust::raw_pointer_cast(u.data()), n, m);
	initBlackKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(u.data()), n, m);

	thrust::device_vector<double> bnew(b);
	thrust::device_vector<double> rnew(r);
	thrust::device_vector<double> error_temp(halfn * m);


	double lambda = 2 / (1 + sqrt(1 - pow(cos(pi / (n - 1)), 2)));
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	double error;
	int iterations = 0;
	while (true) {
		iterations++;
		redKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(rnew.data()), thrust::raw_pointer_cast(r.data()), thrust::raw_pointer_cast(b.data()), n, m, lambda);
		blackKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(bnew.data()), thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(rnew.data()), n, m, lambda);

		errorKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(r.data()), thrust::raw_pointer_cast(rnew.data()),
			thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(bnew.data()), thrust::raw_pointer_cast(error_temp.data()), halfn * m);
		checkForError();
		error = thrust::reduce(error_temp.begin(), error_temp.end(), 0.0, thrust::maximum<double>());
		swap(r, rnew);
		swap(b, bnew);
		if (cc.hasConverged(error, iterations))
			break;
	}

	dim3 joinGridDim((m - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1);
	joinKernel << <joinGridDim, blockDim >> > (thrust::raw_pointer_cast(r.data()), thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(u.data()), n, m);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	return &u;
}

