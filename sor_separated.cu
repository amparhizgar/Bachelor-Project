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

struct position {
	__device__ position(int _i, int _j, int _k) : i(_i), j(_j), k(_k) {}
	int i;
	int j;
	int k;

	__device__ int indexOfColored(int n, int m, int p) {
		int halfn = (n - 1) / 2 + 1;
		return halfn * m * k + halfn * i + j / 2;
	}
	__device__ int index(int n, int m, int p) {
		return n * m * k + n * i + j;
	}
	__device__ int isBlack() {
		return (i + j + k) % 2;
	}
};

__global__ static void redKernel(double* rnew, double* r, double* b, int n, int m, int p, double lambda) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	position global(i, j * 2 + (i + k) % 2, k);
	if (global.i > 0 && global.i < m - 1 && global.j > 0 && global.j < n - 1 && global.k > 0 && global.k < p - 1) {
		position   left(global.i, global.j - 1, global.k);
		position  right(global.i, global.j + 1, global.k);
		position    top(global.i - 1, global.j, global.k);
		position bottom(global.i + 1, global.j, global.k);
		position   back(global.i, global.j, global.k - 1);
		position  front(global.i, global.j, global.k + 1);

		rnew[global.indexOfColored(n, m, p)] = (1 - lambda) * r[global.indexOfColored(n, m, p)] + lambda / 6.0 *
			(b[left.indexOfColored(n, m, p)] + b[right.indexOfColored(n, m, p)] +
				b[top.indexOfColored(n, m, p)] + b[bottom.indexOfColored(n, m, p)] +
				b[back.indexOfColored(n, m, p)] + b[front.indexOfColored(n, m, p)]);
	}
}

__global__ static void blackKernel(double* bnew, double* b, double* r, int n, int m, int p, double lambda) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	position global(i, j * 2 + (i + k + 1) % 2, k);
	if (global.i > 0 && global.i < m - 1 && global.j > 0 && global.j < n - 1 && global.k > 0 && global.k < p - 1) {
		position left(global.i, global.j - 1, global.k);
		position right(global.i, global.j + 1, global.k);
		position top(global.i - 1, global.j, global.k);
		position bottom(global.i + 1, global.j, global.k);
		position back(global.i, global.j, global.k - 1);
		position front(global.i, global.j, global.k + 1);
		bnew[global.indexOfColored(n, m, p)] = (1 - lambda) * b[global.indexOfColored(n, m, p)] + lambda / 6.0 *
			(r[left.indexOfColored(n, m, p)] + r[right.indexOfColored(n, m, p)] +
				r[top.indexOfColored(n, m, p)] + r[bottom.indexOfColored(n, m, p)] +
				r[back.indexOfColored(n, m, p)] + r[front.indexOfColored(n, m, p)]);
	}
}

__global__ static void initRedKernel(double* red, double* u, int n, int m, int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	position global(i, j * 2 + (i + k) % 2, k);
	if (global.i < m && global.j < n && global.k < p)
		red[global.indexOfColored(n, m, p)] = u[global.index(n, m, p)];
}

__global__ static void initBlackKernel(double* black, double* u, int n, int m, int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	position global(i, j * 2 + (i + k + 1) % 2, k);
	if (global.i < m && global.j < n && global.k < p)
		black[global.indexOfColored(n, m, p)] = u[global.index(n, m, p)];
}

__global__ static void joinKernel(double* red, double* black, double* u, int n, int m, int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	position global(i, j, k);
	if (i < m && j < n && k < p) {
		if (global.isBlack()) {
			u[global.index(n, m, p)] = black[global.indexOfColored(n, m, p)];
		}
		else {
			u[global.index(n, m, p)] = red[global.indexOfColored(n, m, p)];
		}
	}
}
__global__ static void errorKernel(double* r, double* rnew, double* b, double* bnew, double* error, int sizer) {
	int i = blockIdx.z * blockDim.y * blockDim.x + 
		blockIdx.y * blockDim.x +
		threadIdx.x;
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

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((m - 1) / blockDim.x + 1, (halfn - 1) / blockDim.y + 1, (p - 1) / blockDim.z + 1);

	thrust::device_vector<double> r(halfn * m * p);
	thrust::device_vector<double> b(halfn * m * p);

	initRedKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(r.data()), thrust::raw_pointer_cast(u.data()), n, m, p);
	initBlackKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(u.data()), n, m, p);

	thrust::device_vector<double> bnew(b);
	thrust::device_vector<double> rnew(r);
	thrust::device_vector<double> error_temp(halfn * m * p);


	double lambda = 2 / (1 + sin(pi / (n + 1)));
	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	double error;
	int iterations = 0;
	while (true) {
		iterations++;
		redKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(rnew.data()), thrust::raw_pointer_cast(r.data()), thrust::raw_pointer_cast(b.data()), n, m, p, lambda);
		blackKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(bnew.data()), thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(rnew.data()), n, m, p, lambda);

		errorKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(r.data()), thrust::raw_pointer_cast(rnew.data()),
			thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(bnew.data()), thrust::raw_pointer_cast(error_temp.data()), halfn * m * p);
		checkForError();
		error = thrust::reduce(error_temp.begin(), error_temp.end(), 0.0, thrust::maximum<double>());
		swap(r, rnew);
		swap(b, bnew);
		if (cc.hasConverged(error, iterations))
			break;
	}

	dim3 joinGridDim((m - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1, (p - 1) / blockDim.z + 1);
	joinKernel << <joinGridDim, blockDim >> > (thrust::raw_pointer_cast(r.data()), thrust::raw_pointer_cast(b.data()), thrust::raw_pointer_cast(u.data()), n, m, p);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);	
	
	return &u;
}

