#include <stdio.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>

#include "util.h"


__global__ static void laplacianKernel(double* u, double* unew, int n, int m, int p) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < m && j < n && k < p) {
		int index = k * n * m + i * n + j;
		if (i > 0 && i < m - 1 && j > 0 && j < n - 1 && k > 0 && k < p - 1) {
			unew[index] = (
				u[k * n * m + (i - 1) * n + j] + u[k * n * m + (i + 1) * n + j]
				+ u[k * n * m + i * n + (j - 1)] + u[k * n * m + i * n + (j + 1)]
				+ u[(k - 1) * n * m + i * n + j] + u[(k + 1) * n * m + i * n + j])
				- 6 * u[index];
		}
	}
}

double dot(thrust::device_vector<double>& v, thrust::device_vector<double>& u) {
	return thrust::inner_product(v.begin(), v.end(), u.begin(), 0.0, thrust::plus<double>(), thrust::multiplies<double>());
}

double dot(thrust::device_vector<double>& v) {
	return  dot(v, v);
}
struct Multiply {
	const double factor;

	Multiply(double _factor) : factor(_factor) {}

	__host__ __device__
		double operator()(const double& x) const {
		return x * factor;
	}
};
void multiplyVector(double d, thrust::device_vector<double>& vec) {
	thrust::transform(vec.begin(), vec.end(), vec.begin(), Multiply(d));
}
struct SumWithScalarProduct {
	const double scalar;

	SumWithScalarProduct(double _scalar) : scalar(_scalar) {}

	__host__ __device__
		double operator()(const double& v, const double& u) const {
		return v + (scalar * u);
	}
};

void sumWithScalarProduct(thrust::device_vector<double>& v, double d, thrust::device_vector<double>& u) {
	thrust::transform(v.begin(), v.end(), u.begin(), v.begin(), SumWithScalarProduct(d));
}
void sumWithScalarProductRight(thrust::device_vector<double>& v, double d, thrust::device_vector<double>& u) {
	thrust::transform(v.begin(), v.end(), u.begin(), u.begin(), SumWithScalarProduct(d));
}


double getError(thrust::device_vector<double>& v, thrust::device_vector<double>& u) {
	auto begin = thrust::make_zip_iterator(thrust::make_tuple(v.begin(), u.begin()));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(v.end(), u.end()));
	return thrust::transform_reduce(begin, end, abs_difference(), 0.0, thrust::maximum<double>());
}


double getError(thrust::device_vector<double>& vec) {
	return thrust::transform_reduce(vec.begin(), vec.end(), AbsoluteValue(), 0.0, thrust::maximum<double>());
}

extern thrust::device_vector<double>* conjugate_gradient(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc)
{
	int size = n * m * p;

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((m - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1, (p - 1) / blockDim.z + 1);

	int iterations = 0;
	thrust::device_vector<double> Ap(size);
	thrust::device_vector<double> p_array(size);
	thrust::device_vector<double> r(size);
	thrust::device_vector<double> b(size, 0);
	thrust::device_vector<double> temp(size, 0);
	// temp = A * x
	laplacianKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(u.data()), thrust::raw_pointer_cast(temp.data()), n, m, p);
	// r = b - temp;
	thrust::transform(b.begin(), b.end(), temp.begin(), r.begin(), thrust::minus<int>());
	// p = r;
	thrust::copy(r.begin(), r.end(), p_array.begin());
	// rDot = r'*r;
	double rDot = dot(r);
	double rDotNew;

	while (true) {
		iterations++;

		// Ap = A*p;
		laplacianKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(p_array.data()), thrust::raw_pointer_cast(Ap.data()), n, m, p);
		checkForError();

		//alpha = rDot / (p'*Ap);
		double alpha = rDot / dot(p_array, Ap);
		//	x = x + alpha * p;
		sumWithScalarProduct(u, alpha, p_array);
		//r = r - alpha * Ap;
		sumWithScalarProduct(r, -alpha, Ap);
		if (cc.hasConverged(getError(r), iterations))
			break;
		//	newRDot = r'*r;
		rDotNew = dot(r);

		//	b = newRDot / rDot;
		double beta = rDotNew / rDot;
		//p = r + beta * p;
		sumWithScalarProductRight(r, beta, p_array);

		//rDot = newRDot;
		rDot = rDotNew;
	}
	return &u;
}

