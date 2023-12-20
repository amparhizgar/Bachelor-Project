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


__global__ static void laplacianKernel(double* u, double* unew, int n, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < m && j < n) {
		int index = i * n + j;
		if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
			unew[index] = (u[(i - 1) * n + j] + u[(i + 1) * n + j]
				+ u[i * n + (j - 1)] + u[i * n + (j + 1)]) - 4 * u[index];
		}
	}
}

void print2DArray(thrust::device_vector<double>& v, int n, int m) {
	for (int j = 0; j < m; ++j) {
		for (int i = 0; i < n; ++i) {
			std::cout << std::fixed << v[j * n + i] << "   ";
		}
		std::cout << std::endl;
	}
	printf("-----------\n");
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

extern thrust::device_vector<double>* conjugate_gradient(thrust::device_vector<double>& u, int n, int m)
{
	int size = n * m;
	double tol = 1e-5;

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDim((n - 1) / blockDim.x + 1, (n - 1) / blockDim.y + 1);

	double error = tol + 1.0;
	int iterations = 0;
	thrust::device_vector<double> Ap(size);
	thrust::device_vector<double> p(size);
	thrust::device_vector<double> r(size);
	thrust::device_vector<double> b(size, 0);
	thrust::device_vector<double> temp(size, 0);
	// temp = A * x
	laplacianKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(u.data()), thrust::raw_pointer_cast(temp.data()), n, m);
	// r = b - temp;
	thrust::transform(b.begin(), b.end(), temp.begin(), r.begin(), thrust::minus<int>());
	// p = r;
	thrust::copy(r.begin(), r.end(), p.begin());
	// rDot = r'*r;
	double rDot = dot(r);
	double rDotNew;

	while (error > tol) {
		iterations++;

		// Ap = A*p;
		laplacianKernel << <gridDim, blockDim >> > (thrust::raw_pointer_cast(p.data()), thrust::raw_pointer_cast(Ap.data()), n, m);

		//alpha = rDot / (p'*Ap);
		double alpha = rDot / dot(p, Ap);
		//	x = x + alpha * p;
		sumWithScalarProduct(u, alpha, p);
		//r = r - alpha * Ap;
		sumWithScalarProduct(r, -alpha, Ap);
		if (getError(r) < tol)
			break;

		//	newRDot = r'*r;
		rDotNew = dot(r);

		//	b = newRDot / rDot;
		double beta = rDotNew / rDot;
		//p = r + beta * p;
		sumWithScalarProductRight(r, beta, p);

		//rDot = newRDot;
		rDot = rDotNew;
	}
	return &u;
}

