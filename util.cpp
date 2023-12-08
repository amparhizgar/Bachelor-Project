#include "cuda_runtime.h"

#include "util.h"


void print2DArray(const double* array, int n, int m) {
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

