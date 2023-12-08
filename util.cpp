#include "cuda_runtime.h"

#include "util.h"


void print2DArray(const double* array, int n, int m) {
	for (int j = 0; j < m; ++j) {
		for (int i = 0; i < n; ++i) {
			std::cout << std::fixed << array[j * n + i] << "   ";
		}
		std::cout << std::endl;
	}
}

void print2DArray(const double* array, int n) {
	print2DArray(array, n, n);
}

