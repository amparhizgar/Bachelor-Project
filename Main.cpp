#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/fill.h>
#include <thrust/sequence.h>
#include "util.h"
#include <chrono>
#include <iomanip>

extern thrust::device_vector<double>* jacubi(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc);
extern thrust::device_vector<double>* jacubi_redblack(thrust::device_vector<double>& u, int n, int m, ConvergenceCriteria cc);
extern thrust::device_vector<double>* sor(thrust::device_vector<double>& u, int n, int m, ConvergenceCriteria cc);
extern thrust::device_vector<double>* sor_separated(thrust::device_vector<double>& u, int n, int m, ConvergenceCriteria cc);
extern thrust::device_vector<double>* conjugate_gradient(thrust::device_vector<double>& u, int n, int m, ConvergenceCriteria cc);

void printAlgorithm(std::string name) {
	printf("running %s\n", name.c_str());
}
int getAlg() {
	std::cout << "Enter an algorithm number: ";
	std::string input;
	std::getline(std::cin, input);
	size_t pos;
	if (input.empty()) {
		exit(0);
	}
	int number = std::stoi(input, &pos);

	if (pos == input.length()) {
		return number;
	}
	else {
		exit(0);
	}
}


int main() {
	int n = 20;
	int m = 10;
	int p = 20;
	int size = n * m * p;

	int mode = 1;
	bool plot = true;

	do {
		thrust::device_vector<double> u(size, 0);
		thrust::fill(u.begin(), u.begin() + n * m, -20);
		thrust::fill(u.begin() + n * m * (p - 1), u.end(), 20);
		//thrust::sequence(u.begin(), u.end());

		ConvergenceCriteria cc(0.0, 100);

		thrust::device_vector<double>* (*algorithm)(thrust::device_vector<double>&, int, int, ConvergenceCriteria);
		int selectedAlg;
		if (mode == 0)
			selectedAlg = getAlg();
		else
			selectedAlg = mode;

		switch (selectedAlg)
		{
		case 1:
			//algorithm = &jacubi;
			printAlgorithm("Jacubi");
			break;
		case 2:
			algorithm = &jacubi_redblack;
			printAlgorithm("Jacubi Red Black");
			break;
		case 3:
			algorithm = &sor;
			printAlgorithm("SOR");
			break;
		case 4:
			algorithm = &sor_separated;
			printAlgorithm("SOR Separated");
			break;
		case 5:
			algorithm = &conjugate_gradient;
			printAlgorithm("Conjugate Gradient");
			break;
		default:
			continue;
		}

		auto start = std::chrono::high_resolution_clock::now();

		thrust::device_vector<double> result = *jacubi(u, n, m, p, cc);

		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

		std::cout.imbue(std::locale(""));
		std::cout << "Time taken by the algorithm: "
			<< std::fixed << std::setprecision(3) << duration.count() / 1000.0 << " milliseconds"
			<< std::endl;


		thrust::host_vector<double> host_result(result);

		if (n <= 20 && m <= 20) {
			print2DArray(thrust::raw_pointer_cast(host_result.data()), n, m);
		}

		if (plot) {
			writeToFile(thrust::raw_pointer_cast(host_result.data()), n, m, p);
			std::string scriptPath = ".\\plot3d\\PlotFile.m";

			std::string command = "matlab -batch \"run('" + std::string(scriptPath) + "');\"";
			std::system(command.c_str());

		}
		checkForError();
	} while (mode == 0);
	return 0;
}
