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
extern thrust::device_vector<double>* jacubi_redblack(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc);
extern thrust::device_vector<double>* sor(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc);
extern thrust::device_vector<double>* sor_half_thread(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc);
extern thrust::device_vector<double>* sor_separated(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc);
extern thrust::device_vector<double>* conjugate_gradient(thrust::device_vector<double>& u, int n, int m, int p, ConvergenceCriteria cc);

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
	int n = 200;
	int m = 100;
	int p = 200;
	int size = n * m * p;

	int mode = 0;
	bool plot = false;

	do {
		thrust::device_vector<double> u(size, 0);
		thrust::fill(u.begin(), u.begin() + n * m, -20);
		thrust::fill(u.begin() + n * m * (p - 1), u.end(), 20);
		//thrust::sequence(u.begin(), u.end());

		ConvergenceCriteria cc(0.0, 1000);

		thrust::device_vector<double>* (*algorithm)(thrust::device_vector<double>&, int, int, int, ConvergenceCriteria);
		int selectedAlg;
		std::string algorithm_name;
		if (mode == 0)
			selectedAlg = getAlg();
		else
			selectedAlg = mode;

		switch (selectedAlg)
		{
		case 1:
			algorithm = &jacubi;
			algorithm_name = "Jacubi";
			break;
		case 2:
			algorithm = &jacubi_redblack;
			algorithm_name = "Jacubi Red Black";
			break;
		case 3:
			algorithm = &sor;
			algorithm_name = "SOR";
			break;
		case 4:
			algorithm = &sor_half_thread;
			algorithm_name = "SOR hlaf thread";
			break;
		case 5:
			algorithm = &sor_separated;
			algorithm_name = "SOR Separated";
			break;
		case 6:
			algorithm = &conjugate_gradient;
			algorithm_name = "Conjugate Gradient";
			break;
		default:
			continue;
		}
		printAlgorithm(algorithm_name);

		auto start = std::chrono::high_resolution_clock::now();

		thrust::device_vector<double> result = *algorithm(u, n, m, p, cc);

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
			std::cout << "writing to file...\n";
 			writeToFile(thrust::raw_pointer_cast(host_result.data()), n, m, p, algorithm_name, duration.count() / 1000, -1);
			std::string scriptPath = ".\\plot3d\\PlotFile.m";

			std::string command = "matlab -nosplash -nodesktop -r \"run('" + std::string(scriptPath) + "');quit();\"";
			std::cout << "opening matlab...\n";
			std::system(command.c_str());

		}
		checkForError();
	} while (mode == 0);
	return 0;
}
