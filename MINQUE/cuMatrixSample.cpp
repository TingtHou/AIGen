#include "pch.h"
#include "cuMatrixSample.h"

void cuMatrixInv_timetest(int N)
{
	MatrixXd A = MatrixXd::Random(N, N);
	std::cout << "\\\\\\\\\\\\\\\\\\Calculate Inverse Matrix Using GPU\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	double *a = A.data();
	double *p = (double*)malloc(N*N * sizeof(double));
	memset(p, 0, N*N * sizeof(double));
	clock_t t2 = clock();
	cuMatrixInv(a, p, N);
	std::cout << "GPU Elapse Time : " << (clock() - t2) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;
	MatrixXd NewC = Eigen::Map<Eigen::MatrixXd>(p, N, N);
	std::cout << "\\\\\\\\\\\\\\\\\\Calculate Inverse Matrix Using Eigen\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	clock_t t1 = clock();
	MatrixXd Inv = A.inverse();
	std::cout << "CPU Elapse Time : " << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << " ms" << std::endl;

	if ((Inv - NewC).norm() < 10e-10)
	{
		std::cout << "Equality test passed." << std::endl;
	}
	else
	{
		std::cout << "Equality test failed." << std::endl;

	}
	free(p);
}

void cuMatrixINVTest(int N)
{
	MatrixXd A = MatrixXd::Random(N, N);
	std::cout << "\\\\\\\\\\\\\\\\\\Origin Matrix\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	std::cout << A << std::endl;
	std::cout << "\\\\\\\\\\\\\\\\\\Inversed Matrix using Eigen\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;

	MatrixXd Inv = A.inverse();
	std::cout << "\\\\\\\\\\\\\\\\\\\\A*A_INV using Eigen\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	std::cout << A * Inv << std::endl;
	std::cout << "\\\\\\\\\\\\\\\\\\\\Inversed Matrix using GPU\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	double *a = A.data();
	double *p = (double*)malloc(N*N * sizeof(double));
	memset(p, 0, N*N * sizeof(double));

	cuMatrixInv(a, p, N);
	MatrixXd NewC = Eigen::Map<Eigen::MatrixXd>(p, N, N);
	std::cout << NewC << std::endl;
	double *E = (double*)malloc(N*N * sizeof(double));
	cuMatrixMult(a, p, E, N, N, N);
	MatrixXd Ematrix = Eigen::Map<Eigen::MatrixXd>(E, N, N);
	std::cout << "\\\\\\\\\\\\\\\\\\\\A*A_INV using GPU\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	std::cout << Ematrix << std::endl;

}

void cuMatrixMultTest(int M,int N,int S)
{
	MatrixXd A = MatrixXd::Random(M, N);
	MatrixXd B = MatrixXd::Random(N, S);
	std::cout << "Simulation data generated" << std::endl;
	std::cout << "Starting CPU analysis" << std::endl;
	clock_t t1 = clock();
	MatrixXd C = A * B;
	std::cout << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;
	std::cout << C << std::endl;
	std::cout << "\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" << std::endl;
	std::cout << "Starting GPU analysis" << std::endl;
	// 	double **a = (double **)malloc(M * sizeof(double*));
	// 	for (int i = 0; i < M; i++)
	// 	{
	// 		a[i] = (double *)malloc(N * sizeof(double));
	// 	}
	// 	double **b = (double **)malloc(N * sizeof(double*));
	// 	for (int i = 0; i < N; i++)
	// 	{
	// 		b[i] = (double *)malloc(S * sizeof(double));
	// 	}
	// 	double **c = (double **)malloc(M * sizeof(double*));
	// 	for (int i = 0; i < M; i++)
	// 	{
	// 		c[i] = (double *)malloc(S * sizeof(double));
	// 	}
	// 	for (int i = 0; i < N; i++)
	// 	{
	// 		b[i] = (double *)malloc(S * sizeof(double));
	// 	}
	// 	for (int i = 0; i < N; i++)
	// 	{
	// 		for (int j = 0; j < M; j++)
	// 		{
	// 			a[j][i] = A(j, i);
	// 		}
	// 		for (int j = 0; j < S; j++)
	// 		{
	// 			b[i][j] = B(i, j);
	// 		}
	// 	}
	// 	cuMatrixMult(a, b, c, M, N, S);
	double *a = A.data();
	double *b = B.data();
	double *c = (double *)malloc(M*S * sizeof(double));
	t1 = clock();
	cuMatrixMult(a, b, c, M, N, S);
	std::cout << (clock() - t1) * 1.0 / CLOCKS_PER_SEC * 1000 << std::endl;
	MatrixXd NewC = Eigen::Map<Eigen::MatrixXd>(c, M, S);
	std::cout << NewC << std::endl;
	// 	MatrixXd NewC(M, S);
	// 	for (int i = 0; i < M; i++)
	// 	{
	// 		for (int j = 0; j < S; j++)
	// 		{
	// 			NewC(i, j) = c[i][j];
	// 		}
	// 	}
	if ((C - NewC).norm() < 10e-10)
	{
		std::cout << "Equality test passed." << std::endl;
	}
	else
	{
		std::cout << "Equality test failed." << std::endl;

	}
	free(c);
}