#include "pch.h"
#include "ToolKit.h"

void ToolKit::ArraytoVector(double ** a, int n, int m, vector<vector<double>>& v, bool Transpose)
{
	v.clear();
	if (Transpose)
	{
		v.resize(m);
		for (int i = 0; i < m; i++)
		{
			v[i].resize(n);
			for (int j = 0; j < n; j++)
			{
				v[i][j] = a[j][i];
			}
		}
	}
	else
	{
		v.resize(n);
		for (int i = 0; i < n; i++)
		{
			v[i].resize(m);
			for (int j = 0; j < m; j++)
			{
				v[i][j] = a[i][j];
			}
		}

	}
}


void ToolKit::Array2toArrat1(double **a, int n, int m, double *b, bool Colfirst)
{
	int k = 0;
	if (Colfirst)
	{
		for (int i=0;i<m;i++)
		{
			for (int j=0;j<n;j++)
			{
				b[k++] = a[j][i];
			}
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				b[k++] = a[i][j];
			}
		}
	}
}

void ToolKit::Vector2toArray1(vector<vector<double>>& v, double * b, bool Colfirst)
{
	int n = v.size();
	int m = v[0].size();
	int k = 0;
	if (Colfirst)
	{
		for (int i = 0; i < m; i++)
		{
			for (int j = 0; j < n; j++)
			{
				b[k++] = v[j][i];
			}
		}
	}
	else
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				b[k++] = v[i][j];
			}
		}
	}
}

void ToolKit::Stringsplit(string & org, vector<string>& splited, string delim)
{
	splited.clear();
	size_t current;
	size_t next = -1;
	do
	{
		current = next + 1;
		next = org.find_first_of(delim, current);
		splited.push_back(org.substr(current, next - current));
	} while (next != string::npos);
}

void ToolKit::dec2bin(int num, int *bin)
{
	for (int i = 8; i >= 0; i--) {
		if (num & (1 << i))
			bin[7 - i] = 1;
		else
			bin[7 - i] = 0;
	}
}

bool ToolKit::Inv_Cholesky(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix)
{
 	Eigen::LDLT<Eigen::MatrixXd> LDLT;
	Eigen::MatrixXd IdentityMatrix(Ori_Matrix.rows(), Ori_Matrix.cols());
	IdentityMatrix.setIdentity();
	LDLT.compute(Ori_Matrix);
	if (LDLT.info() == Eigen::NumericalIssue)
	{
		return false;
	}
 	Inv_Matrix = LDLT.solve(IdentityMatrix);
//	std::cout << IdentityMatrix.isApprox(Ori_Matrix*Inv_Matrix) << std::endl;  // false
	bool a_solution_exists = (Ori_Matrix*Inv_Matrix).isApprox(IdentityMatrix,1e-10);
//	Inv_Matrix = Ori_Matrix.ldlt().solve(IdentityMatrix);
	return a_solution_exists;
}

bool ToolKit::comput_inverse_logdet_LDLT_mkl(Eigen::MatrixXd &Vi, double &logdet)
{
	long i = 0, j = 0, n = Vi.cols();
	double* Vi_mkl = new double[n * n];
	//float* Vi_mkl=new float[n*n];

#pragma omp parallel for private(j)
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			Vi_mkl[i * n + j] = Vi(i, j);
		}
	}

	// MKL's Cholesky decomposition
	int info = 0, int_n = (int)n;
	char uplo = 'L';
	dpotrf(&uplo, &int_n, Vi_mkl, &int_n, &info);
	//spotrf( &uplo, &n, Vi_mkl, &n, &info );
	if (info < 0) throw ("Error: Cholesky decomposition failed. Invalid values found in the matrix.\n");
	else if (info > 0) return false;
	else {
		logdet = 0.0;
		for (i = 0; i < n; i++) {
			double d_buf = Vi_mkl[i * n + i];
			logdet += log(d_buf * d_buf);
		}

		// Calcualte V inverse
		dpotri(&uplo, &int_n, Vi_mkl, &int_n, &info);
		//spotri( &uplo, &n, Vi_mkl, &n, &info );
		if (info < 0) throw ("Error: invalid values found in the varaince-covaraince (V) matrix.\n");
		else if (info > 0) return false;
		else {
#pragma omp parallel for private(j)
			for (j = 0; j < n; j++) {
				for (i = 0; i <= j; i++) Vi(i, j) = Vi(j, i) = Vi_mkl[i * n + j];
			}
		}
	}

	// free memory
	delete[] Vi_mkl;

	return true;

}

bool ToolKit::Inv_LU(Eigen::MatrixXd & Ori_Matrix, Eigen::MatrixXd & Inv_Matrix)
{
	Eigen::PartialPivLU<Eigen::MatrixXd> LU(Ori_Matrix);
	Eigen::MatrixXd IdentityMatrix(Ori_Matrix.rows(), Ori_Matrix.cols());
	IdentityMatrix.setIdentity();
	Inv_Matrix = LU.inverse();
	bool a_solution_exists = (Ori_Matrix*Inv_Matrix).isApprox(IdentityMatrix, 1e-10);
	return a_solution_exists;
}

bool ToolKit::comput_inverse_logdet_LU_mkl(Eigen::MatrixXd &Vi, double &logdet)
{
	 long i = 0, j = 0, n = Vi.cols();
	double* Vi_mkl = new double[n * n];

#pragma omp parallel for private(j)
	for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			Vi_mkl[i * n + j] = Vi(i, j);
		}
	}

	int N = (int)n;
	int *IPIV = new int[n + 1];
	int LWORK = N * N;
	double *WORK = new double[n * n];
	int INFO;
	dgetrf(&N, &N, Vi_mkl, &N, IPIV, &INFO);
	if (INFO < 0) throw ("Error: LU decomposition failed. Invalid values found in the matrix.\n");
	else if (INFO > 0) {
		delete[] Vi_mkl;
		return false;
	}
	else {
		logdet = 0.0;
		for (i = 0; i < n; i++) {
			double d_buf = Vi_mkl[i * n + i];
			logdet += log(fabs(d_buf));
		}

		// Calcualte V inverse
		dgetri(&N, Vi_mkl, &N, IPIV, WORK, &LWORK, &INFO);
		if (INFO < 0) throw ("Error: invalid values found in the varaince-covaraince (V) matrix.\n");
		else if (INFO > 0) return false;
		else {
#pragma omp parallel for private(j)
			for (j = 0; j < n; j++) {
				for (i = 0; i <= j; i++) Vi(i, j) = Vi(j, i) = Vi_mkl[i * n + j];
			}
		}
	}

	// free memory
	delete[] Vi_mkl;
	delete[] IPIV;
	delete[] WORK;

	return true;
}


bool ToolKit::Inv_SVD(Eigen::MatrixXd & Ori_Matrix, bool allowPseudoInverse)
{

	auto svd = Ori_Matrix.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
	const auto &singularValues = svd.singularValues();
	Eigen::MatrixXd singularValuesInv(Ori_Matrix.cols(), Ori_Matrix.rows());
	Eigen::MatrixXd Inv_Matrix(Ori_Matrix.cols(), Ori_Matrix.rows());
	singularValuesInv.setZero();
	double  pinvtoler = 1.e-20; // choose your tolerance wisely
	bool singlar = false;

	for (unsigned int i = 0; i < singularValues.size(); ++i)
	{
		if (abs(singularValues(i)) > pinvtoler)
			singularValuesInv(i, i) = (double)1.0 / singularValues(i);
		else
		{
			if (!allowPseudoInverse)
			{
				return false;
			}
			singularValuesInv(i, i) = (double)0.0;
//			singlar = true;
		}
	}
	Inv_Matrix = svd.matrixV() * singularValuesInv * svd.matrixU().transpose();
	Ori_Matrix = Inv_Matrix;
	return true;
}

bool ToolKit::Inv_QR(Eigen::MatrixXd & Ori_Matrix, bool allowPseudoInverse)
{
	Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> QR;
	Eigen::MatrixXd IdentityMatrix(Ori_Matrix.rows(), Ori_Matrix.cols());
	IdentityMatrix.setIdentity();
	QR.compute(Ori_Matrix);
	Eigen::MatrixXd Inv_Matrix(Ori_Matrix.rows(), Ori_Matrix.cols());
	if (!QR.isInvertible()&&!allowPseudoInverse)
	{
		return false;
	}
	if (QR.isInvertible())
	{
		Inv_Matrix = QR.solve(IdentityMatrix);
	}
	else
	{
		Inv_Matrix = QR.pseudoInverse();
	}
	Ori_Matrix = Inv_Matrix;
	return true;
}

