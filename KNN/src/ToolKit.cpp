#include "../include/ToolKit.h"

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

bool ToolKit::Inv_Cholesky(Eigen::MatrixXd & Ori_Matrix)
{
	Eigen::MatrixXd Inv_Matrix(Ori_Matrix.rows(), Ori_Matrix.cols());
 	Eigen::LDLT<Eigen::MatrixXd> LDLT;
	Eigen::MatrixXd IdentityMatrix(Ori_Matrix.rows(), Ori_Matrix.cols());
	IdentityMatrix.setIdentity();
	LDLT.compute(Ori_Matrix);
	if (LDLT.info() == Eigen::NumericalIssue)
	{
		return false;
	}
 	Inv_Matrix = LDLT.solve(IdentityMatrix);
	Ori_Matrix = Inv_Matrix;
//	std::cout << IdentityMatrix.isApprox(Ori_Matrix*Inv_Matrix) << std::endl;  // false
//	Inv_Matrix = Ori_Matrix.ldlt().solve(IdentityMatrix);
	return true;
}

bool ToolKit::comput_inverse_logdet_LDLT_mkl(Eigen::MatrixXd &Vi)
{

	int n = Vi.cols();
	double* Vi_mkl = Vi.data();
	// MKL's Cholesky decomposition
	int info = 0, int_n = (int)n;
	char uplo = 'L';
	info=LAPACKE_dpotrf(LAPACK_COL_MAJOR,uplo, int_n, Vi_mkl, int_n);
	if (info != 0) return false;
	else {
		// Calcualte V inverse
		info=LAPACKE_dpotri(LAPACK_COL_MAJOR, uplo, int_n, Vi_mkl, int_n);
		if (info != 0)
		{
			return false;
		}
		else 
		{
			#pragma omp parallel for
			for (int i = 0; i < n; i++) //row
			{
				for (int j = i; j <n; j++) //col
					Vi_mkl[j * n + i]=Vi_mkl[i * n + j];
			}
		}
	}
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

bool ToolKit::comput_inverse_logdet_LU_mkl(Eigen::MatrixXd &Vi)
{
	int n = Vi.cols();
	double* Vi_mkl = Vi.data();

	int N = (int)n;
	int *IPIV = new int[n + 1];
	int LWORK = N * N;
	int INFO=0;
	INFO =LAPACKE_dgetrf(LAPACK_COL_MAJOR, N, N, Vi_mkl, N, IPIV);
	if (INFO!=0) {
		delete[] IPIV;
		return false;
	}
	else {
		// Calcualte V inverse
		INFO=LAPACKE_dgetri(LAPACK_COL_MAJOR , N, Vi_mkl, N, IPIV);
		if (INFO != 0)
			return false;
	}
	// free memory
	delete[] IPIV;
	return true;
}

bool ToolKit::comput_inverse_logdet_QR_mkl(Eigen::MatrixXd& Vi)
{
	int n = Vi.cols();
	double* Vi_mkl = Vi.data();
	Eigen::MatrixXd Rinv(n, n);
	Eigen::MatrixXd Qt(n, n);
	Rinv.setIdentity();
	Qt.setIdentity();
	double* pr_Rinv = Rinv.data();
	double* pr_Qt = Qt.data();
	double* tau = new double[n + 1];
	int INFO = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, n, n, Vi_mkl, n, tau);
	if (INFO != 0)
	{
		delete[] tau;
		throw ("Error: QR decomposition failed. Invalid values found in the matrix.\n");
	}
	cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, n, 1, Vi_mkl, n, pr_Rinv, n);
	LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', n, n, n, Vi_mkl, n, tau, pr_Qt, n);
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n,1, pr_Rinv, n, pr_Qt, n, 0, Vi_mkl,n);
	delete[] tau;
	return true;
}

bool ToolKit::comput_inverse_logdet_SVD_mkl(Eigen::MatrixXd& Vi)
{
	int n = Vi.cols();
	double* Vi_mkl = Vi.data();
	MKL_INT  lwork;
	MKL_INT info;
	double wkopt;
	double* work;
	char jobu = 'S';
	char jobvt = 'S';
	double* s = (double*)malloc(n * sizeof(double));
	double* u = (double*)malloc(n * n * sizeof(double));
	double* vt = (double*)malloc(n * n * sizeof(double));
	lwork = -1;
	dgesvd(&jobu, &jobvt, &n, &n, Vi_mkl, &n, s, u, &n, vt, &n, &wkopt, &lwork, &info);
	lwork = (MKL_INT)wkopt;
	work = (double*)malloc(lwork * sizeof(double));
	dgesvd(&jobu, &jobvt, &n, &n, Vi_mkl, &n, s, u, &n, vt, &n, work, &lwork, &info);
	if (info > 0)
	{
		free(s);
		free(u);
		free(vt);
		throw ("The algorithm computing SVD failed to converge.\n");
	}
	if (info < 0)
	{
		free(s);
		free(u);
		free(vt);
		throw ("Error: SVD decomposition failed. Invalid values found in the matrix.\n");
	}
	//u=(s^-1)*U
	MKL_INT incx = 1;
	#pragma omp parallel for
	for (int i = 0; i < n; i++)
	{
		double ss;
		if (s[i] > 1.0e-9)
			ss = 1.0 / s[i];
		else
			ss = s[i];
		dscal(&n, &ss, &u[i * n], &incx);
	}
	//inv(A)=(Vt)^T *u^T
	double alpha = 1.0, beta = 0.0;
	MKL_INT ld_inva = n;
	dgemm("T", "T", &n, &n, &n, &alpha, vt, &n, u, &n, &beta, Vi_mkl, &ld_inva);
	free(s);
	free(u);
	free(vt);
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

