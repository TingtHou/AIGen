#pragma once
#include <iostream>
#include <time.h>
#include <malloc.h>
#include <Eigen/Dense>
#include "cuMatrix.h"
using Eigen::MatrixXd;
void cuMatrixInv_timetest(int N);
void cuMatrixINVTest(int N);
void cuMatrixMultTest(int N, int M, int S);
