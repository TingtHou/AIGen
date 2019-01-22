#pragma once
#include <iostream>
#include <time.h>
#include <malloc.h>
#include <Eigen/Dense>
#include "cuMatrix.h"
using Eigen::MatrixXd;
#define M 1
#define N 5000
#define S 10
void cuMatrixInv_timetest();
void cuMatrixINVTest();
void cuMatrixMultTest();
