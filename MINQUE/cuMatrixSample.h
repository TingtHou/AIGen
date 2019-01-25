#pragma once
#include <iostream>
#include <time.h>
#include <malloc.h>
#include <Eigen/Dense>
#include "cuMatrix.h"
using Eigen::MatrixXd;
#define M 1000
#define N 5000
#define S 1000
void cuMatrixInv_timetest();
void cuMatrixINVTest();
void cuMatrixMultTest();
