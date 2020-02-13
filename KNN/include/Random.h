#pragma once
#define EIGEN_USE_MKL_ALL
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <time.h>
#include<Eigen/Dense>

class Random
{
public:
	Random();
	Random(float seed);
	boost::mt19937 *engine;
	~Random();
public:
	float Uniform();
	float Uniform(float min, float max);
	float Normal();
	float Normal(float mean, float sd);
};

class mvnorm
{
public:
	mvnorm(int n, Eigen::VectorXf& mu, Eigen::MatrixXf& Sigma);
	Eigen::MatrixXf rmvnorm();
private:
	Random rd;
	int n = 0;
	int nind = 0;
	Eigen::VectorXf mu;
	Eigen::MatrixXf Sigma;
	Eigen::MatrixXf Y;
	void generate();
	Eigen::VectorXf simulation(Eigen::MatrixXf& LowerMatrix);
};

