
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <time.h>
#pragma once
class Random
{
public:
	Random();
	Random(double seed);
	boost::mt19937 *engine;
	~Random();
public:
	double Uniform();
	double Uniform(double min, double max);
	double Normal();
	double Normal(double mean, double sd);
};

