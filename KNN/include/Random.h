#pragma once
#include <boost/random.hpp>
#include <boost/random/random_device.hpp>
#include <time.h>

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

