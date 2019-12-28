//////////////////////////////////////////////////////////////////////////
////////  The class is used to generate a float random number
///////	  it is based on the boost
///////   you can use it like this:
///////           Random x;
///////	          x.generation(), then it returns a float random numder
///////
////////////////////////////////////////////////////////////////////////////


#include "../include/Random.h"
using namespace boost;

Random::Random()
{

	engine = new boost::mt19937(time(0));
}

Random::Random(float seed)
{
	engine = new boost::mt19937(seed);
}

Random::~Random()
{
	delete engine;
}

float Random::Uniform()
{
	
	boost::uniform_01<> *u01 =new boost::uniform_01<>();
	boost::variate_generator<boost::mt19937&, boost::uniform_01<> > die = boost::variate_generator<boost::mt19937&, boost::uniform_01<> >(*engine, *u01);
	float rd = die();
	delete u01;
	return rd;
}

float Random::Uniform(float min, float max)
{
	boost::uniform_real<> *u01 = new boost::uniform_real<>(min,max);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<> > die =  boost::variate_generator<boost::mt19937&, boost::uniform_real<> >(*engine, *u01);
	float rd = die();
	delete u01;
	return rd;
}

float Random::Normal()
{
	boost::normal_distribution<> *u01 = new boost::normal_distribution<>();
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die = boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >(*engine, *u01);
	float rd = die();
	delete u01;
	return rd;
}

float Random::Normal(float mean, float sd)
{
	boost::normal_distribution<> *u01 = new boost::normal_distribution<>(mean,sd);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die= boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >(*engine, *u01);
	float rd = die();
	delete u01;
	return rd;
}

mvnorm::mvnorm(int n, Eigen::VectorXf& mu, Eigen::MatrixXf& Sigma)
{
	this->mu = mu;
	this->Sigma = Sigma;
	this->n = n;
	nind = mu.size();
	Y.resize(nind, n);
	generate();
}

Eigen::MatrixXf mvnorm::rmvnorm()
{
	Y.setZero();
	generate();
	return Y;
}



void mvnorm::generate()
{
	auto svd = Sigma.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
	const auto& singularValues = svd.singularValues();
	for (unsigned int i = 0; i < singularValues.size(); ++i)
	{
		if (singularValues(i) < -10E-6)
			throw("The variance covariance matrix is not positive definite.");
	}
	Eigen::MatrixXf Lower = svd.matrixU() * singularValues.cwiseSqrt().asDiagonal();
	for (int i = 0; i < n; i++)
	{
		Y.col(i) = simulation(Lower);
	}
}

Eigen::VectorXf mvnorm::simulation(Eigen::MatrixXf& LowerMatrix)
{
	Eigen::VectorXf Z(nind);
	Eigen::VectorXf tmpY(nind);
	for (size_t i = 0; i < nind; i++)
	{
		Z(i) = rd.Normal();
	}
	tmpY = LowerMatrix * Z + mu;
	return tmpY;
}

