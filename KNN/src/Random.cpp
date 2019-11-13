//////////////////////////////////////////////////////////////////////////
////////  The class is used to generate a double random number
///////	  it is based on the boost
///////   you can use it like this:
///////           Random x;
///////	          x.generation(), then it returns a double random numder
///////
////////////////////////////////////////////////////////////////////////////


#include "../include/Random.h"
using namespace boost;

Random::Random()
{

	engine = new boost::mt19937(time(0));
}

Random::Random(double seed)
{
	engine = new boost::mt19937(seed);
}

Random::~Random()
{
	delete engine;
}

double Random::Uniform()
{
	
	boost::uniform_01<> *u01 =new boost::uniform_01<>();
	boost::variate_generator<boost::mt19937&, boost::uniform_01<> > die = boost::variate_generator<boost::mt19937&, boost::uniform_01<> >(*engine, *u01);
	double rd = die();
	delete u01;
	return rd;
}

double Random::Uniform(double min, double max)
{
	boost::uniform_real<> *u01 = new boost::uniform_real<>(min,max);
	boost::variate_generator<boost::mt19937&, boost::uniform_real<> > die =  boost::variate_generator<boost::mt19937&, boost::uniform_real<> >(*engine, *u01);
	double rd = die();
	delete u01;
	return rd;
}

double Random::Normal()
{
	boost::normal_distribution<> *u01 = new boost::normal_distribution<>();
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die = boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >(*engine, *u01);
	double rd = die();
	delete u01;
	return rd;
}

double Random::Normal(double mean, double sd)
{
	boost::normal_distribution<> *u01 = new boost::normal_distribution<>(mean,sd);
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > die= boost::variate_generator<boost::mt19937&, boost::normal_distribution<> >(*engine, *u01);
	double rd = die();
	delete u01;
	return rd;
}

