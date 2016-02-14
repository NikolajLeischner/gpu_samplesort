#pragma once

#ifdef _MSC_VER
#include <windows.h>

class Timer
{
	LARGE_INTEGER starttime;
	LARGE_INTEGER endtime;
public:
	void start();
	void stop();
	double elapsed();
};

#else
#include <sys/time.h>

class Timer
{
	struct timeval starttime;
	struct timeval endtime;
public:
	void start();
	void stop();
	double elapsed();
};

#endif
