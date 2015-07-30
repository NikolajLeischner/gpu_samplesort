#pragma once

#ifdef _MSC_VER
#include <windows.h>

class Timer
{
	LARGE_INTEGER starttime;
public:
	void start();
	double stop(); 
};

#else
#include <sys/time.h>

class Timer
{
	struct timeval starttime;
public:
	void start();
	double stop(); 
};

#endif
