#include "Timer.h"

#ifdef _MSC_VER

void Timer::start()
{
	QueryPerformanceCounter(&starttime);
}

double Timer::stop()
{
	LARGE_INTEGER endtime,freq;
	QueryPerformanceCounter(&endtime);
	QueryPerformanceFrequency(&freq);

	return ((double)(endtime.QuadPart-starttime.QuadPart))/((double)(freq.QuadPart/1000.0));
}


#else

void Timer::start()
{
	gettimeofday(&starttime,0);
}

double Timer::stop()
{
	struct timeval endtime;
	gettimeofday(&endtime,0);

	return (endtime.tv_sec - starttime.tv_sec)*1000.0 + (endtime.tv_usec - starttime.tv_usec)/1000.0;
}

#endif
