#include "Timer.h"

#ifdef _MSC_VER

void Timer::start() {
	QueryPerformanceCounter(&starttime);
}

void Timer::stop() {
	QueryPerformanceCounter(&endtime);
}

double Timer::elapsed() {
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);
	return ((double)(endtime.QuadPart - starttime.QuadPart)) / ((double)(freq.QuadPart / 1000.0));
}

#else

void Timer::start() {
	gettimeofday(&starttime, 0);
}

void Timer::stop() {
	gettimeofday(&endtime, 0);
}

double Timer::stop() {
	return (endtime.tv_sec - starttime.tv_sec) * 1000.0 + (endtime.tv_usec - starttime.tv_usec) / 1000.0;
}

#endif
