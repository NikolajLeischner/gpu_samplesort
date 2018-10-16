#include "timer.h"

#ifdef _MSC_VER

void Timer::start() {
    QueryPerformanceCounter(&start_time);
}

void Timer::stop() {
    QueryPerformanceCounter(&end_time);
}

double Timer::elapsed() {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    return ((double) (end_time.QuadPart - start_time.QuadPart)) / ((double) (freq.QuadPart / 1000.0));
}

#else

void Timer::start() {
    gettimeofday(&start_time, 0);
}

void Timer::stop() {
    gettimeofday(&end_time, 0);
}

double Timer::stop() {
    return (end_time.tv_sec - start_time.tv_sec) * 1000.0 + (end_time.tv_usec - start_time.tv_usec) / 1000.0;
}

#endif
