#pragma once

#ifdef _MSC_VER

#include <Windows.h>

class Timer {
    LARGE_INTEGER start_time;
    LARGE_INTEGER end_time;
public:
    Timer() : start_time({0}), end_time({0}) {}

    void start();

    void stop();

    double elapsed() const;
};

#else
#include <sys/time.h>

class Timer
{
    struct timeval start_time;
    struct timeval end_time;
public:
    void start();
    void stop();
    double elapsed();
};

#endif
