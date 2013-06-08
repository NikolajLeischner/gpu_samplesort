/**
* GPU Sample Sort
* -----------------------
* Copyright (c) 2009-2010 Nikolaj Leischner and Vitaly Osipov
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
**/

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
