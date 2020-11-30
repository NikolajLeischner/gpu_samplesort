/**
* GPU Sample Sort
* -----------------------
* Copyright (c) 2009-2019 Nikolaj Leischner and Vitaly Osipov
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

#pragma once

namespace SampleSort {

    template<typename T>
    struct TemporaryDeviceMemory {
        T *data;
        size_t size;

        explicit TemporaryDeviceMemory(size_t size) : size(size) {
            if (size > 0)
                cudaMalloc((void **) &data, size * sizeof(T));
            else
                data = nullptr;
        }

        ~TemporaryDeviceMemory() {
            if (data != nullptr)
                cudaFree(data);
        }

        void copy_to_device(T *source) {
            cudaMemcpy(data, source, size * sizeof(T), cudaMemcpyHostToDevice);
        }

        void copy_to_host(T *target) {
            cudaMemcpy(target, data, size * sizeof(T), cudaMemcpyDeviceToHost);
        }

    };

}
