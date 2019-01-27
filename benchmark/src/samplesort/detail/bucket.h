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
    struct Bucket {
        // Elements per thread to use for the bucket.
        int keys_per_thread = 1;
        int start;
        int size;
        // Is the bucket located in the input array or in the buffer?
        bool flipped = false;
        // Degenerated means that the sample used to sort the bucket only
        // contained equal keys.
        bool degenerated = false;
        bool constant = false;

        Bucket(int start, int size, bool flipped = false) : keys_per_thread(1), start(start), size(size),
                                                            flipped(flipped), degenerated(false), constant(false) {}

        Bucket() : Bucket(0, 0) {}
    };

    bool operator<(const Bucket &lhs, const Bucket &rhs) {
        return rhs.size > lhs.size;
    }
}
