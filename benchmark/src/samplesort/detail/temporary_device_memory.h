#pragma once

namespace SampleSort {

    template<typename T>
    struct TemporaryDeviceMemory {
        T *data;
        size_t size;

        TemporaryDeviceMemory(size_t size) : size(size) {
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
