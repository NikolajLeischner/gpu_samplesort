#include "gpu_dispatching.h"
#include "distributions.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "../samplesort/samplesort.h"


namespace Benchmark {
    template<typename KeyType>
    void sort_by_algorithm(Algorithm::Value algorithm, bool keys_only, std::vector<KeyType> &data) {
        KeyType *device_keys(0);
        std::uint64_t *device_values(0);
        size_t size = sizeof(KeyType) * data.size();
        cudaMalloc((void **) &device_keys, size);
        cudaMemcpy(device_keys, data.data(), size, cudaMemcpyHostToDevice);

        if (keys_only) {
            if (algorithm == Algorithm::Value::thrust) {
                thrust::device_ptr <KeyType> keys_ptr(device_keys);
                thrust::sort(keys_ptr, keys_ptr + data.size());
            } else if (algorithm == Algorithm::Value::samplesort) {
                SampleSort::sort(device_keys, device_keys + data.size());
            }
        } else {
            auto values = Distributions::uniform<std::uint64_t>(data.size(), Distributions::Settings(64, 1));

            cudaMalloc((void **) &device_values, values.memory_size());
            cudaMemcpy(device_values, values.begin(), values.memory_size(), cudaMemcpyHostToDevice);

            if (algorithm == Algorithm::Value::thrust) {
                thrust::device_ptr <KeyType> keys_ptr(device_keys);
                thrust::device_ptr <std::uint64_t> values_ptr(device_values);
                thrust::sort_by_key(keys_ptr, keys_ptr + data.size(), values_ptr);
            } else if (algorithm == Algorithm::Value::samplesort) {
                SampleSort::sort_by_key(device_keys, device_keys + data.size(), device_values);
            }

            cudaMemcpy(values.as_vector().data(), device_values, values.memory_size(), cudaMemcpyDeviceToHost);
        }

        cudaMemcpy(data.data(), device_keys, size, cudaMemcpyDeviceToHost);
        cudaFree(device_keys);
        cudaFree(device_values);
    }


    template void sort_by_algorithm(Algorithm::Value algorithm, bool keys_only,
                                        std::vector<std::uint16_t> &data);

    template void sort_by_algorithm(Algorithm::Value algorithm, bool keys_only,
                                        std::vector<std::uint32_t> &data);

    template void sort_by_algorithm(Algorithm::Value algorithm, bool keys_only,
                                        std::vector<std::uint64_t> &data);
}