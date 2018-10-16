#include "gpu_dispatching.h"
#include "distributions.h"

namespace Benchmark {
    template<typename KeyType>
    double benchmark_algorithm(Benchmark::Algorithm::Value algorithm, bool keys_have_values,
                               const Distribution <KeyType> &distribution) {

        KeyType *device_keys(0);
        cudaMalloc((void **) &device_keys, distribution.memory_size());
        //cudaMemcpy()


        if (keys_have_values) {

            auto values = Distributions:: < std::uint64_t > create();

            std::uint64_t *device_values(0);
            cudaMalloc((void **) &device_values, values.memory_size());
            //cudaMemcpy()

        }

        cudaFree(device_keys);
        cudaFree(device_values);

        return 0.0;
    }

    template double benchmark_algorithm(Benchmark::Algorithm::Value algorithm, bool keys_have_values,
                                        const Distribution <std::uint16_t> &distribution);

    template double benchmark_algorithm(Benchmark::Algorithm::Value algorithm, bool keys_have_values,
                                        const Distribution <std::uint32_t> &distribution);

    template double benchmark_algorithm(Benchmark::Algorithm::Value algorithm, bool keys_have_values,
                                        const Distribution <std::uint64_t> &distribution);
}