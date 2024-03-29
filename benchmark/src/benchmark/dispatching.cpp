#include "dispatching.h"
#include "gpu_dispatching.h"
#include "timer.h"
#include "parallel_sort.h"
#include <algorithm>
#include <stdexcept>

namespace Benchmark {
    namespace {

        template<typename KeyType>
        void sort_with_algorithm(Algorithm::Value algorithm, bool keys_only, std::vector<KeyType> &data,
                                 Timer &timer) {
            switch (algorithm) {
                case Algorithm::Value::cpu_parallel:
                    timer.start();
                    Benchmark::parallel_sort(data.begin(), data.end());
                    timer.stop();
                    break;
                case Algorithm::Value::cpu_stl:
                    timer.start();
                    std::sort(data.begin(), data.end());
                    timer.stop();
                    break;
                case Algorithm::Value::thrust:
                case Algorithm::Value::samplesort:
                    // Run once to ensure the GPU kernels are compiled before benchmarking.
                    sort_by_algorithm(algorithm, keys_only, data);
                    timer.start();
                    sort_by_algorithm(algorithm, keys_only, data);
                    timer.stop();
                    break;
            }
        }

        template<typename KeyType>
        void assert_result_is_sorted(const std::vector<KeyType> &data, const std::vector<KeyType> &result) {
            std::vector<KeyType> ground_truth(data);
            Benchmark::parallel_sort(ground_truth.begin(), ground_truth.end());
            if (result != ground_truth)
                throw std::runtime_error("Sorting failed!");
        }

        template<typename KeyType>
        Result execute_for_size(const Settings &settings, std::size_t size) {
            auto distribution = Distributions::create<KeyType>(settings.distribution_type, size,
                                                               settings.distribution_settings, settings.p, settings.g,
                                                               settings.range);

            Timer timer;
            std::vector<KeyType> data(distribution.as_vector());
            sort_with_algorithm<KeyType>(settings.algorithm, settings.keys_only, data, timer);

            std::cout << "Sorting time was " << std::fixed << std::setprecision(4) << timer.elapsed() << "ms for " <<
                      size << " elements." << std::endl;

            assert_result_is_sorted(distribution.as_vector(), data);

            return Result(timer.elapsed(), distribution.size());
        }

        template<typename KeyType>
        std::vector<Result> execute_with_settings(const Settings &settings) {
            std::vector<Result> results;
            for (const std::size_t &size: settings.sizes)
                results.push_back(execute_for_size<KeyType>(settings, size));
            return results;
        }
    }

    std::vector<Result> execute_with_settings(const Settings &settings) {
        switch (settings.key_type) {
            case KeyType::Value::uint64_t:
                return execute_with_settings<std::uint64_t>(settings);
            case KeyType::Value::uint32_t:
                return execute_with_settings<std::uint32_t>(settings);
            case KeyType::Value::uint16_t:
                return execute_with_settings<std::uint16_t>(settings);
            default:
                throw std::runtime_error("Unhandled key type!");
        }
    }
}
