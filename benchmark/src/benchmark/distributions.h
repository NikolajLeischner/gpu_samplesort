#pragma once

#include <cstddef>
#include <vector>

// Fill an array with random values. 
// The distributions described in http://www.umiacs.umd.edu/research/EXPAR/papers/3669/node5.html#SECTION00041000000000000000
namespace Distributions {
    template<typename T>
    class Distribution {
    public:
        const T *const begin() const;

        const T *const end() const;

        std::uint64_t size() const;

        std::uint64_t memory_size() const;

        explicit Distribution(std::vector<T> content);

        void print(std::uint64_t count) const;

        std::vector<T> as_vector() const;

    private:
        const std::vector<T> content;
    };

    struct Settings {
        Settings(std::uint64_t bits, std::uint64_t samples) : bits(bits), samples(samples) {}

        const std::uint64_t bits;
        const std::uint64_t samples;
    };

    struct Type {
        enum class Value {
            zero,
            sorted,
            sorted_descending,
            uniform,
            gaussian,
            bucket,
            staggered,
            g_groups,
            random_duplicates,
            deterministic_duplicates
        };

        static Value parse(std::string value);
    };


    template<typename T>
    Distribution<T>
    create(const Type::Value type, std::uint64_t size, const Settings &settings, std::uint32_t p, std::uint32_t g,
           std::uint32_t range);

    template<typename T>
    Distribution<T> zero(std::uint64_t size);

    template<typename T>
    Distribution<T> sorted(std::uint64_t size, const Settings &settings);

    template<typename T>
    Distribution<T> sorted_descending(std::uint64_t size, const Settings &settings);

    template<typename T>
    Distribution<T> uniform(std::uint64_t size, const Settings &settings);

    template<typename T>
    Distribution<T> gaussian(std::uint64_t size, const Settings &settings);

    template<typename T>
    Distribution<T> bucket(std::uint64_t size, const Settings &settings, std::uint32_t p);

    template<typename T>
    Distribution<T> staggered(std::uint64_t size, const Settings &settings, std::uint32_t p);

    template<typename T>
    Distribution<T> g_groups(std::uint64_t size, const Settings &settings, std::uint32_t p, std::uint32_t g);

    template<typename T>
    Distribution<T>
    random_duplicates(std::uint64_t size, const Settings &settings, std::uint32_t p, std::uint64_t range);

    template<typename T>
    Distribution<T> deterministic_duplicates(std::uint64_t size, const Settings &settings, std::uint32_t p);
}
