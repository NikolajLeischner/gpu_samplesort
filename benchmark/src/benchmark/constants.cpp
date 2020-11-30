#include <algorithm>
#include <settings.h>
#include <distributions.h>
#include <keytype.h>

namespace Benchmark {
    Settings Settings::parse_from_cmd(int argc, const char *argv[], bool disable_exception_handling) {
        std::string version("0.1");
        TCLAP::CmdLine cmd("", ' ', version);

        TCLAP::ValueArg<std::string> algorithm("a", "algorithm",
                "Algorithm to benchmark.", true, "samplesort", "cpu_stl, cpu_parallel, thrust, samplesort");
        TCLAP::ValueArg<std::string> distribution("d", "distribution",
                "The type of random distribution to use.", false, "uniform",
                "zero, sorted, uniform, gaussian, bucket, staggered, g-groups, sorted-descending, random-duplicates, deterministic-duplicates");
        TCLAP::ValueArg<std::uint32_t> parameter_p("p", "parameter-p",
                "Parameter P for the distribution types bucket, g-groups, staggered, random-duplicates, deterministic-duplicates.", false, 128, "int");
        TCLAP::ValueArg<std::uint32_t> parameter_g("g", "parameter-g",
                "Parameter G for the distribution type g-groups.", false, 8, "int");
        TCLAP::ValueArg<std::uint32_t> parameter_range("r", "parameter-range",
                "Parameter Range for the distribution type random-duplicates.", false, 30, "int");
        TCLAP::ValueArg<std::string> key_type("k", "key-type", "Key type.", false, "uint32", "uint16, uint32, uint64");
        TCLAP::ValueArg<std::size_t> samples_per_key("s", "samples",
                "Random samples taken per key. The samples are combined with logical and; taking more samples reduces the entropy of the data.", false, 1, "int");
        TCLAP::ValueArg<std::size_t> bits_per_key("b", "bits",
                "Number of bits per key that are not zeroed out. Choosing a lower bit count reduces the entropy of the data.", false, 32, "int");
        TCLAP::SwitchArg keys_have_values("v", "keys-have-values",
                "If set, a key-value sort is done, where the values are 64bit integers.", false);
        TCLAP::ValueArg<std::string> output("o", "output-file",
                "Destination file for the CSV output of the benchmark run.", false, "result.csv", "file name");
        TCLAP::MultiArg<std::uint64_t> sizes("i", "sizes", "Input sizes to benchmark.", true, "int");

        cmd.add(algorithm);
        cmd.add(distribution);
        cmd.add(parameter_p);
        cmd.add(parameter_g);
        cmd.add(parameter_range);
        cmd.add(key_type);
        cmd.add(samples_per_key);
        cmd.add(bits_per_key);
        cmd.add(keys_have_values);
        cmd.add(output);
        cmd.add(sizes);

        cmd.setExceptionHandling(!disable_exception_handling);
        cmd.parse(argc, argv);

        return Settings(Algorithm::parse(algorithm.getValue()),
                        Distributions::Type::parse(distribution.getValue()),
                        Distributions::Settings(bits_per_key.getValue(), samples_per_key.getValue()),
                        parameter_p.getValue(), parameter_g.getValue(), parameter_range.getValue(),
                        KeyType::parse(key_type.getValue()), !keys_have_values.getValue(), output.getValue(),
                        sizes.getValue());
    }

    namespace Algorithm {

        Algorithm::Value parse(std::string value) {
            std::transform(value.begin(), value.end(), value.begin(), tolower);

            auto result = types.find(value);
            if (result == types.end())
                throw std::exception();
            else
                return result->second;
        }

        std::string as_string(Value value) {
            for (const auto &type : types)
                if (type.second == value)
                    return type.first;
            return "";
        }

    }
}
