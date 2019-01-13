#include <algorithm>
#include <iostream>
#include <settings.h>
#include <distributions.h>
#include <keytype.h>
#include <tclap/CmdLine.h>

namespace Benchmark {
    Settings Settings::parse_from_cmd(int argc, char *argv[], bool disable_exception_handling) {
        std::string version("0.1");
        TCLAP::CmdLine cmd("", ' ', version);

        TCLAP::ValueArg<std::string> algorithm("a", "algorithm", "", true, "samplesort", "");
        TCLAP::ValueArg<std::string> distribution("d", "distribution", "", false, "uniform", "");
        TCLAP::ValueArg<std::uint32_t> parameter_p("p", "parameter-p", "", false, 128, "");
        TCLAP::ValueArg<std::uint32_t> parameter_g("g", "parameter-g", "", false, 8, "");
        TCLAP::ValueArg<std::uint32_t> parameter_range("r", "parameter-range", "", false, 30, "");
        TCLAP::ValueArg<std::string> key_type("k", "key-type", "", false, "uint32", "");
        TCLAP::ValueArg<std::size_t> samples_per_key("s", "samples", "", false, 1, "");
        TCLAP::ValueArg<std::size_t> bits_per_key("b", "bits", "", false, 32, "");
        TCLAP::ValueArg<bool> keys_have_values("v", "keys-have-values", "", false, false, "");
        TCLAP::ValueArg<std::string> output("o", "output-file", "", false, "result.csv", "");
        TCLAP::MultiArg<std::uint64_t> sizes("i", "sizes", "", true, "");

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
                        KeyType::parse(key_type.getValue()), keys_have_values.getValue(), output.getValue(),
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
