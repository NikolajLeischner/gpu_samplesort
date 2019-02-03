The Python script `run_benchmark` executes the sort benchmark according to a JSON configuration file.

The configuration consists of a list of algorithms, and a list of experiments.

Each experiment needs to specify:
* the parameters to call the sort benchmark with,
* the minimum input size to measure,
* the maximum input size,
* how many input sizes to benchmark in between the minimum & maximum sizes.
