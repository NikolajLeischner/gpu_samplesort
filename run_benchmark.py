import argparse
import json
import subprocess
import itertools


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--executable", help="path to the benchmark executable")
    parser.add_argument("-c", "--configuration", help="path to the benchmark configuration")
    args = parser.parse_args()

    configuration = json.load(open(args.configuration))

    for experiment in configuration["experiments"]:
        run_experiment(args.executable, configuration["algorithms"], experiment)


def run_experiment(executable, algorithms, experiment):
    sizes = size_arguments_experiment(experiment)
    settings = experiment["settings"].split()
    for algorithm in algorithms:
        args = [executable, "-a", algorithm] + sizes + settings
        print("Running benchmark '" + experiment["title"] + "': " + str(args))
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def size_arguments_experiment(experiment):
    steps = experiment["steps"] - 1
    step = int((experiment["maximum_size"] - experiment["minimum_size"]) / steps)
    sizes = [experiment["minimum_size"] + i * step for i in range(steps)] + [experiment["maximum_size"]]
    return list(flat_map(lambda x: ["-i", str(x)], sizes))


def flat_map(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


if __name__ == "__main__":
    run_benchmark()



