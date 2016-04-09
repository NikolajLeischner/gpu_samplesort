import argparse
import json
import subprocess
import itertools
import pystache


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--executable", help="path to the benchmark executable")
    parser.add_argument("-c", "--configuration", help="path to the benchmark configuration")
    args = parser.parse_args()

    configuration = json.load(open(args.configuration))

    results = []
    for index, experiment in enumerate(configuration["experiments"]):
        results += [run_experiment(args.executable, configuration["algorithms"], experiment, index)]

    template = open("charts.mustache").read()
    html = pystache.render(template, {"chart": results})
    open("charts.html", "w").write(html)


def run_experiment(executable, algorithms, experiment, index):
    sizes = size_arguments_for_experiment(experiment)
    settings = experiment["settings"].split()
    title = experiment["title"]
    results = []
    file_name = "data/result" + str(index) + ".csv"
    for algorithm in algorithms:
        args = [executable, "-a", algorithm, "-o", file_name] + sizes + settings
        print("Running benchmark '" + title + "': " + str(args))
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        results += [parse_result(file_name, algorithm)]
    return chart_data_for_experiment(title, results, index)


def chart_data_for_experiment(title, results, index):
    return {"id": index, "title": title, "series": json.dumps(results)}


def parse_result(result_file, algorithm_name):
    data = [[int(line.split(";")[0]), float(line.split(";")[1])] for line in open(result_file)]
    return {"name": algorithm_name,
            "data": list(map(lambda x: [x[0], 1000000 * x[1] / x[0]], data))}


def size_arguments_for_experiment(experiment):
    steps = experiment["steps"] - 1
    step = int((experiment["maximum_size"] - experiment["minimum_size"]) / steps)
    sizes = [experiment["minimum_size"] + i * step for i in range(steps)] + [experiment["maximum_size"]]
    return list(flat_map(lambda x: ["-i", str(x)], sizes))


def flat_map(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


if __name__ == "__main__":
    run_benchmark()
