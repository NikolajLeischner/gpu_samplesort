import argparse
import itertools
import json
import os
import subprocess

import chevron


def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--executable", help="path to the benchmark executable")
    parser.add_argument("-c", "--configuration", help="path to the benchmark configuration")
    args = parser.parse_args()

    configuration = json.load(open(args.configuration))

    os.makedirs("data", exist_ok=True)

    results = []
    for index, experiment in enumerate(configuration["experiments"]):
        results += [run_experiment(args.executable, configuration["algorithms"], experiment, index)]

    template = open("charts.mustache").read()
    html = chevron.render(template, {"chart": results})
    open("charts.html", "w").write(html)


def run_experiment(executable, algorithms, experiment, index):
    sizes = size_arguments_for_experiment(experiment)
    settings = experiment["settings"].split()
    title = experiment["title"]
    results = []
    for algorithm_index, algorithm in enumerate(algorithms):
        file_name = "data/result-" + str(index) + "-" + algorithm + ".csv"
        args = [executable, "-a", algorithm, "-o", file_name] + sizes + settings
        print("Running benchmark '" + title + "': " + str(args))
        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        results += [parse_result(file_name, algorithm, algorithm_index)]
        # remove_file(file_name)
    return chart_data_for_experiment(title, results, index)


def chart_data_for_experiment(title, results, index):
    return {"id": index, "title": title, "series": json.dumps(results)}


def parse_result(result_file, algorithm_name, algorithm_index):
    colors = ["#ff6384", "#36a2eb", "#cc65fe", "#ffce56"]
    data = [[int(line.split(";")[0]), float(line.split(";")[1])] for line in open(result_file)]
    return {"label": algorithm_name, "backgroundColor": colors[algorithm_index], "fill": False,
            "data": list(map(lambda x: {"x": x[0], "y": (x[0] / x[1]) / 1000.0}, data))}


def size_arguments_for_experiment(experiment):
    steps = experiment["steps"] - 1
    step = int((experiment["maximum_size"] - experiment["minimum_size"]) / steps)
    sizes = [experiment["minimum_size"] + i * step for i in range(steps)] + [experiment["maximum_size"]]
    return list(flat_map(lambda x: ["-i", str(x)], sizes))


def remove_file(file_name):
    try:
        os.remove(file_name)
    except OSError:
        pass


def flat_map(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))


if __name__ == "__main__":
    run_benchmark()
