import sys
import json
import subprocess
from os import listdir, chdir
from os.path import isfile, join, abspath, basename
import argparse
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

import GraphGenerator

def GetOutput(outputPath, inputFilename):
    path = f"{outputPath}/{basename(inputFilename)}{config['outputExtension']}"
    with open(path) as json_file:
        return json.load(json_file)

def ExecuteAlgorithm(pathToBin, algorithmType, inputFile, outputPath):

    command = f"{pathToBin} -a {algorithmType} -o {outputPath} {inputFile}"
    result = subprocess.run(command)
    if result.returncode != 0:

        raise ChildProcessError(f"{command} did not succeed. Return code: {result}")

    return GetOutput(outputPath, inputFile)

def GetAbsoluteFilePaths(path):
    return [join(path, element) for element in listdir(path) if isfile(join(path, element))]


def RunBenchmarkPlot(path):
    binPath = config["paths"]["bin"]
    outputPath = config["paths"]["benchmarkGraphs"] + "Out"
    filenames = GetAbsoluteFilePaths(path)
    columns = ["algorithm","filename","timeElapsed"]
    data = {}
    for column in columns:
        data[column] = []
    for filename in filenames:
        for algorithmType in args["algorithm"]:
            result = ExecuteAlgorithm(binPath, algorithmType, filename, outputPath)
            data["timeElapsed"].append(float(result["timeElapsed"]))
            data["filename"].append(basename(filename))
            data["algorithm"].append(algorithmType)
    df = pd.DataFrame(data)
    print(df)
    ax = sns.barplot(x="filename", y="timeElapsed", hue="algorithm", data=df)

    ax.plot()
    plt.show()

def CreateParser():
    parser = argparse.ArgumentParser(description='Run benchmarks for treedepth algorithms')
    parser.add_argument('--algorithm', '-a', metavar='a', type=str, nargs="+",
                    help='algorithm to run', required=True, choices=['bnb', 'dyn', 'hyb'])
    return parser

def LoadConfig(path):
    with open('config.json') as jsonConfig:
        global config
        config = json.load(jsonConfig)
    for key, value in config["paths"].items():
        config["paths"][key] = abspath(value)

config = {}
args = {}

if __name__ == "__main__":
    LoadConfig("config.json")
    gg = GraphGenerator.GraphGenerator(config["paths"]["randomGraphs"])
    gg.GenerateRandomGraphs(10,0.5,15)
    parser = CreateParser()
    args = vars(parser.parse_args());
    RunBenchmarkPlot(config["paths"]["randomGraphs"])
