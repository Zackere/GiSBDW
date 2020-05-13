import sys
import json
import subprocess
from os import listdir, chdir
from os.path import isfile, join, abspath, basename
import argparse

import matplotlib.pyplot as plt


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
    results = [ExecuteAlgorithm(binPath, 'dyn', filename, outputPath) for filename in filenames]
    times = [float(result['timeElapsed']) for result in results]
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    #langs = ['C', 'C++', 'Java', 'Python', 'PHP']
    #students = [23,17,35,29,12]
    ax.bar(filenames,times)
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


if __name__ == "__main__":

    LoadConfig("config.json")
    parser = CreateParser()
    args = parser.parse_args();
    RunBenchmarkPlot(config["paths"]["benchmarkGraphs"])
