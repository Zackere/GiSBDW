import sys
import json
import subprocess
import argparse
import GraphGenerator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir, chdir
from os.path import isfile, join, abspath, basename

def RunTest(inputPath, xAxis, yAxis, plotType, description):
    df = GatherData(inputPath)
    df.sort_values(by=[xAxis], inplace = True)
    ShowResults(df, xAxis=xAxis, yAxis=yAxis, hue="algorithm", plotType=plotType, description=description)


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


def ShowResults(dataFrame, xAxis, yAxis, hue, plotType, description):
    print(dataFrame.to_string())
    if plotType == "bar":
        ax = sns.barplot(x=xAxis, y=yAxis, hue=hue, data=dataFrame)
    elif plotType == "scatter":
        ax = sns.scatterplot(x=xAxis, y=yAxis, data=dataFrame, hue=hue)
    elif plotType == "lm":
        ax = sns.lmplot(x=xAxis, y=yAxis, data=dataFrame, hue=hue, lowess=True)
    else:
        raise ValueError(f"Wrong plotType specified: {plotType}")
    plt.figtext(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=12)
    plt.show()


def GatherData(path):
    binPath = config["paths"]["bin"]
    outputPath = path + "Out"
    filenames = GetAbsoluteFilePaths(path)
    columns = ["algorithm","filename","timeElapsed","edges","vertices"]
    data = {}
    for column in columns:
        data[column] = []
    for filename in filenames:
        for algorithmType in args["algorithm"]:
            runsPerGraph, = args["runsPerGraph"]
            timeElapsed = 0
            for i in range(runsPerGraph):
                result = ExecuteAlgorithm(binPath, algorithmType, filename, outputPath)
                timeElapsed = timeElapsed + float(result["timeElapsed"])
            data["timeElapsed"].append(timeElapsed/runsPerGraph)
            data["filename"].append(basename(filename))
            data["algorithm"].append(algorithmType)
            data["edges"].append(int(result["edges"]))
            data["vertices"].append(int(result["vertices"]))

    return pd.DataFrame(data)



def CreateParser():
    algorithms = ['bnbCPU', 'dyn', 'hyb', 'dynGPU']
    tests = ['timeElapsed']

    parser = argparse.ArgumentParser(
        description='Run benchmarks for treedepth algorithms',
        epilog="Example app.py --random 10 0.4 12 --a dyn bnb"
        )

    parser.add_argument('--algorithm', '-a', metavar='alg', type=str, nargs="+",
                    help='Algorithm to run.\nOne or more from: [%(choices)s]', required=True, choices=algorithms)
    parser.add_argument('--runsPerGraph', '-r', metavar='numOfRuns', type=int, nargs=1, default=[1],
                        help='Specify how many times time measurement should be repeated for each graph. Default = 1')

    inputGroup = parser.add_mutually_exclusive_group(required=True)
    inputGroup.add_argument('--benchmark', action='store_true', help="Run on benchmark graphs.")
    inputGroup.add_argument('--random', metavar=('v','d','n'), type=float, nargs=3,
                           help="""Run on random graphs.
                           v - number of vertices,
                           d - density,
                           n - number of graphs""")
    inputGroup.add_argument('--density', metavar=('v','dLow','dHigh','n'), type=float, nargs=4,
                           help="""Run on random graphs with incrasing density.
                           v - number of vertices,
                           dLow - starting density,
                           dHigh - end density
                           n - number of graphs""")
    inputGroup.add_argument('--vertices', metavar=('vLow','vHigh', 'd', 'n'), type=float, nargs=4,
                           help="""Run on random graphs with incrasing vertices count.
                           vLow - starting number of vertices,
                           vHigh- ending number of vertices,
                           d - density
                           n - number of graphs""")
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
    parser = CreateParser()
    args = vars(parser.parse_args());

    executePath = ""
    if args["benchmark"]:
        executePath = config["paths"]["benchmarkGraphs"]
        RunTest(executePath, "filename", "timeElapsed", "bar", "Benchmark graphs")

    elif args["random"]:
        v, d, n = args["random"]
        v = int(v)
        n = int(n)
        gg.GenerateRandomGraphs(n,d,v)
        executePath = config["paths"]["randomGraphs"]
        RunTest(executePath, "filename", "timeElapsed", "scatter", f"Random graphs with {v} vertices")

    elif args["density"]:
        v, dLow, dHigh, n = args["density"]
        v = int(v)
        n = int(n)
        gg.GenerateGraphsWithIncrasingDensity(n,dLow,dHigh,v)
        executePath = config["paths"]["randomGraphs"]
        RunTest(executePath, "edges", "timeElapsed", "lm", f"Random graphs with {v} vertices")

    elif args["vertices"]:
        vLow, vHigh, d, n = args["vertices"]
        vLow = int(vLow)
        vHigh = int(vHigh)
        n = int(n)
        gg.GenerateGraphsWithIncrasingNumberOfVertices(
            n, vLow, vHigh, d)
        executePath = config["paths"]["randomGraphs"]
        RunTest(executePath, "vertices", "timeElapsed", "lm", f"Graphs with density {d}")
