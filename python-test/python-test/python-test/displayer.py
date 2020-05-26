import sys
import json
import utils
import subprocess
import argparse
import GraphGenerator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil, sqrt
from os import listdir, chdir
from os.path import isfile, join, abspath, basename


def PlotLowerBound(x, n, ax, label):
    def LowerBound(n,e):
        val = 0.5 + n - sqrt(0.25+n+(n*n)-(2*n)-(2*e))
        return ceil(val)
    x = np.arange(np.amin(x),np.amax(x)+1)
    y = [LowerBound(n,edge) for edge in x]
    ax.plot(x,y,".m", label=label)
    ax.legend()


def RunTest(inputPath, groupBy, xAxes, yAxes, plotTypes, description):
    dataFrame = GatherData(inputPath)
    ShowResultss(dataFrame=dataFrame,
                groupBy=groupBy,
                xAxes=xAxes,
                yAxes=yAxes,
                plotTypes=plotTypes,
                description=description)

def ShowResultss(dataFrame, groupBy, xAxes, yAxes, plotTypes, description):
    numberOfSubplots = len(xAxes)
    fig, axes = plt.subplots(1,numberOfSubplots)
    for ax, group, xAxis, yAxis, plotType in zip(axes, groupBy, xAxes, yAxes, plotTypes):
        dataFrame.sort_values(by=[group, xAxis], inplace = True)
        print(f"Result dataframe. Ordered by [{group}, {xAxis}]")
        print(dataFrame.to_string())
        if plotType == "scatterFit":
            ScatterFitPlot(dataFrame, group, xAxis, yAxis, ax, 5)
            if(xAxis == "edges" and yAxis == "treedepth"):
                axisIndex = yAxes.index("treedepth")
                PlotLowerBound(
                    dataFrame["edges"],
                    dataFrame["vertices"][0],
                    axes[axisIndex],
                    "Lower bound")
        elif plotType == "bar":
            sns.barplot(x=xAxis, y=yAxis, data=dataFrame, hue=group, ax=ax)
        else:
            raise ValueError(f"Wrong plot type specified -> {plotType}")
        if xAxis == "filename":
            plt.setp(ax.get_xticklabels(), rotation=90)
    plt.figtext(0.5, 0.01, description, wrap=True, horizontalalignment='center', fontsize=12)
    plt.tight_layout()
    plt.show()

def ScatterFitPlot(dataFrame, groupBy, xAxis, yAxis, ax, degree = 3):
    colors = "bgrcmykw"
    markers = "os+D*px"
    for i, uniqueGroup in enumerate(dataFrame[groupBy].unique()):
        filteredDf = dataFrame.loc[dataFrame[groupBy] == uniqueGroup]
        xAxisData = filteredDf[xAxis]
        yAxisData = filteredDf[yAxis]
        p = np.poly1d(np.polyfit(xAxisData,yAxisData,degree))
        x0 = np.amax(xAxisData)
        x1 = np.amin(xAxisData)
        t = np.linspace(x0, x1, 200)
        ax.plot(xAxisData, yAxisData, f'{colors[i%len(colors)]}{markers[i%len(markers)]}', label=uniqueGroup)
        ax.plot(t, p(t), f':{colors[i]}')
    ax.set_xlabel(xAxis)
    ax.set_ylabel(yAxis)
    ax.legend()

def LoadJson(path):
    with open(path) as json_file:
        return json.load(json_file)

def GetOutput(outputPath, inputFilename):
    path = f"{outputPath}/{basename(inputFilename)}{config['outputExtension']}"
    with open(path) as json_file:
        return json.load(json_file)

def ExecuteAlgorithm(pathToBin, algorithmType, inputFile, outputPath, timeout):
    command = f"{pathToBin} -a {algorithmType} -o {outputPath} {inputFile}"
    result = subprocess.run(command, timeout=timeout)
    if result.returncode != 0:
        raise ChildProcessError(f"{command} did not succeed. Return code: {result}")

    return GetOutput(outputPath, inputFile)

def GetAbsoluteFilePaths(path):
    return [join(path, element) for element in listdir(path) if isfile(join(path, element))]


def GatherData(path):
    timeouted = False
    binPath = config["paths"]["bin"]
    outputPath = path + "Out"
    filenames = GetAbsoluteFilePaths(path)
    columns = ["algorithm","filename","timeElapsed","edges","vertices","treedepth"]
    data = {}
    for column in columns:
        data[column] = []
    for filename in filenames:
        if not timeouted:
            for algorithmType in args["algorithm"]:
                runsPerGraph, = args["runsPerGraph"]
                timeElapsed = 0
                for i in range(runsPerGraph):
                    try:
                        result = ExecuteAlgorithm(binPath, algorithmType, filename, outputPath, args["timeout"][0])
                    except subprocess.TimeoutExpired as ex:
                        print(f"Timeout on graph {basename(filename)}")
                        timeouted = True
                        data["timeElapsed"].append(0)
                        data["filename"].append(basename(filename))
                        data["algorithm"].append(algorithmType)
                        data["edges"].append(0)
                        data["vertices"].append(0)
                        data["treedepth"].append(0)
                        break
                    timeElapsed = timeElapsed + float(result["timeElapsed"])
                data["timeElapsed"].append(timeElapsed/runsPerGraph)
                data["filename"].append(basename(filename))
                data["algorithm"].append(algorithmType)
                data["edges"].append(int(result["edges"]))
                data["vertices"].append(int(result["vertices"]))
                data["treedepth"].append(int(result["treedepth"]))
    return pd.DataFrame(data)


def CreateParser():
    #algorithms = ['bnbCPU', 'dynCPU', 'dynCPUImprov']
    #tests = ['timeElapsed']
    #benchmarkGraphs = ['paths', 'cliques']

    parser = argparse.ArgumentParser(
        description='Run benchmarks for treedepth algorithms',
        epilog="Example app.py --random 10 0.4 12 --a dyn bnb"
        )

    #parser.add_argument('--algorithm', '-a', metavar='alg', type=str, nargs="+",
    #                help='Algorithm to run.\nOne or more from: [%(choices)s]', required=True, choices=algorithms)
    #parser.add_argument('--runsPerGraph', '-r', metavar='numOfRuns', type=int, nargs=1, default=[1],
    #                    help='Specify how many times time measurement should be repeated for each graph. Default = 1')

    parser.add_argument('--input', '-i', metavar='dir', type=str, nargs=1,
                        help='Input dir for stats')

    #inputGroup = parser.add_mutually_exclusive_group(required=True)
    #inputGroup.add_argument('--benchmark', type=str, nargs=1,
    #                       help="Run on group of benchmark graphs. One from [%(choices)s]", choices=benchmarkGraphs)
    #inputGroup.add_argument('--random', metavar=('v','d','n'), type=float, nargs=3,
    #                       help="""Run on random graphs.
    #                       v - number of vertices,
    #                       d - density,
    #                       n - number of graphs""")
    #inputGroup.add_argument('--density', metavar=('v','dLow','dHigh','n'), type=float, nargs=4,
    #                       help="""Run on random graphs with incrasing density.
    #                       v - number of vertices,
    #                       dLow - starting density,
    #                       dHigh - end density
    #                       n - number of graphs""")
    #inputGroup.add_argument('--vertices', metavar=('vLow','vHigh', 'd', 'n'), type=float, nargs=4,
    #                       help="""Run on random graphs with incrasing vertices count.
    #                       vLow - starting number of vertices,
    #                       vHigh- ending number of vertices,
    #                       d - density
    #                       n - number of graphs""")
    return parser

def LoadConfig(path):
    with open('config.json') as jsonConfig:
        global config
        config = json.load(jsonConfig)
    for key, value in config["paths"].items():
        config["paths"][key] = abspath(value)

config = {}
args = {}

def LoadDataToDataFrame(path):
    files = utils.GetAbsoluteFilePaths(path)
    files = list(filter(lambda x : not x.endswith(".gviz"), files))
    columns = ["algorithm","filename","timeElapsed","edges","vertices","treedepth"]
    data = {}
    for column in columns:
        data[column] = []
    for file in files:
        stats = LoadJson(file)
        data["filename"].append(basename(stats["graphName"]))
        data["algorithm"].append(stats["algorithmType"])
        data["edges"].append(int(stats["edges"]))
        data["vertices"].append(int(stats["vertices"]))
        data["treedepth"].append(int(stats["treedepth"]))
        data["timeElapsed"].append(float(stats["timeElapsed"]))
    return pd.DataFrame(data)

config = {}

if __name__ == "__main__":
    LoadConfig("config.json")
    parser = CreateParser()
    args = vars(parser.parse_args());
    inputDir = args["input"][0]
    df = LoadDataToDataFrame(inputDir)
    print(df)
    #if args["benchmark"]:
    #    executePath = config["paths"][args["benchmark"][0]]
    #    utils.OverwriteDir(f"{executePath}Out")
    #    RunTest(inputPath=executePath,
    #           groupBy=["algorithm", "algorithm"],
    #           xAxes=["filename","filename"],
    #           yAxes=["timeElapsed", "treedepth"],
    #           plotTypes=["bar", "bar"],
    #           description=f"Benchmark graphs")

    #elif args["random"]:
    #    v, d, n = args["random"]
    #    v = int(v)
    #    n = int(n)
    #    gg.GenerateRandomGraphs(n,d,v)
    #    executePath = config["paths"]["randomGraphs"]
    #    RunTest(inputPath=executePath,
    #           groupBy=["algorithm", "algorithm"],
    #           xAxes=["filename","filename"],
    #           yAxes=["timeElapsed", "treedepth"],
    #           plotTypes=["bar", "bar"],
    #           description=f"Random graphs with {v} vertices and density {d}")

    #elif args["density"]:
    #    v, dLow, dHigh, n = args["density"]
    #    v = int(v)
    #    n = int(n)
    #    gg.GenerateGraphsWithIncrasingDensity(n,dLow,dHigh,v)
    #    executePath = config["paths"]["randomGraphs"]
    #    RunTest(inputPath=executePath,
    #           groupBy=["algorithm", "algorithm"],
    #           xAxes=["edges","edges"],
    #           yAxes=["timeElapsed", "treedepth"],
    #           plotTypes=["scatterFit", "scatterFit"],
    #           description=f"Graphs with {v} vertices")

    #elif args["vertices"]:
    #    vLow, vHigh, d, n = args["vertices"]
    #    vLow = int(vLow)
    #    vHigh = int(vHigh)
    #    n = int(n)
    #    gg.GenerateGraphsWithIncrasingNumberOfVertices(
    #        n, vLow, vHigh, d)
    #    executePath = config["paths"]["randomGraphs"]
    #    RunTest(inputPath=executePath,
    #           groupBy=["algorithm", "algorithm"],
    #           xAxes=["vertices","vertices"],
    #           yAxes=["timeElapsed", "treedepth"],
    #           plotTypes=["scatterFit", "scatterFit"],
    #           description=f"Graphs with density {d}")