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


def ShowResultss(dataFrame, groupBy, xAxes, yAxes, plotTypes, description):
    numberOfSubplots = len(xAxes)
    fig, axes = plt.subplots(1,numberOfSubplots)
    if numberOfSubplots == 1:
        axes = [axes]
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

def CreateParser():
    parser = argparse.ArgumentParser(
        description='Run benchmarks for treedepth algorithms',
        epilog="Example app.py --random 10 0.4 12 --a dyn bnb"
        )

    parser.add_argument('--input', '-i', metavar='dir', type=str, nargs=1,
                        help='Input dir for stats')

    parser.add_argument('--xAxis', '-x', metavar='stat', type=str, nargs='+',
                        help='Variable of x axis', required=True)
    parser.add_argument('--yAxis', '-y', metavar='stat', type=str, nargs='+',
                        help='Variable of y axis', required=True)
    parser.add_argument('--plot', '-p', metavar='plotType', type=str, nargs='+',
                        help='Plot type', required=True)
    parser.add_argument('--desc', '-d', metavar='description', type=str, nargs=1,
                        help='Plot description', default=[''], required=False)
    return parser


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
    df = pd.DataFrame(data)
    df = df.groupby(["algorithm", "filename"], as_index=False).mean()
    return df


if __name__ == "__main__":
    parser = CreateParser()
    args = vars(parser.parse_args());
    inputDir = args["input"][0]
    df = LoadDataToDataFrame(inputDir)
    xAxes = args['xAxis']
    yAxes = args['yAxis']
    plotTypes = args['plot']
    groupBy = ["algorithm" for i in range(len (xAxes))]
    description = args['desc'][0]
    ShowResultss(
        df,
        groupBy=groupBy,
        xAxes=xAxes,
        yAxes=yAxes,
        plotTypes=plotTypes,
        description=description
    )
