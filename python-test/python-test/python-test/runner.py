import argparse
import os
import utils
import json
import subprocess
import sys
from os.path import isfile, join, abspath, basename
from os import listdir, chdir



def CreateParser():
    algorithms = ['bnbCPU', 'dynCPU', 'dynCPUImprov']

    parser = argparse.ArgumentParser(
        description='Run algorithms on graphs in given directory',
        epilog=f"Example {sys.argv[0]} --input /path/to/inputdir --output /path/to/outputdir"
        )
    parser.add_argument('--input', '-i', metavar='dir', type=str, nargs=1,
                    help='Path to input directory', required=True)

    parser.add_argument('--output', '-o', metavar='dir', type=str, nargs=1,
                    help='Path to output directory', required=True)

    parser.add_argument('--algorithm', '-a', metavar='alg', type=str, nargs="+",
                    help='Algorithms to run.\nOne or more from: [%(choices)s]', required=True, choices=algorithms)
    parser.add_argument('--runsPerGraph', '-r', metavar='numOfRuns', type=int, nargs=1, default=[1],
                        help='Specify how many times time measurement should be repeated for each graph. Default = 1')

    parser.add_argument('--timeout', '-t', metavar='seconds', type=float, nargs=1, default=[None],
                        help='Results will be displayed after first timeout')
    return parser

def LoadConfig(path):
    with open('config.json') as jsonConfig:
        global config
        config = json.load(jsonConfig)
    for key, value in config["paths"].items():
        config["paths"][key] = abspath(value)

def ExecuteAlgorithm(pathToBin, algorithmType, inputFile, outputPath, timeout):
    command = f"{pathToBin} -a {algorithmType} -o {outputPath} {inputFile}"
    result = subprocess.run(command, timeout=timeout)
    if result.returncode != 0:
        raise ChildProcessError(f"{command} did not succeed. Return code: {result}")

def GetAbsoluteFilePaths(path):
    return [join(abspath(path), element) for element in listdir(path) if isfile(join(path, element))]

def ExecuteAlgorithms(inputDir, outputDir, algorithms, timeout, runsPerGraph):
    graphsAbsolutePaths = GetAbsoluteFilePaths(inputDir)
    binPath = config["paths"]["bin"]
    timeouted = []
    for graphPath in graphsAbsolutePaths:
        if len(timeouted) == len(algorithms):
            break
        for algorithmType in algorithms:
            if not algorithmType in timeouted:
                for i in range(runsPerGraph):
                    try:
                        ExecuteAlgorithm(binPath, algorithmType, graphPath, outputDir, timeout)
                    except subprocess.TimeoutExpired as ex:
                        print(f"{algorithmType} timeouted")
                        timeouted.append(algorithmType)
                        break


config = {}
if __name__ == "__main__":
    LoadConfig("config.json")
    parser = CreateParser()
    args = vars(parser.parse_args())
    inputDir = args['input'][0]
    outputDir = args['output'][0]
    algorithms = args['algorithm']
    timeout = args['timeout'][0]
    runsPerGraph = args['runsPerGraph'][0]
    utils.OverwriteDir(outputDir)
    ExecuteAlgorithms(
           inputDir,
           outputDir,
           algorithms,
           timeout,
           runsPerGraph
       )
