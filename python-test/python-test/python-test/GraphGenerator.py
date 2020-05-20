import networkx as nx
import utils
import os

class GraphGenerator(object):
    def __init__(self, randomGraphsPath):
        self.randomGraphsPath = randomGraphsPath
        self.randomGraphsOutputPath = f"{randomGraphsPath}Out"


    def DeleteRandomGraphsDir(self):
        utils.DeleteDir(self.randomGraphsPath)

    def CreateRandomGraphsDir(self):
        utils.CreateDir(self.randomGraphsPath)

    def DeleteRandomGraphsOutputDir(self):
        utils.DeleteDir(self.randomGraphsOutputPath)

    def CreateRandomGraphsOutputDir(self):
        utils.CreateDir(self.randomGraphsOutputPath)

    def ResetRandomGraphsOutputDir(self):
        self.DeleteRandomGraphsOutputDir()
        self.CreateRandomGraphsOutputDir()

    def ResetRandomGraphsDir(self):
        self.DeleteRandomGraphsDir()
        self.CreateRandomGraphsDir()

    def ResetRandomGraphsDirs(self):
        self.ResetRandomGraphsDir()
        self.ResetRandomGraphsOutputDir()

    def SaveRandomGraphs(self,graphs):
        for i, graph in enumerate(graphs):
            self.SaveGraph(self.randomGraphsPath, f"G{i:05d}", graph)

    def SaveGraph(self, path, filename, graph):
        outputPath = os.path.join(self.randomGraphsPath, filename)
        nx.drawing.nx_pydot.write_dot(graph, outputPath)

    def GenerateRandomGraphs(self, numberOfGraphs, density, numberOfVertices):
        self.ResetRandomGraphsDir()
        graphs = [nx.erdos_renyi_graph(numberOfVertices, density) for i in range(numberOfGraphs)]
        self.SaveRandomGraphs(graphs)

    def GenerateGraphsWithIncrasingDensity(self, numberOfGraphs, densityMin, densityMax, numberOfVertices):
        self.ResetRandomGraphsDirs()
        step = (densityMax - densityMin) / numberOfGraphs
        graphs = [nx.erdos_renyi_graph(numberOfVertices, densityMin + i*step) for i in range(numberOfGraphs)]
        self.SaveRandomGraphs(graphs)

    def GenerateGraphsWithIncrasingNumberOfVertices(self, numberOfGraphs, vLow, vHigh, density):
        self.ResetRandomGraphsDirs()
        step = (vHigh - vLow) / numberOfGraphs
        graphs = [nx.erdos_renyi_graph(round(vLow + i*step), density) for i in range(numberOfGraphs)]
        self.SaveRandomGraphs(graphs)

    def GeneratePaths(self, vLow, vHigh, step):
        self.ResetRandomGraphsDirs()
        graphs = [(nx.path_graph(i), f"path{(i):05d}") for i in range(vLow,vHigh,step)]
        for graph, filename in graphs:
            self.SaveGraph(self.randomGraphsPath, filename, graph)

    def GenerateCliques(self, vLow, vHigh, step):
        self.ResetRandomGraphsDirs()
        graphs = [(nx.complete_graph(i), f"clique{(i):05d}") for i in range(vLow,vHigh,step)]
        for graph, filename in graphs:
            self.SaveGraph(self.randomGraphsPath, filename, graph)
