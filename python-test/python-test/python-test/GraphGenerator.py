import networkx as nx
import shutil
import os

class GraphGenerator(object):
    def __init__(self, randomGraphsPath):
        self.randomGraphsPath = randomGraphsPath
        self.randomGraphsOutputPath = f"{randomGraphsPath}Out"


    def DeleteRandomGraphsDir(self):
        if os.path.exists(self.randomGraphsPath):
            shutil.rmtree(self.randomGraphsPath)

    def CreateRandomGraphsDir(self):
        os.mkdir(self.randomGraphsPath)

    def DeleteRandomGraphsOutputDir(self):
        if os.path.exists(self.randomGraphsOutputPath):
            shutil.rmtree(self.randomGraphsOutputPath)

    def CreateRandomGraphsOutputDir(self):
        os.mkdir(self.randomGraphsOutputPath)

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
            outputPath = os.path.join(self.randomGraphsPath, f"G{i:05d}")
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
