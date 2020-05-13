import networkx as nx
import shutil
import os

class GraphGenerator(object):
    """description of class"""
    def __init__(self, randomGraphsPath):
        self.randomGraphsPath = randomGraphsPath

    def GenerateTemporaryGraphs(self, func, *args):
        dir = tf.TemporaryDirectory()
        func(args)
        dir.cleanup()

    def DeleteRandomGraphsDir(self):
        if os.path.exists(self.randomGraphsPath):
            shutil.rmtree(self.randomGraphsPath)

    def CreateRandomGraphsDir(self):
        os.mkdir(self.randomGraphsPath)

    def ResetRandomGraphsDir(self):
        self.DeleteRandomGraphsDir()
        self.CreateRandomGraphsDir()

    def GenerateRandomGraphs(self, numberOfGraphs, density, numberOfVertices):
        self.ResetRandomGraphsDir()
        graphs = [nx.erdos_renyi_graph(numberOfVertices, density) for i in range(numberOfGraphs)]
        for i, graph in enumerate(graphs):
            outputPath = os.path.join(self.randomGraphsPath, f"G{i}")
            nx.drawing.nx_pydot.write_dot(graph, outputPath)
