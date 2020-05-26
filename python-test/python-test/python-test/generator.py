import argparse
import networkx
from networkx.generators import *
import utils
import os
import sys
from numpy import linspace

def CreateParser():
    parser = argparse.ArgumentParser(
        description='Generate graphs using NetworkX library',
        epilog=f"Example {sys.argv[0]} --exp '(networkx.generators.classic.binomial_tree(i) for i in range(1,10))' -o /path/to/dir"
        )
    parser.add_argument('--exp', '-e', metavar = 'expression', type=str, nargs=1,
                        help='Graphs generating expression', required=True)
    parser.add_argument('--output', '-o', metavar='dir', type=str, nargs=1,
                    help='Output directory path.', required=True)
    return parser


if __name__ == "__main__":
    parser = CreateParser()
    args = vars(parser.parse_args())
    expression = args['exp'][0].strip('"').strip("'")
    outputDir = args['output'][0]
    print(f"Args supplied:\nExpression -> {expression}\nOutput directory -> {outputDir}")
    graphs = eval(expression)
    utils.OverwriteDir(outputDir)
    for i, graph in enumerate(graphs):
        filename = f"G{(i):05d}"
        outputPath = os.path.join(outputDir, filename)
        networkx.drawing.nx_pydot.write_dot(graph, outputPath)
        print(f"Saved graph {filename}")
    print("Done")
