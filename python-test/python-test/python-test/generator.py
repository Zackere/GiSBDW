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
    parser.add_argument('--connected', '-c', metavar='dir',
                        action='store_const', const=True, default = False,
                    help='Filter out not connected graphs.', required=False)

    return parser


if __name__ == "__main__":
    parser = CreateParser()
    args = vars(parser.parse_args())
    expression = args['exp'][0].strip('"').strip("'")
    outputDir = args['output'][0]
    filterNotConnected = args['connected']
    print(f"Args supplied:\nExpression -> {expression}")
    print(f"Output directory -> {outputDir}")
    print(f"Filter not connected graphs -> {filterNotConnected}")
    graphs = eval(expression)
    if filterNotConnected:
        graphs = filter(lambda graph : networkx.is_connected(graph), graphs)
    utils.OverwriteDir(outputDir)
    for i, graph in enumerate(graphs):
        filename = f"G{(i):05d}"
        outputPath = os.path.join(outputDir, filename)
        networkx.drawing.nx_pydot.write_dot(graph, outputPath)
        print(f"Saved graph {filename}")
    print("Done")
