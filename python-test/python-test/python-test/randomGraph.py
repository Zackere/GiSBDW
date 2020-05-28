from scipy.special import comb
from networkx.generators.random_graphs import gnm_random_graph
import random

def maxEdges(n):
    return n*(n-1)//2

def getRandomGraph(n):
    x = random.randint(0, 2**maxEdges(n))
    e = distribuant(n,x)
    return gnm_random_graph(n,e)
    
def distribuant(n, x):
    sum = 0
    maxE = maxEdges(n)
    for e in range(maxE+1):
        numberOfGraphs = comb(maxE,e)
        sum = sum + numberOfGraphs
        if x < sum:
            return e
