import random
from nest.lib.hl_api_types import *

import numpy as np
import matplotlib.pyplot as plt
import nest
#import nest.voltage_trace

nest.ResetKernel()


def CreateWTA(width=10, height=5, min_k=2, max_k=10):
    # Generate list of positions corresponding to the 10x5 grid with K neurons in each point (x,y) of the grid
    pos_list = [(x, y, z) for x in range(width) for y in range(height) for z in range(0, random.randint(min_k, max_k))]
    res_layer = nest.Create('iaf_psc_exp', positions=nest.spatial.free(pos_list))
    return res_layer


# Creates a visualization showing the number of neurons k per WTA circuit on the grid
def visualizeNodes(nc: NodeCollection):
    nodelocationlist = nest.GetPosition(nc)
    data = np.zeros((5, 10))
    for i in nodelocationlist:
        data[int(i[1]), int(i[0])] += 1.0
    #print(data)

    fig, ax = plt.subplots()
    im = ax.imshow(data)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(5))

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(5):
        for j in range(10):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Number of neurons of each WTA circuit on the (%dx%d) grid" % (10, 5))
    ax.set_xlabel("%d neurons total" % nc.spatial['network_size'])
    fig.tight_layout()
    plt.show()


def plotNodes(nc: NodeCollection):
    nest.PlotLayer(nc, nodecolor='b')
    plt.show()


layer = CreateWTA()
visualizeNodes(layer)
plotNodes(layer)
