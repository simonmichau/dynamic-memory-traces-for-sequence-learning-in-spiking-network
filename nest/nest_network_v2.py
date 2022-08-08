import random
import math
from nest.lib.hl_api_types import *

import numpy as np
import matplotlib.pyplot as plt
import nest

nest.ResetKernel()


class WTACircuit:
    def __init__(self, nc: NodeCollection, pos):
        self.nc = nc
        self.pos = pos
        self.k = self.getSize()

    def getPos(self):
        """Returns the (x, y) position of the WTA circuit"""
        return self.pos

    def getNodeCollection(self):
        """Returns the NodeCollection nc"""
        return self.nc

    def getSize(self):
        """Returns the size of the NodeCollection nc"""
        return len(self.nc.get('global_id'))


class Network(object):
    def __init__(self, **kwds):
        # Rise and decay time constant for EPSP (in ms)
        self.tau_rise = kwds.get('tau_rise', 2)
        self.tau_decay = kwds.get('tau_decay', 20)
        # Dimensions of the grid of WTA circuits
        self.grid_shape = kwds.get('grid_shape', (10, 5))
        self.n, self.m = self.grid_shape
        # Upper and lower bound for randomly drawn number k of neurons in each WTA circuit
        self.min_k = kwds.get('min_k', 2)
        self.max_k = kwds.get('max_k', 10)

        # List containing all WTA circuits
        self.circuits = self.createNodes()

    def createNodes(self) -> list:
        circuit_list = []
        for m in range(self.m):
            for n in range(self.n):
                K = random.randint(self.min_k, self.max_k)
                nc = nest.Create('iaf_psc_exp', K, params={'I_e': 188.0,  # 0.0
                                                           'tau_syn_ex': self.tau_rise,
                                                           'tau_m': self.tau_decay
                                                           })
                circuit_list.append(WTACircuit(nc, (n, m)))
        return circuit_list

    def getCircuitSizeGrid(self):
        """Returns a (nxm) array containing the neuron frequencies per grid point"""
        data = np.zeros((self.m, self.n))
        for circuit in self.circuits:
            data[circuit.getPos()[1], circuit.getPos()[0]] = circuit.getSize()
        return data

    def visualizeCircuits(self):
        """Creates a 2D visualization showing the number of neurons k per WTA circuit on the grid"""
        data = self.getCircuitSizeGrid()

        fig, ax = plt.subplots()
        im = ax.imshow(data)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(self.n))
        ax.set_yticks(np.arange(self.m))

        # Rotate the tick labels and set their alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Loop over data dimensions and create text annotations
        for i in range(self.m):
            for j in range(self.n):
                text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")

        ax.set_title("Number of neurons of each WTA circuit on the (%dx%d) grid" % (self.n, self.m))
        ax.set_xlabel("%d neurons total" % np.sum(data))
        fig.tight_layout()
        plt.show()


test = Network()
print(test.getCircuitSizeGrid())
test.visualizeCircuits()
