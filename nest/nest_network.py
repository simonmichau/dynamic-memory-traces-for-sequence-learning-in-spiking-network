import random
import math
from nest.lib.hl_api_types import *

import numpy as np
import matplotlib.pyplot as plt
import nest
#import nest.voltage_trace


nest.ResetKernel()


class WTACircuitGrid(object):
    def __init__(self, **kwds):
        # GLOBAL VARIABLES
        # Rise and decay time constant for EPSP (in ms)
        self.tau_rise = kwds.get('tau_rise', 2)
        self.tau_decay = kwds.get('tau_decay', 20)
        # Dimensions of the grid of WTA circuits
        self.grid_shape = kwds.get('grid_shape', (10, 5))
        self.n, self.m = self.grid_shape
        # Upper and lower bound for randomly drawn number k of neurons in each WTA circuit
        self.min_k = kwds.get('min_k', 2)
        self.max_k = kwds.get('max_k', 10)
        # Number of external inputs
        self.nInputs = kwds.get('nInputs', 50)
        # parameter of exponential distance distribution
        self.lam = kwds.get('lambda', 0.088)
        # simulation time (in ms)
        self.t_sim = kwds.get('t_sim', 2000.0)

        # Initialize NodeCollection
        self.nc = self.createNodes()

    def getNeighbours(self, n: NodeCollection) -> NodeCollection:  # TODO
        """Returns the NodeCollection of all nodes in the same WTA circuit as Node n"""
        #print(nest.GetPosition(n)[2])
        self.getNodeByPos(z=1)
        #nest.PlotProbabilityParameter(n, parameter=nest.GetConnections())

    def getNodeByPos(self, x=None, y=None, z=None) -> NodeCollection | None:
        """Return all Nodes matching the specified position"""
        # (x: 0-n, y: 0-m, z: 0-k_max)
        # Associate NodeID with position
        arr = []
        for node in self.nc:
            arr.append([nest.GetPosition(node), node.global_id])

        # loop over arr and add all NodeIDs to list when coordinated match
        global_id_list = []
        for a in arr:
            #if x == int(a[0][0]) and y == int(a[0][1]) and z == int(a[0][2]):
            #    global_id_list.append(a[1])
            if x is None:
                if y is None:
                    if z is None:
                        if True: global_id_list.append(a[1])
                    else:
                        if z == int(a[0][2]): global_id_list.append(a[1])
                else:
                    if z is None:
                        if y == int(a[0][1]): global_id_list.append(a[1])
                    else:
                        if y == int(a[0][1]) and z == int(a[0][2]): global_id_list.append(a[1])
            else:
                if y is None:
                    if z is None:
                        if x == int(a[0][0]): global_id_list.append(a[1])
                    else:
                        if x == int(a[0][0]) and z == int(a[0][2]): global_id_list.append(a[1])
                else:
                    if z is None:
                        if x == int(a[0][0]) and y == int(a[0][1]): global_id_list.append(a[1])
                    else:
                        if x == int(a[0][0]) and y == int(a[0][1]) and z == int(a[0][2]): global_id_list.append(a[1])

        matching_nodes = nest.NodeCollection(global_id_list)
        return matching_nodes

    def createNodes(self) -> NodeCollection:
        """Generate list of positions corresponding to the 10x5 grid with K neurons in each point (x,y) of the grid"""
        pos_list = [(x, y, z) for x in range(self.n) for y in range(self.m) for z in range(0, random.randint(self.min_k, self.max_k))]
        res_layer = nest.Create('iaf_psc_exp',
                                positions=nest.spatial.free(pos_list),
                                params={'I_e': 188.0,  # 0.0
                                        'tau_syn_ex': self.tau_rise,
                                        'tau_m': self.tau_decay
                                        }
                                )
        return res_layer

    def formConnections(self):
        conn_dict = {
            'rule': 'pairwise_bernoulli',
            'allow_autapses': False,
            # distance dependent connection probability p(d)=lam*exp(-lam*d)
            'p': self.lam * nest.spatial_distributions.exponential(
                (nest.spatial.distance.x**2 + nest.spatial.distance.y**2)**(1/2),
                beta=1/self.lam)
            #'mask': { # TODO: ensure neurons from the same WTA cannot be connected to each other
            #    'doughnut': {
            #        'inner_radius': 1.,
            #        'outer_radius': 10.
            #    }
            #}
        }
        syn_dict = {}  # TODO: implement synapse model with STP and STDP
        nest.Connect(self.nc, self.nc, conn_dict)

        print(nest.GetConnections(self.nc[0]))
        self.getNeighbours(self.nc[0])

        #self.measureNodeCollection(nc)

    def measureNodeCollection(self, nc: NodeCollection) -> None:
        """Simulates given NodeCollection for t_sim and plots the recorded spikes and membrane potential"""
        multimeter = nest.Create('multimeter')
        multimeter.set(record_from=['V_m'])
        spikerecorder = nest.Create('spike_recorder')
        nest.Connect(multimeter, nc)
        nest.Connect(nc, spikerecorder)

        nest.Simulate(self.t_sim)

        dmm = multimeter.get()
        Vms = dmm["events"]["V_m"]
        ts = dmm["events"]["times"]
        #plt.figure(1)
        plt.plot(ts, Vms)
        dSD = spikerecorder.get("events")
        evs = dSD["senders"]
        ts = dSD["times"]
        #plt.figure(2)
        plt.plot(ts, evs, ".")

    def getNeuronFrequencyGrid(self) -> np.ndarray:
        """Returns a (nxm) array containing the neuron frequencies per grid point"""
        nodelocationlist = nest.GetPosition(self.nc)
        data = np.zeros((self.m, self.n))
        for i in nodelocationlist:
            data[int(i[1]), int(i[0])] += 1.0
        return data

    def visualizeNodes(self, nc: NodeCollection) -> None:
        """Creates a visualization showing the number of neurons k per WTA circuit on the grid"""
        data = self.getNeuronFrequencyGrid()

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
        ax.set_xlabel("%d neurons total" % nc.spatial['network_size'])
        fig.tight_layout()
        plt.show()

    def plotNodes(self):
        nest.PlotLayer(self.nc, nodecolor='b')
        plt.show()


# Initialize new grid
grid = WTACircuitGrid()
# Form connections between nodes
grid.formConnections()

#grid.visualizeNodes(grid.nc)
#grid.plotNodes()
nest.PlotTargets(grid.nc[0], grid.nc)
plt.show()
