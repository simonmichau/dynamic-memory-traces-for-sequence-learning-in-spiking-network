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

    def getX(self):
        """Returns the x coordinate of the WTA circuit"""
        return self.pos[0]

    def getY(self):
        """Returns the y coordinate of the WTA circuit"""
        return self.pos[1]

    def getNodeCollection(self):
        """Returns the NodeCollection nc"""
        return self.nc

    def getSize(self):
        """Returns the size of the NodeCollection nc"""
        return len(self.nc.get('global_id'))


class InputPopulation(object):
    def __init__(self, n, **kwds):
        # Number of neurons in input population
        self.n = n
        # Rise and decay time constant for EPSP (in ms)
        self.tau_rise = kwds.get('tau_rise', 2)
        self.tau_decay = kwds.get('tau_decay', 20)
        # Resting membrane potential (in mV)
        self.E_L = kwds.get('E_L', -70.0)
        # Capacity of the membrane (in pF)
        self.C_m = kwds.get('C_m', 250.0)
        # Duration of refractory period (V_m = V_reset) (in ms)
        self.t_ref = kwds.get('t_ref', 2.0)
        # Membrane potential (in mV)
        self.V_m = kwds.get('V_m', -70.0)
        # Spike threshold (in mV)
        self.V_th = kwds.get('V_th', -55.0)
        # Reset membrane potential after a spike (in mV)
        self.V_reset = kwds.get('V_reset', -70.0)
        # Constant input current
        self.I_e = kwds.get('I_e', 0.0)

        # Actual input population
        self.pop = nest.Create('iaf_psc_exp', self.n, params={'E_L': self.E_L,
                                                              'C_m': self.C_m,
                                                              'tau_syn_ex': self.tau_rise,
                                                              'tau_m': self.tau_decay,
                                                              't_ref': self.t_ref,
                                                              'V_m': self.V_m,
                                                              'V_th': self.V_th,
                                                              'V_reset': self.V_reset,
                                                              'I_e': self.I_e
                                                              })


class Network(object):
    def __init__(self, **kwds):
        # FUNCTIONAL VARIABLES
        # Rise and decay time constant for EPSP (in ms)
        self.tau_rise = kwds.get('tau_rise', 2)
        self.tau_decay = kwds.get('tau_decay', 20)
        # Dimensions of the grid of WTA circuits
        self.grid_shape = kwds.get('grid_shape', (10, 5))
        self.n, self.m = self.grid_shape
        # Upper and lower bound for randomly drawn number k of neurons in each WTA circuit
        self.k_min = kwds.get('k_min', 2)
        self.k_max = kwds.get('k_max', 10)
        # Number of external inputs
        self.nInputs = kwds.get('nInputs', 50)
        # parameter of exponential distance distribution
        self.lam = kwds.get('lambda', 0.088)
        # simulation time (in ms)
        self.t_sim = kwds.get('t_sim', 2000.0)

        # List containing all WTA circuits
        self.circuits = self.createNodes()

        # ADMINISTRATIVE VARIABLES
        self.save_figures = kwds.get('save_figures', False)
        self.show_figures = kwds.get('show_figures', True)

    def createNodes(self) -> list:
        """
        Returns a list of WTACircuit objects of size K for each coordinate on the (nxm) grid

        - **K**: number of neurons in a WTA circuit, randomly drawn with lower and upper bound [k_min, k_max]
        """
        circuit_list = []
        for m in range(self.m):
            for n in range(self.n):
                K = random.randint(self.k_min, self.k_max)
                nc = nest.Create('iaf_psc_exp', K, params={'I_e': 188.0,  # 0.0
                                                           'tau_syn_ex': self.tau_rise,
                                                           'tau_m': self.tau_decay
                                                           })
                circuit_list.append(WTACircuit(nc, (n, m)))
        return circuit_list

    def getCircuitSizeGrid(self) -> np.ndarray:
        """Returns a (nxm) array containing the neuron frequencies per grid point"""
        data = np.zeros((self.m, self.n))
        for circuit in self.circuits:
            data[circuit.getPos()[1], circuit.getPos()[0]] = circuit.getSize()
        return data

    def getNodeCollections(self, *index):
        """Return an element or slice of **self.circuits**"""
        if len(index) == 1:
            return self.circuits[index].getNodeCollection()
        elif len(index) >= 2:
            id_list = []
            for circuit in self.circuits[index[0]:index[1]]:
                id_list += circuit.getNodeCollection().get()['global_id']
            return nest.NodeCollection(id_list)
        else:
            return None

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

        if self.save_figures:
            plt.savefig("grid_visualization.png")
        if self.show_figures:
            plt.show()

    def getPosByID(self, node_id: int) -> tuple | None:
        """Returns the position of the WTA circuit which contains the node with the given ID"""
        for i in self.circuits:
            if node_id in i.nc.get()['global_id']:
                return i.getPos()

    def visualizeConnections(self, nc: NodeCollection):
        """Visualizes all the outgoing connections from some node"""
        # Get List of X and Y coordinates of each target's position
        X = []
        Y = []
        for target_id in nest.GetConnections(nc).target:
            target_pos = self.getPosByID(target_id)
            X.append(target_pos[0])
            Y.append(target_pos[1])

        # Get positions of each source node and save to list
        source_pos_list = []
        for source_node in nc:
            source_id = source_node.get()['global_id']
            source_pos = self.getPosByID(source_id)
            source_pos_list.append(source_pos)

        # create position frequency array stating the number of occurrences of each position in the target list
        data = np.zeros((self.m, self.n))
        for i in range(len(X)):
            data[Y[i], X[i]] += 1

        ########################################################
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
                if (i, j) not in source_pos_list:
                    text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")
                else:
                    text = ax.text(j, i, "x", ha="center", va="center", color="w")
        ax.set_title("Outgoing connections from node(s) %s" % str(nc.get()['global_id']))
        ax.set_xlabel("%d outgoing connections total" % np.sum(data))
        fig.tight_layout()

        if self.save_figures:
            plt.savefig("conn_visualization.png")
        if self.show_figures:
            plt.show()

    def formConnections(self):
        """Connect every WTA circuit """
        conn_dict = {
            'rule': 'pairwise_bernoulli',
            'allow_autapses': False,
            'p': 1.0
        }
        syn_dict = {}  # TODO: implement synapse model with STP and STDP

        # Iterate over each WTACircuit object and establish connections to every other population with p(d)
        for i in range(len(self.circuits)):
            self.circuits[i].getPos()
            for j in range(len(self.circuits)):
                if i != j:
                    d = math.sqrt((self.circuits[i].getX()-self.circuits[j].getX())**2
                                  + (self.circuits[i].getY()-self.circuits[j].getY())**2)
                    conn_dict['p'] = self.lam * math.exp(-self.lam * d)
                    nest.Connect(self.circuits[i].getNodeCollection(), self.circuits[j].getNodeCollection(), conn_dict)

    def connectInput(self, inp_pop: InputPopulation) -> None:
        """Connects an **InputPopulation** to this **Network**"""
        # TODO: implement incoming connections
        # also TODO: find out by what metrics these connections are formed
        nest.Connect(inp_pop.pop, self.circuits[0].getNodeCollection())
        #print(nest.GetConnections(inp_pop.pop))


def measureNodeCollection(nc: NodeCollection) -> None:
    """Simulates given NodeCollection for t_sim and plots the recorded spikes and membrane potential"""
    multimeter = nest.Create('multimeter')
    multimeter.set(record_from=['V_m'])
    spikerecorder = nest.Create('spike_recorder')
    nest.Connect(multimeter, nc)
    nest.Connect(nc, spikerecorder)

    nest.Simulate(2000.0)

    dmm = multimeter.get()
    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]
    # plt.figure(1)
    plt.plot(ts, Vms)
    dSD = spikerecorder.get("events")
    evs = dSD["senders"]
    ts = dSD["times"]
    # plt.figure(2)
    plt.plot(ts, evs, ".")
    plt.show()


grid = Network()
grid.visualizeCircuits()
grid.formConnections()
# grid.visualizeConnections(grid.circuits[0].getNodeCollection())

nest.Simulate(2000.0)

inpPop = InputPopulation(10)
grid.connectInput(inpPop)

# measureNodeCollection(grid.circuits[0].getNodeCollection())
# measureNodeCollection(inpPop.pop)


# (TODO): version of visualizeConnections where the percentage of connected neurons is given instead of total amount
# TODO: implement lateral inhibition
