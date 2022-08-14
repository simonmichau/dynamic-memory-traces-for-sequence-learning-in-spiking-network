import random
import math

import numpy as np
import matplotlib.pyplot as plt
import nest
from nest.lib.hl_api_types import *


nest.ResetKernel()


class WTACircuit:
    def __init__(self, nc: NodeCollection, pos):
        self.nc = nc
        self.pos = pos
        self.k = self.get_size()

    def get_pos(self):
        """Returns the (x, y) position of the WTA circuit"""
        return self.pos

    def get_x(self):
        """Returns the x coordinate of the WTA circuit"""
        return self.pos[0]

    def get_y(self):
        """Returns the y coordinate of the WTA circuit"""
        return self.pos[1]

    def get_node_collection(self):
        """Returns the NodeCollection nc"""
        return self.nc

    def get_size(self):
        """Returns the size of the NodeCollection nc"""
        return len(self.nc.get('global_id'))


class InputGenerator(object):
    def __init__(self, target_network, **kwds):
        # Number of input channels
        self.n = target_network.nInputs
        # Poisson firing rate of the noise (in Hz)
        self.rNoise = kwds.get('rNoise', 2)
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
        # Create noise
        self.noise(target_network)

        self.generate_patterns(1, 50, 5)  # test

    def noise(self, target_network):
        # Create n poisson input channels with 2Hz firing rate
        poisson_gens = nest.Create('poisson_generator', self.n, params={'rate': self.rNoise})
        # Create parrot_neuron and connect poisson generators to it
        parrots = nest.Create('parrot_neuron', self.n)
        nest.Connect(poisson_gens, parrots, 'one_to_one')
        # Connect parrots to target network with random weights
        conn_dict = {
            'rule': 'pairwise_bernoulli',
            'allow_autapses': False,
            'p': 1.0
        }
        syn_dict = {"weight": -nest.random.uniform()}  # TODO: set init weight
        nest.Connect(parrots, target_network.get_node_collections(), conn_dict, syn_dict)

        print(nest.GetConnections(parrots))

    def configure_spikes(self, spike_duration, frequency, t_start):
        # Create n spikegenerators with 5Hz firing rate
        pass

    def generate_patterns(self, nPatterns, tPatterns, rin):
        # TODO: Create an array with one element for each pattern, this element contains all the spike_times
        #  lists of the spike_generators
        # rin: input firing rate |

        # TODO: uniformly randomly distribute the expected number of spikes over the duration of the pattern
        expected_spikes = int(rin*10e-3 * tPatterns)  # expected number of spikes per spike generator
        # generate list of possible spike times in duration of pattern [tic_list]
        tic_list = list(range(1, tPatterns+1))
        # sample randomly k elements from tic_list without replacement
        spikes = random.sample(tic_list, k=expected_spikes)


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
        # Number of external input channels
        self.nInputs = kwds.get('nInputs', 50)
        # parameter of exponential distance distribution
        self.lam = kwds.get('lambda', 0.088)
        # simulation time (in ms)
        self.t_sim = kwds.get('t_sim', 2000.0)

        # List containing all WTA circuits
        self.circuits = self.create_grid()

        # ADMINISTRATIVE VARIABLES
        self.save_figures = kwds.get('save_figures', False)
        self.show_figures = kwds.get('show_figures', True)

    def get_circuit_grid(self) -> np.ndarray:
        """Returns a (nxm) array containing the neuron frequencies per grid point"""
        data = np.zeros((self.m, self.n))
        for circuit in self.circuits:
            data[circuit.get_pos()[1], circuit.get_pos()[0]] = circuit.get_size()
        return data

    def get_node_collections(self, s=None) -> NodeCollection | None:
        """Return a slice of **self.circuits** as a **NodeCollection**"""
        if s is None:  # Set default parameter for s
            s = slice(0, len(self.circuits) + 1)
        id_list = []
        for circuit in self.circuits[s]:
            id_list += circuit.get_node_collection().get()['global_id']
        return nest.NodeCollection(id_list)

    def get_pos_by_id(self, node_id: int) -> tuple | None:
        """Returns the position of the WTA circuit which contains the node with the given ID"""
        for i in self.circuits:
            if node_id in i.nc.get()['global_id']:
                return i.get_pos()

    def create_grid(self) -> list:
        """
        Create a **WTACircuit** object for every point on the (nxm) grid and returns all those objects in a list

        - **K**: number of neurons in a WTA circuit, randomly drawn with lower and upper bound [k_min, k_max]
        """
        circuit_list = []
        for m in range(self.m):
            for n in range(self.n):
                K = random.randint(self.k_min, self.k_max)
                nc = nest.Create('iaf_psc_exp', K, params={'I_e': 0.0,  # 0.0
                                                           'tau_syn_ex': self.tau_rise,
                                                           'tau_m': self.tau_decay
                                                           })
                circuit_list.append(WTACircuit(nc, (n, m)))
        return circuit_list

    def form_connections(self) -> None:
        """Connect every WTA circuit """
        conn_dict = {
            'rule': 'pairwise_bernoulli',
            'allow_autapses': False,
            'p': 1.0
        }
        syn_dict = {"weight": np.log(np.random.rand())}  # TODO: implement synapse model with STP and STDP

        # Iterate over each WTACircuit object and establish connections to every other population with p(d)
        for i in range(len(self.circuits)):
            self.circuits[i].get_pos()
            for j in range(len(self.circuits)):
                if i != j:
                    d = math.sqrt((self.circuits[i].get_x()-self.circuits[j].get_x())**2
                                  + (self.circuits[i].get_y()-self.circuits[j].get_y())**2)
                    conn_dict['p'] = self.lam * math.exp(-self.lam * d)
                    nest.Connect(self.circuits[i].get_node_collection(), self.circuits[j].get_node_collection(),
                                 conn_dict, {"weight": np.log(np.random.rand())})

    def visualize_circuits(self) -> None:
        """Creates a **pcolormesh** visualizing the number of neurons k per WTA circuit on the grid"""
        data = self.get_circuit_grid()

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

    def visualize_circuits_3d(self) -> None:
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        x = []
        y = []
        z = []
        data = self.get_circuit_grid()
        for i in range(len(self.get_circuit_grid())):
            for j in range(len(self.get_circuit_grid()[i])):
                x.append(j)
                y.append(i)
                z.append(data[i][j])

        # Trimesh
        ax.plot_trisurf(x, y, z, color='blue')
        # Scatterplot
        ax.scatter3D(x, y, z, c=z, cmap='cividis')
        # Select Viewpoint
        ax.view_init(30, -90)
        if self.save_figures:
            plt.savefig("grid_visualization_3d.png")
        if self.show_figures:
            plt.show()

    def visualize_connections(self, nc: NodeCollection) -> None:
        """Visualizes all the outgoing connections from some **NodeCollection** nc as a **pcolormesh**"""
        # Get List of X and Y coordinates of each target's position
        X = []
        Y = []
        for target_id in nest.GetConnections(nc).target:
            target_pos = self.get_pos_by_id(target_id)
            X.append(target_pos[0])
            Y.append(target_pos[1])

        # Get positions of each source node and save to list
        source_pos_list = []
        for source_node in nc:
            source_id = source_node.get()['global_id']
            source_pos = self.get_pos_by_id(source_id)
            source_pos_list.append(source_pos)

        # create position frequency array stating the number of occurrences of each position in the target list
        data = np.zeros((self.m, self.n))
        for i in range(len(X)):
            data[Y[i], X[i]] += 1

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


def measure_node_collection(nc: NodeCollection, **kwds) -> None:
    """Simulates given NodeCollection for t_sim and plots the recorded spikes and membrane potential"""
    multimeter = nest.Create('multimeter')
    multimeter.set(record_from=['V_m'])
    spikerecorder = nest.Create('spike_recorder')
    nest.Connect(multimeter, nc)
    nest.Connect(nc, spikerecorder)

    nest.Simulate(kwds.get('t_sim', 2000.0))

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


if __name__ == '__main__':
    grid = Network()
    grid.visualize_circuits()
    # grid.visualize_circuits_3d()
    grid.form_connections()
    # grid.visualize_connections(grid.get_node_collections(slice(1, 5)))
    # grid.get_node_collections(slice(1, 3))

    inp = InputGenerator(grid)
    # print(inp.pop.get())
    # grid.connect_input(inp)
    # measure_node_collection(inp.pop)

    # measure_node_collection(grid.get_node_collections(slice(1, 5)), t_sim=100000.0)
    measure_node_collection(grid.get_node_collections(), t_sim=1000.0)
    measure_node_collection(grid.get_node_collections(), t_sim=1000.0)

    # (TODO) version of visualize_connections where the percentage of connected neurons is given instead of total amount
    # TODO: implement lateral inhibition
