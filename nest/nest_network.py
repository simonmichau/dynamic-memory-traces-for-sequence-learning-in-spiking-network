import random
import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import os
import nest

from nest_utils import *
from pynestml.frontend.pynestml_frontend import generate_target, generate_nest_target
# from pynestml.frontend.pynestml_frontend import *


nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": 5116})

NEURON_MODEL = 'iaf_psc_exp_wta'
SYNAPSE_MODEL = 'stdp_stp'
RATE_CONN_SYN_MODEL = 'rate_connection_instantaneous'
regen = False

_NEURON_MODEL_NAME = NEURON_MODEL + "__with_" + SYNAPSE_MODEL
_SYNAPSE_MODEL_NAME = SYNAPSE_MODEL + "__with_" + NEURON_MODEL


class WTACircuit:

    def __init__(self, nc, pos):
        self.nc = nc
        self.pos = pos
        self.k = self.get_size()
        self.form_WTA()

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

    def form_WTA(self):
        """Connects all neurons within the same WTA via rate_connection_instantaneous connections"""
        for i in range(self.k):
            for j in range(self.k):
                if i != j:
                    nest.Connect(self.nc[i], self.nc[j], "all_to_all", {"synapse_model": RATE_CONN_SYN_MODEL})


class InputGenerator(object):
    """
    Contains all functionality to stimulate an assigned target network. Takes keywords:

    - **r_noise** [Hz]: Noise firing rate
    - **r_input** [Hz]: Input firing rate
    - **n_patterns**: Number of different patterns
    - **pattern_sequences**: List of pattern sequences
    - **pattern_mode**: Mode of how patterns are sampled from *pattern_sequences* during presentation. Can be either
      ``random_iterate`` or ``random_independent``
    - **p_switch**: Switching probability for *pattern_mode*
    - **t_pattern** [ms]: List containing the durations for all patterns
    - **t_noise_range** [ms]: Range from which the noise phase duration is randomly chosen from
    - **use_noise** [Bool]: States whether noise should be produced or not
    - **use_input** [Bool]: States whether inputs should be produced or not
    """
    def __init__(self, target_network, **kwds):
        # Number of input channels
        self.n = target_network.nInputs
        # Target network
        self.target_network = target_network
        # Poisson firing rate of the noise (in Hz)
        self.r_noise = kwds.get('r_noise', 2)
        # Input firing rate (in Hz)
        self.r_input = kwds.get('r_input', 5)
        # Number of patterns
        self.n_patterns = kwds.get('n_patterns', 3)
        # Pattern sequences (contains lists of pattern sequences; their presentation order is determined [elsewhere])
        self.pattern_sequences = kwds.get('pattern_sequences', [[0, 1], [2]])
        # Pattern mode (can be either 'random_independent' or 'random_iterate')
        self.pattern_mode = kwds.get('pattern_mode', 'random_iterate')
        # Switching probability for pattern picking
        self.p_switch = kwds.get('p_switch', 1.0)
        # Pattern durations
        self.t_pattern = kwds.get('t_pattern', [300.0] * self.n_patterns)
        # Range from which the noise phase duration is randomly chosen from (in ms)
        self.t_noise_range = kwds.get('t_noise_range', [100.0, 500.0])
        # Dictionary of stored patterns
        self.pattern_list = []
        # Spiketrain for all n input channels
        self.spiketrain = [[]] * self.n
        # Tuple storing the sequence index and the index of the current pattern
        self.current_pattern_index = [0, 0]

        self.use_noise = kwds.get('use_noise', True)
        # self.use_input = kwds.get('use_input', True)

        # Create noise
        if self.use_noise:
            self.generate_noise()

        # self.visualize_spiketrain(self.pattern_list[0], 500)
        # self.visualize_spiketrain(self.pattern_list[1], 500)

    def generate_noise(self) -> None:
        """Creates and connects poisson generators to target network to stimulate it with poisson noise."""
        # Create n poisson input channels with firing rate r_noise
        poisson_gens = nest.Create('poisson_generator', self.n, params={'rate': self.r_noise})
        # Create n parrot_neuron and connect one poisson generator to each of it
        parrots = nest.Create('parrot_neuron', self.n)
        nest.Connect(poisson_gens, parrots, 'one_to_one')
        # Connect parrots to target network
        conn_dict = {
            'rule': 'pairwise_bernoulli',
            'p': 1.0,
            'allow_autapses': False,
        }
        syn_dict = {"synapse_model": _SYNAPSE_MODEL_NAME,
                    'delay': 3.
                    }
        nest.Connect(parrots, self.target_network.get_node_collections(), conn_dict, syn_dict)
        # Update connection weights to random values
        randomize_outgoing_connections(parrots)

    def create_patterns(self) -> None:
        """Creates poisson patterns according to the InputGenerator's

        - number of patterns **n_patterns**,
        - pattern durations **t_pattern** (in ms),
        - pattern firing rate **r_input** (in Hz)

        and stores them in **pattern_list**."""
        self.pattern_list = []
        for i in range(self.n_patterns):
            pattern = []
            for j in range(self.n):
                pattern.append(generate_poisson_spiketrain(self.t_pattern[i], self.r_input))
            self.pattern_list.append(pattern)

    def generate_input(self, duration, t_origin=0.0, force_refresh_patterns=False):
        """Generates Input for a given duration. Needs to be run for every simulation

        - duration: duration of input (in ms)
        -
        """
        # Create new patterns if none have been created yet, or it is demanded explicitly
        if not self.pattern_list or force_refresh_patterns:
            self.create_patterns()

        # create n spike_generators
        spike_generators = nest.Create('spike_generator', self.n, params={'allow_offgrid_times': True,
                                                                          'origin': t_origin})

            # Connect spike generators to target network
            conn_dict = {'rule': 'pairwise_bernoulli',
                         'allow_autapses': False,
                         'p': 1.0}
            syn_dict = {"synapse_model": _SYNAPSE_MODEL_NAME,
                        'delay': 3.
                        }
            nest.Connect(self.spike_generators, self.target_network.get_node_collections(), conn_dict,
                         syn_dict)

            # Randomize connection weights
            randomize_outgoing_connections(self.spike_generators)

        # generate a list of spiketrains that alternate between noise phase and pattern presentation phase
        t = 0
        spiketrain_list = [[]] * self.n  # list to store the spiketrain of each input channel
        current_pattern_id = self.pattern_sequences[self.current_pattern_index[0]][self.current_pattern_index[1]]
        while t < duration:
            # Randomly draw the duration of the noise phase
            t_noise_phase = self.t_noise_range[0] + np.random.rand()*(self.t_noise_range[1]-self.t_noise_range[0])

            # append pattern spike times to spiketrain list
            for i in range(self.n):  # iterate over input channels
                st = np.add(t+t_noise_phase, self.pattern_list[current_pattern_id][i])
                spiketrain_list[i] = spiketrain_list[i] + st.tolist()
            t += t_noise_phase + self.t_pattern[current_pattern_id]

            # Update the pattern to present next round
            current_pattern_id = self.get_next_pattern_id()

        # cutoff values over t=origin+duration
        t_threshold = duration
        for i in range(len(spiketrain_list)):
            threshold_index = np.searchsorted(spiketrain_list[i], t_threshold)
            spiketrain_list[i] = spiketrain_list[i][0: threshold_index]

        self.spiketrain = spiketrain_list

        # Assign spiketrain_list to spike_generators
        for i in range(self.n):
            spike_generators[i].set({'spike_times': spiketrain_list[i]})

        # Connect spike generators to target network
        conn_dict = {'rule': 'pairwise_bernoulli',
                     'allow_autapses': False,
                     'p': 1.0}
        nest.Connect(spike_generators, self.target_network.get_node_collections(), conn_dict)

        # Randomize connection weights
        randomize_outgoing_connections(spike_generators)

    def get_next_pattern_id(self) -> int:
        # if sequence is not over just progress to next id in sequence
        if not self.current_pattern_index[1] + 1 >= len(self.pattern_sequences[self.current_pattern_index[0]]):
            self.current_pattern_index[1] += 1
        else:
            # if sequence is over pick new sequence from pattern_sequences using rules
            if self.pattern_mode == 'random_independent':
                print("Error: random_independent switching mode not implemented yet")  # TODO
            elif self.pattern_mode == 'random_iterate':
                # with probability p_switch move on to the next sequence/repeat the current sequence with 1-p_switch
                if np.random.rand() <= self.p_switch:
                    self.current_pattern_index[0] = (self.current_pattern_index[0]+1) % len(self.pattern_sequences)
                self.current_pattern_index[1] = 0  # reset index to beginning of sequence
        return self.pattern_sequences[self.current_pattern_index[0]][self.current_pattern_index[1]]

    def get_patterns(self):
        return self.pattern_list

    def set_patterns(self, patterns):
        self.pattern_list = []
        self.pattern_list += patterns

    def visualize_spiketrain(self, st):
        """Visualizes a given spiketrain"""
        fig = plt.figure()
        fig, ax = plt.subplots()
        for i in range(len(st)):
            ax.scatter(st[i], [i]*len(st[i]), color=(i/(len(st)), 0.0, i/(len(st))))
            # ax.plot(st[i], [i]*len(st[i]), ".", color='orange')
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("Input channels")
        ax.axis([0, np.amax(st)[0]*1.5, -1, self.n])


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
        self.circuits = []
        # Create WTA circuits
        self.create_grid()
        # Establish interneuron connections
        self.form_connections()

        self.multimeter = None
        self.spikerecorder = None

        # ADMINISTRATIVE VARIABLES
        self.save_figures = kwds.get('save_figures', False)
        self.show_figures = kwds.get('show_figures', True)

    def get_circuit_grid(self) -> np.ndarray:
        """Returns a (nxm) array containing the neuron frequencies per grid point"""
        data = np.zeros((self.m, self.n))
        for circuit in self.circuits:
            data[circuit.get_pos()[1], circuit.get_pos()[0]] = circuit.get_size()
        return data

    def get_node_collections(self, slice_min=None, slice_max=None):
        """Return a slice of **self.circuits** as a **NodeCollection**"""
        if slice_min is None:
            slice_min = 0
        if slice_max is None:
            slice_max = len(self.circuits) + 1
        s = slice(slice_min, slice_max)

        id_list = []
        for circuit in self.circuits[s]:
            id_list += circuit.get_node_collection().get()['global_id']
        return nest.NodeCollection(id_list)

    def get_pos_by_id(self, node_id: int) -> Optional[tuple]:
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
                nc = nest.Create(_NEURON_MODEL_NAME, K, {'tau_m': 20.0})
                circuit_list.append(WTACircuit(nc, (n, m)))
        self.circuits = circuit_list
        return circuit_list

    def form_connections(self) -> None:
        """Connect every WTA circuit """
        conn_dict = {'rule': 'pairwise_bernoulli',
                     'p': 1.0,
                     'allow_autapses': False}
        syn_dict = {"synapse_model": _SYNAPSE_MODEL_NAME,
                    'delay': 3.
                    }

        # Iterate over each WTACircuit object and establish connections to every other population with p(d)
        for i in range(len(self.circuits)):
            self.circuits[i].get_pos()
            for j in range(len(self.circuits)):
                if i != j:
                    d = math.sqrt((self.circuits[i].get_x()-self.circuits[j].get_x())**2
                                  + (self.circuits[i].get_y()-self.circuits[j].get_y())**2)
                    conn_dict['p'] = self.lam * math.exp(-self.lam * d)
                    nest.Connect(self.circuits[i].get_node_collection(), self.circuits[j].get_node_collection(),
                                 conn_dict, syn_dict)

        # Randomize weights of each WTA circuit
        for i in range(len(self.circuits)):
            randomize_outgoing_connections(self.circuits[i].get_node_collection())

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

    def visualize_connections(self, nc) -> None:
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


if __name__ == '__main__':
    # Setup nest
    nest.resolution = 1.
    generate_nest_code(NEURON_MODEL, SYNAPSE_MODEL, regen=regen)
    print(_SYNAPSE_MODEL_NAME, " installed: ", _SYNAPSE_MODEL_NAME in nest.synapse_models)
    print(_NEURON_MODEL_NAME, " installed: ", _NEURON_MODEL_NAME in nest.node_models)
    nest.print_time = True

    # Initialize weight recorder
    weight_recorder = nest.Create('weight_recorder')
    nest.CopyModel(_SYNAPSE_MODEL_NAME, "synapse_rec", {"weight_recorder": weight_recorder})
    _SYNAPSE_MODEL_NAME = "synapse_rec"

    # Initialize Network
    grid = Network()
    # grid.visualize_circuits()
    # grid.visualize_circuits_3d()
    # grid.visualize_connections(grid.get_node_collections(1, 2))
    # grid.get_node_collections(1, 5)

    # Initialize Input Generator
    inp = InputGenerator(grid, n_patterns=3, pattern_sequences=[[0]], use_noise=True)

    # measure_node_collection(grid.get_node_collections(1, 5), t_sim=100000.0)
    # measure_node_collection(grid.get_node_collections()[0], t_sim=5000.0)
    # measure_node_collection(grid.get_node_collections()[0], inp, t_sim=5000.0)
    # measure_node_collection(grid.get_node_collections()[0:2], t_sim=5000.0)
    measure_node_collection(grid.get_node_collections(0, 1), inp, t_sim=5000.0)
    measure_node_collection(grid.get_node_collections(0, 1), inp, t_sim=5000.0)
    measure_node_collection(grid.get_node_collections(0, 1), inp, t_sim=50000.0)
    measure_node_collection(grid.get_node_collections(0, 1), inp, t_sim=5000.0)
    measure_node_collection(grid.get_node_collections(0, 1), inp, t_sim=5000.0)


    # (TODO) version of visualize_connections where the percentage of connected neurons is given instead of total amount
    # TODO: implement lateral inhibition
