"""

"""


import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
import math
from typing import Optional

import numpy as np
import os
import sys
import nest

sys.path.append('../')
import shared_params

import nest_utils as utils
# from nest_utils import *
# from pynestml.frontend.pynestml_frontend import *


nest.ResetKernel()
nest.SetKernelStatus({"rng_seed": 1337})
np.random.seed(1337)


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
        try:
            return len(self.nc.get('global_id'))
        except TypeError:
            return 1

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
        self.n = target_network.n_inputs
        # Target network
        self.target_network = target_network
        # Poisson firing rate of the noise during pattern presentation (in Hz)
        self.r_noise_pattern = kwds.get('r_noise_pattern', 4)
        # Standard poisson firing rate of the noise (in Hz)
        self.r_noise = kwds.get('r_noise', 5)  # poisson noise rate during pattern presentation
        # Input firing rate (in Hz)
        self.r_input = kwds.get('r_input', 3)
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

        # Parameters used for `get_order`
        # List of time points where the phase changes (noise, patternA, patternB, ...)
        self.phase_times = [0]  # TODO
        # List of length [duration] containing the index of the pattern for each time step (or -1 if noise)
        self.pattern_trace = []  # TODO
        self.next_pattern_length = [self.t_pattern[0]]  # TODO

        # NodeCollection of spike_generators
        self.spike_generators = None
        # NodeCollection of inhomogeneous_poisson_generators
        self.poisson_generators = None
        # NodeCollection of parrot_neurons used for poisson noise and input patterns
        self.parrots = nest.Create('parrot_neuron', self.n)
        self.connect_parrots()  # connect parrots to network with stdp and w/o stp

        # Spikerecorder for poisson noise
        self.noiserecorder = None
        self.spike_recorder = None

        self.use_noise = kwds.get('use_noise', True)
        # self.use_input = kwds.get('use_input', True)

        # Create noise
        if self.use_noise:
            self.generate_noise()

        # self.visualize_spiketrain(self.pattern_list[0], 500)
        # self.visualize_spiketrain(self.pattern_list[1], 500)

    def connect_parrots(self):
        # Connect parrots to target network
        conn_dict = {
            'rule': 'pairwise_bernoulli',
            'p': 1.0,
            'allow_autapses': False,
        }
        syn_dict = {"synapse_model": _SYNAPSE_MODEL_NAME,
                    'delay': 1.,
                    'U': 0.5,
                    'u': 0.5,
                    'use_stp': 0  # TODO for some reason, input synapses are not dynamic.
                    }
        nest.Connect(self.parrots, self.target_network.get_node_collections(), conn_dict, syn_dict)

    def generate_noise(self) -> None:
        """Creates and connects poisson generators to target network to stimulate it with poisson noise."""
        # Create n poisson input channels with firing rate r_noise
        #poisson_gens = nest.Create('poisson_generator', self.n, params={'rate': self.r_noise})
        assert False
        self.poisson_generators = nest.Create('inhomogeneous_poisson_generator', self.n)
        # Connect one poisson generator to each parrot neuron
        nest.Connect(self.poisson_generators, self.parrots, 'one_to_one')

        # Update connection weights to random values
        utils.randomize_outgoing_connections(self.parrots)

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
                pattern.append(utils.generate_poisson_spiketrain(self.t_pattern[i], self.r_input))
            self.pattern_list.append(pattern)

    def generate_input(self, duration, t_origin=0.0, force_refresh_patterns=False):
        """Generates Input for a given duration. Needs to be run for every simulation

        - duration: duration of input (in ms)
        -
        """
        # Create new patterns if none have been created yet, or it is demanded explicitly
        if not self.pattern_list or force_refresh_patterns:
            self.create_patterns()

        if self.spike_generators is None:
            # create n spike_generators if none exist yet
            self.spike_generators = nest.Create('spike_generator', self.n, params={'allow_offgrid_times': False,
                                                                                   'origin': t_origin})

            # Connect spike generators to target network
            conn_dict = {'rule': 'pairwise_bernoulli',
                         'allow_autapses': False,
                         'p': 1.0}
            syn_dict = {
                "synapse_model": _SYNAPSE_MODEL_NAME,
                'delay': 1.,
                'U': 0.5,
                'u': 0.5,
                'use_stp': 0  # TODO for some reason, input synapses are not dynamic.
                # 'use_stp': float(self.target_network.use_stp)  # TODO double check above assumption?
            }
            nest.Connect(self.spike_generators, self.target_network.get_node_collections(), conn_dict,
                        syn_dict)
            # nest.Connect(self.spike_generators, self.parrots, 'one_to_one')

            # Randomize connection weights
            # utils.randomize_outgoing_connections(self.spike_generators)

        noise_rate_times = []
        noise_rate_values = []
        # generate a list of spiketrains that alternate between noise phase and pattern presentation phase
        t = nest.biological_time
        spiketrain_list = [[]] * self.n  # list to store the spiketrain of each input channel
        current_pattern_id = self.pattern_sequences[self.current_pattern_index[0]][self.current_pattern_index[1]]
        while t < nest.biological_time + duration:
            # Randomly draw the duration of the noise phase
            t_noise_phase = self.t_noise_range[0] + np.random.rand()*(self.t_noise_range[1]-self.t_noise_range[0])
            t_noise_phase = np.round(t_noise_phase, decimals=0)

            # Get noise and pattern times for poisson gens
            noise_rate_times += [t + nest.resolution]
            noise_rate_values += [self.r_noise]  # noise rate during noise phase
            noise_rate_times += [t+t_noise_phase]
            noise_rate_values += [self.r_noise_pattern]  # noise rate during pattern presentation

            # append pattern spike times to spiketrain list
            for i in range(self.n):  # iterate over input channels
                st = np.add(t+t_noise_phase, self.pattern_list[current_pattern_id][i])
                spiketrain_list[i] = spiketrain_list[i] + st.tolist()

            # Append phase times (for get_order)
            self.phase_times += [int(t+t_noise_phase), int(t+t_noise_phase + self.t_pattern[current_pattern_id])]
            # Append next pattern length (for get_order)
            self.next_pattern_length += [int(self.t_pattern[current_pattern_id]), int(self.t_pattern[current_pattern_id])]
            # Append pattern trace (for get_order)
            self.pattern_trace += [-1]*int(t_noise_phase)  # append noise id
            self.pattern_trace += [current_pattern_id]*int(self.t_pattern[current_pattern_id])  # append input pattern id

            t += t_noise_phase + self.t_pattern[current_pattern_id]

            # Update the pattern to present next round
            current_pattern_id = self.get_next_pattern_id()

        # cutoff values over t=origin+duration
        #t_threshold = nest.biological_time + duration
        #for i in range(len(spiketrain_list)):
        #    threshold_index = np.searchsorted(spiketrain_list[i], t_threshold)
        #    spiketrain_list[i] = spiketrain_list[i][0: threshold_index]

        #self.spiketrain = spiketrain_list
        for i in range(len(self.spiketrain)):
            self.spiketrain[i] = self.spiketrain[i] + spiketrain_list[i]
            self.spiketrain[i] = np.unique(self.spiketrain[i]).tolist()  # avoid redundant spike times

        # Set noise and pattern times for poisson gens
        if self.use_noise:
            self.poisson_generators.set({'rate_times': noise_rate_times, 'rate_values': noise_rate_values})

        # Assign spiketrain_list to spike_generators
        for i in range(self.n):
            # past_spikes = self.spike_generators[i].get('spike_times')
            # new_spiketrain = past_spikes.tolist().append(spiketrain_list[i])
            # self.spike_generators[i].spike_times = self.spiketrain[i]  # TODO fix non descending order issue
            #self.spike_generators[i].spike_times.append(spiketrain_list[i])  # TODO revert this
            #self.spike_generators[i].set({'spike_times': [1146., 1255., 1646., 1769.]})
            self.spike_generators[i].set({'spike_times': shared_params.input_spikes[i]})

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
        # Dimensions of the grid of WTA circuits
        self.grid_shape = kwds.get('grid_shape', (10, 5))
        self.n, self.m = self.grid_shape
        # Upper and lower bound for randomly drawn number k of neurons in each WTA circuit
        self.k_min = kwds.get('k_min', 2)
        self.k_max = kwds.get('k_max', 10)
        # Number of external input channels
        self.n_inputs = kwds.get('n_inputs', 50)
        # parameter lambda of exponential distance distribution
        self.lam = kwds.get('lam', 0.088)
        # List containing all WTA circuits
        self.circuits = []
        # NodeCollection containing all neurons of the grid
        self.neuron_collection = None
        # ADMINISTRATIVE VARIABLES
        self.save_figures = kwds.get('save_figures', False)
        self.show_figures = kwds.get('show_figures', True)
        #
        self.use_stp = shared_params.use_stp_rec
        self.use_stdp = True
        # Create WTA circuits
        self.create_grid()
        # Establish interneuron connections
        self.form_connections()

        self.weight_recorder = list()
        self.epsp_recorder = list()
        self.weight_recorder_manual = list()

        self.multimeter = None
        self.spikerecorder = None

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
            try:
                id_list += circuit.get_node_collection().get()['global_id']
            except TypeError:
                id_list += [circuit.get_node_collection().get()['global_id']]
        return nest.NodeCollection(id_list)

    def get_pos_by_id(self, node_id: int) -> Optional[tuple]:
        """Returns the position of the WTA circuit which contains the node with the given ID"""
        for i in self.circuits:
            if isinstance(i.nc.get()['global_id'], int):
                if node_id == i.nc.get()['global_id']:
                    return i.get_pos()
            else:
                if node_id in i.nc.get()['global_id']:
                    return i.get_pos()

    def get_wta_by_id(self, node_id: int) -> Optional[tuple]:
        """Returns the position of the WTA circuit which contains the node with the given ID"""
        for i in self.circuits:
            if node_id in i.nc.get()['global_id']:
                return i
        raise RuntimeError(f"Neuron {node_id} not in any circuit?")

    def refresh_neurons(self):
        """Refreshes self.neurons based on self.circuits"""
        self.neuron_collection = self.get_node_collections()

    def create_grid(self) -> list:
        """
        Create a **WTACircuit** object for every point on the (nxm) grid and returns all those objects in a list

        - **K**: number of neurons in a WTA circuit, randomly drawn with lower and upper bound [k_min, k_max]
        """
        circuit_list = []
        for m in range(self.m):
            for n in range(self.n):
                K = np.random.randint(self.k_min, self.k_max + 1)
                nc = nest.Create(_NEURON_MODEL_NAME, K,
                                 {'tau_m': 20.0, 'use_variance_tracking': int(shared_params.use_variance_tracking),
                                  'use_stdp': int(self.use_stdp), 'rate_fraction': 1./K})

                # TODO this is only for toy model - disable for normal conditions
                if shared_params.use_fixed_spike_times:
                    for neuron in range(K):
                        nc[neuron].set({'fixed_spiketimes': np.array(shared_params.output_spikes[m][neuron]).astype(float)})

                circuit_list.append(WTACircuit(nc, (n, m)))
                print(f"Position and size of WTA circuit: ({n}, {m}) - {K}")
        self.circuits = circuit_list
        self.refresh_neurons()
        return circuit_list

    def form_connections(self) -> None:
        """Connect every WTA circuit """
        conn_dict = {'rule': 'pairwise_bernoulli',
                     'p': 1.0,
                     'allow_autapses': False}
        syn_dict = {
            "synapse_model": _SYNAPSE_MODEL_NAME,
            'delay': 1.,
            'use_stp': float(self.use_stp)
        }

        # Iterate over each WTACircuit object and establish connections to every other population with p(d)
        for i in range(len(self.circuits)):
            self.circuits[i].get_pos()
            for j in range(len(self.circuits)):
                if i != j:
                    d = math.sqrt((self.circuits[i].get_x()-self.circuits[j].get_x())**2
                                  + (self.circuits[i].get_y()-self.circuits[j].get_y())**2)
                    # conn_dict['p'] = self.lam * math.exp(-self.lam * d)  # TODO revert
                    conn_dict['p'] = 1.
                    # nest.Connect(self.circuits[i].get_node_collection(), self.circuits[j].get_node_collection(),
                    #              conn_dict, syn_dict)

                    conns = nest.Connect(self.circuits[i].get_node_collection(), self.circuits[j].get_node_collection(),
                                         conn_dict, syn_dict, return_synapsecollection=True)

                    U_mean = 0.5
                    # tau_d_mean = 0.11
                    # tau_f_mean = 0.005
                    tau_d_mean = 110.
                    tau_f_mean = 5.

                    Us = U_mean + U_mean / 2 * np.random.randn(len(conns)) * 0
                    tau_ds = tau_d_mean + tau_d_mean / 2 * np.random.randn(len(conns)) * 0
                    tau_fs = tau_f_mean + tau_f_mean / 2 * np.random.randn(len(conns)) * 0
                    conns.set({
                        'U': np.maximum(Us, 0),
                        'tau_d': np.maximum(tau_ds, 1.),
                        'tau_f': np.maximum(tau_fs, 1.),
                    })

        # Randomize weights of each WTA circuit
        # for i in range(len(self.circuits)):
        #      utils.randomize_outgoing_connections(self.circuits[i].get_node_collection())

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

    def visualize_circuits_3d_barchart(self):
        # setup the figure and axes
        fig = plt.figure(figsize=(8, 3))
        ax1 = fig.add_subplot(121, projection='3d')
        ax2 = fig.add_subplot(122, projection='3d')

        # fake data
        data = self.get_circuit_grid()
        _x = np.arange(10)
        _y = np.arange(5)
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        top = data.flatten()
        bottom = np.zeros_like(top)
        width = depth = 1

        ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
        ax1.set_title('Shaded')

        ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
        ax2.set_title('Not Shaded')

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


NEURON_MODEL = 'iaf_psc_exp_wta'
SYNAPSE_MODEL = 'stdp_stp'
RATE_CONN_SYN_MODEL = 'rate_connection_instantaneous'
_NEURON_MODEL_NAME = NEURON_MODEL + "__with_" + SYNAPSE_MODEL
_SYNAPSE_MODEL_NAME = SYNAPSE_MODEL + "__with_" + NEURON_MODEL
regen = False


if __name__ == '__main__':
    # Generate NEST code
    utils.generate_nest_code(NEURON_MODEL, SYNAPSE_MODEL)

    print(_SYNAPSE_MODEL_NAME, " installed: ", _SYNAPSE_MODEL_NAME in nest.synapse_models)
    print(_NEURON_MODEL_NAME, " installed: ", _NEURON_MODEL_NAME in nest.node_models)
    nest.resolution = 1.
    nest.set_verbosity('M_ERROR')
    nest.print_time = False
    nest.SetKernelStatus({'resolution': 1.,
                          'use_compressed_spikes': False,
                          "local_num_threads": 1,
                          "total_num_virtual_procs": 1,
                          })  # no touch!

    utils.SYNAPSE_MODEL_NAME = _SYNAPSE_MODEL_NAME
    # TOY EXAMPLE
    grid = Network(grid_shape=(1, shared_params.n_wta), k_min=shared_params.n_neurons, k_max=shared_params.n_neurons,
                   n_inputs=shared_params.n_inp_channels)

    grid.get_node_collections().max_neuron_gid = max(grid.get_node_collections().global_id)

    print(grid.get_node_collections().rate_fraction)
    inpgen = InputGenerator(grid, r_noise=5., r_input=1e-12, r_noise_pattern=5., use_noise=False,
                            t_noise_range=[300.0, 500.0],
                            n_patterns=1, t_pattern=[300.], pattern_sequences=[[0]])

    recorder = utils.Recorder(grid, save_figures=False, show_figures=False, create_plot=False)
    recorder.set(create_plot=False)
    # id_list = recorder.run_network(inpgen=inpgen, t_sim=1, dt_rec=None, title="Simulation #1") #readout_size=30
    recorder.set(create_plot=True)
    recorder.run_network(inpgen=inpgen,
                         t_sim=shared_params.sim_time,
                         # t_sim=20.,
                         dt_rec=1, title="Test #1", train=True, order_neurons=False)
    recorder.set(plot_history=True)
    exit()
    recorder.run_network(inpgen=inpgen, t_sim=1, dt_rec=None, title="History", id_list=id_list, order_neurons=True)

    print(nest.GetKernelStatus('rng_seed'))
