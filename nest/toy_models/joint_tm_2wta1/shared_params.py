import numpy as np
np.random.seed(2)
sim_time = 2000
sim_time_sec = int(sim_time / 1000.)
n_inp_channels = 1
rate = 50
inp_rate = 30
n_wta = 2
n_neurons = 1

input_spikes = [sorted(np.random.random_integers(1, sim_time - 10, inp_rate * sim_time_sec).astype(float).tolist())
                for _ in range(n_inp_channels)]
output_spikes = [
    [sorted(np.random.random_integers(1, sim_time - 10, rate * sim_time_sec).astype(float).tolist())
     for _ in range(n_neurons)] for _ in range(n_wta)
]

# input_spikes = [ [5] ]
# output_spikes = [ [ [10, 15] ],
#                   [ [20, ] ] ]

use_fixed_spike_times = True

use_stp_rec = False                 # STP on recurrent weights
use_variance_tracking = True       # adaptive learning rate

# plot_neuron_ids = [1, 6]  # which neurons to plot, from 1 to 6
plot_neuron_ids = list(range(1, n_neurons + 1))
