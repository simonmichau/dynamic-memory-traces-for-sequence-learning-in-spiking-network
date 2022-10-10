import numpy as np
np.random.seed(1)
sim_time = 5000
sim_time_sec = int(sim_time / 1000.)
n_inp_channels = 50
rate = 33
inp_rate = 30

input_spikes = [sorted(np.random.random_integers(1, sim_time - 10, 5 * sim_time_sec).astype(float).tolist())
                for _ in range(n_inp_channels)]
n_neurons = 3
output_spikes = [
    [sorted(np.random.random_integers(1, sim_time - 10, rate * sim_time_sec).astype(float).tolist()) for _ in range(n_neurons)],
    [sorted(np.random.random_integers(1, sim_time - 10, rate * sim_time_sec).astype(float).tolist()) for _ in range(n_neurons)],
]

use_fixed_spike_times = True

use_stp_rec = True                 # STP on recurrent weights
use_variance_tracking = True       # adaptive learning rate

# plot_neuron_ids = [1, 6]  # which neurons to plot, from 1 to 6
plot_neuron_ids = np.arange(1, 6 + 1, 1)
