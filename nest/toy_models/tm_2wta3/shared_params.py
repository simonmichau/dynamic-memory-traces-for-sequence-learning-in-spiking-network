import numpy as np
np.random.seed(1)
sim_time = 1000
input_spikes = sorted(np.random.random_integers(10, 900, 10).astype(float).tolist())
n_neurons = 3
output_spikes = [
    [sorted(np.random.random_integers(10, 900, 10).astype(float).tolist()) for _ in range(n_neurons)],
    [sorted(np.random.random_integers(10, 900, 10).astype(float).tolist()) for _ in range(n_neurons)],
]

use_fixed_spike_times = True

use_stp_rec = True                 # STP on recurrent weights
use_variance_tracking = True       # adaptive learning rate

# plot_neuron_ids = [1, 6]  # which neurons to plot, from 1 to 6
plot_neuron_ids = [2, 4]  # which neurons to plot, from 1 to 6
