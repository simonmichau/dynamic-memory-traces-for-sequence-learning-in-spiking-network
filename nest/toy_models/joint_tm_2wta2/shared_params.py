import numpy as np
np.random.seed(1)
sim_time = 2000
xlim = [0, 1000]
sim_time_sec = max(1, int(sim_time / 1000.))
n_inp_channels = 1
rate = 50
inp_rate = 20
n_wta = 2
n_neurons = 2

input_spikes = [sorted(np.unique(np.random.random_integers(1, sim_time - 10, inp_rate * sim_time_sec)).astype(float).tolist())
                for _ in range(n_inp_channels)]
output_spikes = [
    [sorted(np.unique(np.random.random_integers(2, sim_time - 10, rate * sim_time_sec)).astype(float).tolist())
     for _ in range(n_neurons)] for _ in range(n_wta)
]

# TODO OKAY
# input_spikes = [ [] ]
# output_spikes = [ [ [2, 3, 10, 12], [5, 6] ],
#                   [ [2, 3, 9, 10, ], [4] ] ]

# TODO OKAY
# input_spikes = [ [] ]
# output_spikes = [ [ [2, ], [5, ] ],
#                   [ [2, ], [4] ] ]

# TODO OKAY
# input_spikes = [ [] ]
# output_spikes = [ [ [4, 5, ], [2, 4, 5] ],
#                   [ [2, 7,], [-4, ] ] ]

# input_spikes = [ [] ]
# output_spikes = [ [ [3, 4, 5, ], [2, 4, 7] ],  # TODO seems to work
#                   [ [2, 7,], [-1, ] ] ]

use_fixed_spike_times = True

use_stp_rec = False                 # STP on recurrent weights
use_variance_tracking = True       # adaptive learning rate

# plot_neuron_ids = [1, 6]  # which neurons to plot, from 1 to 6
plot_neuron_ids = list(range(1, n_neurons + 1))
