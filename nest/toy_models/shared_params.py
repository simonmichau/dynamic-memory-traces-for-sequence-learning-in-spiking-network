# input_spikes = [10, 20, 30, 40, 50, 80]
# output_spikes = [15, 25, 70, 95]
# sim_time = 100

input_spikes = [ 5 ]
output_spikes = [ [10, 15],
                  [20, ] ]
sim_time = 30

use_stp_rec = True                  #  STP on recurrent weights
use_variance_tracking = True       # adaptive learning rate
# use_variance_tracking = False       # adaptive learning rate