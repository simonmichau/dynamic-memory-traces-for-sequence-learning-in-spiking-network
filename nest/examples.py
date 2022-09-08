import random
import math
import nest

from nest_network import Network, InputGenerator

nest.ResetKernel()

# NETWORK ##############################################################################################################

# Create a new grid of WTA circuits
grid = Network(grid_shape=(10, 5), k_min=2, k_max=10, n_inputs=50, lam=0.088, save_figures=False, show_figures=True)
# Show a visualization of the grid
grid.visualize_circuits()
# Show a 3D visualization of the grid
grid.visualize_circuits_3d()
# Visualize the targets of nodes from a given NodeCollection
grid.visualize_connections(grid.circuits[0].get_node_collection())
# Returns the NodeCollections of circuits within slice
grid.get_node_collections(5, 8)

# INPUT GENERATOR ######################################################################################################

# Create a new input generator for the `grid` Network
inpgen = InputGenerator(grid, r_noise=2, r_input=5, n_patterns=3, pattern_sequences=[[0, 1], [2]],
                        pattern_mode='random_iterate', p_switch=1., t_pattern=[300., 200, 300.],
                        t_noise_range=[100.0, 500.0], use_noise=True)

# MEASURE ##############################################################################################################
# Simulates network and plots input spike events along with membrane potential and spiketrains

# Measure whole grid for [t_sim] without pattern input
measure_network(grid, t_sim=5000.0)
# Measure 20 randomly selected nodes from grid (with pattern input) and return the ids of the measured nodes
id_list = measure_network(grid, inpgen=inpgen,  readout_size=20, t_sim=5000.0)
# Measure a previously chosen list of nodes
measure_network(grid, inpgen=inpgen, id_list=id_list, t_sim=5000.0)
# Measure a specified NodeCollection
measure_network(grid, inpgen=inpgen, node_collection=grid.get_node_collections(6, 9), t_sim=5000.0)
