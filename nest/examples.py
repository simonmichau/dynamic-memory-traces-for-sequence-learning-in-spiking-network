import random
import math

from nest_network_v2 import *

nest.ResetKernel()

# Create a new grid
grid = Network()
# Show a visualization of the grid
grid.visualize_circuits()
# Connect the nodes in the grid to each other
grid.form_connections()
# Visualize the targets of nodes from a given NodeCollection
grid.visualize_connections(grid.circuits[0].get_node_collection())

# Create a new population of input node
inpPop = InputPopulation(10)
# Connect the input population to the network
grid.connect_input(inpPop)

# Simulate network and plot spike events along with membrane potential
measure_node_collection(grid._get_node_collections(slice(1, 5)))
measure_node_collection(inpPop.pop)
