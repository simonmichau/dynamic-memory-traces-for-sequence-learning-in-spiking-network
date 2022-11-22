### Info: This is the development version of the project. For a cleaned up and more usable version check out the [sequence-learning](https://github.com/simonmichau/sequence-learning) repository.
# Dynamic memory traces for sequence learning in spiking networks
Repo of the bachelor thesis 'Dynamic memory traces for sequence learning in spiking networks'

## Setup
To run the code from the ``nest/`` subdirectory [NEST](https://nest-simulator.readthedocs.io/) and [NESTML](https://nestml.readthedocs.io/en/v5.0.0/index.html) (with `gh pr checkout 805` for issue #805) are required.

To run the code in the ``legacy/`` subdirectory Python 2.7 is required.

## Changing the NESTML
Updating the custom NESTML neuron and synapse models can be a bit tedious. The easiest way to do so is laid out here:

1. Generate a new target by setting the global variables `NEURON_MODEL` and `SYNAPSE_MODEL` to the names of your custom models in the `nest/nestml_models` subdirectory and calling `python nest_network.py regen_models`. This will create and install a new target in the `nest/nestml_targets` subdirectory.
2. Reimport all the custom changes made to support `normalization_sum` and `InstantaneousRateConnectionEvent` in the corresponding .cpp and .h files.
3. Run `make -j 4 install` to recompile the model with the custom changes.
