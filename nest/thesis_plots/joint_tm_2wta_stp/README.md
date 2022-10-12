### Joint model with 2 wta 2 neurons 
Compare traces for the 2 circuits, 2 neurons each

* Solved STP issues in legacy code - one has to decide whether to allow previous errors, and adapt NEST accordingly

#### Result
* Perfect overlap for many tested values
* This holds if the legacy code is corrected such that STP is entirely disabled on the input nodes

#### Model assumptions
* WITH STP
* only accepts one input channel
* use fixed spike times, separate for each neuron
* 