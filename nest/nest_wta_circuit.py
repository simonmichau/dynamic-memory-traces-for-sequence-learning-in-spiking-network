import random
import nest
import nest.voltage_trace
import matplotlib.pyplot as plt

nest.ResetKernel()
nest.set_verbosity("M_ERROR")

class WTACircuit(object):
    def __init__(self, params, **kwds):
        # Random number K of neurons in a WTA circuit uniformly drawn from [2,10]
        self.K = random.randint(2, 10)
        # Rise and decay time constants in ms
        self.tau_rise = 2
        self.tau_decay = 20
        # Create K neurons with alpha shaped kernels and tau_rise=2ms, tau_decay=20ms
        self.neurons = nest.Create('iaf_psc_exp', K, params={'I_e': 0.0,
                                                             'tau_syn_ex': self.tau_rise,
                                                             'tau_m': self.tau_decay
                                                             })

        #self.noise
        # TODO: Inhibition


K = random.randint(2, 10)
wta_circuit = nest.Create('iaf_psc_exp', K, params={'I_e': 0.0,
                                                    'tau_syn_ex': 2,
                                                    'tau_m': 20
                                                    })

# Create Multimeter for measuring membrane potential
multimeter = nest.Create('multimeter')
multimeter.set(record_from=['V_m'])

# Create spike recorder recording spiking events of a neuron
spikerecorder = nest.Create('spike_recorder')

################################################################
# Add exhibitory and inhibitory noise to wta circuit
noise_ex = nest.Create("poisson_generator")
noise_in = nest.Create("poisson_generator")
noise_ex.set(rate=80000.0)
noise_in.set(rate=15000.0)

syn_dict_ex = {"weight": 1.2}
syn_dict_in = {"weight": -2.0}
nest.Connect(noise_ex, wta_circuit, syn_spec=syn_dict_ex)
nest.Connect(noise_in, wta_circuit, syn_spec=syn_dict_in)
################################################################
# Connect multimeter, WTA circuit and spikerecorder
nest.Connect(multimeter, wta_circuit)
nest.Connect(wta_circuit, spikerecorder)

nest.Simulate(300.0)

################################################################
# Plot multimeter
dmm = multimeter.get("events")
plt.figure(1)
for k in range(2):  # k in K
    Vms = dmm["V_m"][k::K]
    ts = dmm["times"][k::K]
    plt.plot(ts, Vms)

# Plot spikerecorder
dSD = spikerecorder.get("events")
plt.figure(2)
for k in range(2):  # k in K
    evs = dSD["senders"][k::K]
    ts = dSD["times"][k::K]
    plt.plot(ts, evs, ".")

#nest.voltage_trace.from_device(multimeter)
plt.show()
