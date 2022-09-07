"""
nest_stdp_window.py

---

Visualizes the STDP window of our custom synapse. Because `iaf_psc_exp_wta` is not suited for this due to its reliance
on InstantaneousRateConnectionEvents, `iaf_psc_exp_test` is used here.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import nest
from nest_utils import plot_weights


nest.Install("nestmlmodule")

NEURON_MODEL = 'iaf_psc_exp_test'
SYNAPSE_MODEL = 'stdp'

NEURON_MODEL_NAME = NEURON_MODEL + "__with_" + SYNAPSE_MODEL
SYNAPSE_MODEL_NAME = SYNAPSE_MODEL + "__with_" + NEURON_MODEL


def weight_update_test(pre_spike_time, post_spike_time):
    # Setup weight recorder
    weight_recorder = nest.Create('weight_recorder')
    global SYNAPSE_MODEL_NAME
    nest.CopyModel(SYNAPSE_MODEL_NAME, 'recording_synapse', {"weight_recorder": weight_recorder})
    SYNAPSE_MODEL_NAME = 'recording_synapse'

    # Setup test network
    spikegen_pre = nest.Create('spike_generator', params={"spike_times": [post_spike_time]})
    spikegen_post = nest.Create('spike_generator', params={"spike_times": [pre_spike_time]})
    parrot_pre = nest.Create('parrot_neuron')
    parrot_post = nest.Create('parrot_neuron')
    neuron_post = nest.Create(NEURON_MODEL_NAME)

    nest.Connect(spikegen_pre, parrot_pre)
    nest.Connect(parrot_pre, neuron_post, "one_to_one", syn_spec={'synapse_model': SYNAPSE_MODEL_NAME})
    nest.Connect(spikegen_post, parrot_post)
    nest.Connect(parrot_post, neuron_post, 'one_to_one', syn_spec={"weight": 9999.})

    # Measure weights before and after Simulation
    print(nest.GetConnections(parrot_pre, neuron_post))
    nest.Simulate(100.)
    print(nest.GetConnections(parrot_pre, neuron_post))
    plot_weights(weight_recorder)


def run_network(pre_spike_time, post_spike_time,
                          neuron_model_name,
                          synapse_model_name,
                          resolution=1., # [ms]
                          delay=1., # [ms]
                          sim_time=None,  # if None, computed from pre and post spike times
                          synapse_parameters=None,  # optional dictionary passed to the synapse
                          fname_snip=""):

    nest.set_verbosity("M_WARNING")
    #nest.set_verbosity("M_ALL")

    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': resolution})

    weight_var = 'w'
    wr = nest.Create('weight_recorder')
    nest.CopyModel(synapse_model_name, "stdp_nestml_rec",
                {"weight_recorder": wr[0],
                 "delay": delay,
                 "d": delay,
                 "receptor_type": 0})

    # create spike_generators with these times
    pre_sg = nest.Create("spike_generator",
                         params={"spike_times": [pre_spike_time, sim_time - 10.]})
    post_sg = nest.Create("spike_generator",
                          params={"spike_times": [post_spike_time],
                                  'allow_offgrid_times': True})

    # create parrot neurons and connect spike_generators
    pre_neuron = nest.Create("parrot_neuron")
    post_neuron = nest.Create(neuron_model_name,
                              params={'t_ref': 2000.0}  # set t_ref very high so it doesn't affect the investigated time window
                              )

    spikedet_pre = nest.Create("spike_recorder")
    spikedet_post = nest.Create("spike_recorder")
    #mm = nest.Create("multimeter", params={"record_from" : ["V_m"]})

    nest.Connect(pre_sg, pre_neuron, "one_to_one", syn_spec={"delay": 1.})
    nest.Connect(post_sg, post_neuron, "one_to_one", syn_spec={"delay": 1., "weight": 9999.})
    nest.Connect(pre_neuron, post_neuron, "all_to_all", syn_spec={'synapse_model': 'stdp_nestml_rec'})
    #nest.Connect(mm, post_neuron)

    nest.Connect(pre_neuron, spikedet_pre)
    nest.Connect(post_neuron, spikedet_post)

    # get STDP synapse and weight before protocol
    syn = nest.GetConnections(source=pre_neuron, synapse_model="stdp_nestml_rec")

    initial_weight = syn.get(weight_var)
    np.testing.assert_allclose(initial_weight, 0.)
    nest.Simulate(sim_time)
    updated_weight = syn.get(weight_var)

    actual_t_pre_sp = nest.GetStatus(spikedet_pre)[0]["events"]["times"][0]
    actual_t_post_sp = nest.GetStatus(spikedet_post)[0]["events"]["times"][0]

    dt = actual_t_post_sp - actual_t_pre_sp
    dw = (updated_weight - initial_weight)

    return dt, dw


def stdp_window(neuron_model_name, synapse_model_name, synapse_parameters=None):
    sim_time = 1000.  # [ms]
    pre_spike_time = 100. #sim_time / 2  # [ms]
    delay = 10. # dendritic delay [ms]

    dt_vec = []
    dw_vec = []
    for post_spike_time in np.arange(25, 175).astype(float):
        dt, dw = run_network(pre_spike_time, post_spike_time,
                          neuron_model_name,
                          synapse_model_name,
                          resolution=1., # [ms]
                          delay=delay, # [ms]
                          synapse_parameters=synapse_parameters,
                          sim_time=sim_time)
        dt_vec.append(dt)
        dw_vec.append(dw)

    return dt_vec, dw_vec, delay


def plot_stdp_window(dt_vec, dw_vec, delay):
    fig, ax = plt.subplots(dpi=120)
    # ax.scatter(dt_vec, dw_vec)
    ax.plot(dt_vec, dw_vec, 'o-', lw=2)
    ax.set_xlabel(r"t_post - t_pre [ms]")
    ax.set_ylabel(r"$\Delta w$")

    for _ax in [ax]:
        _ax.grid(which="major", axis="both")
        _ax.grid(which="minor", axis="x", linestyle=":", alpha=.4)
        _ax.set_xlim(np.amin(dt_vec), np.amax(dt_vec))
        #_ax.set_xlim(-100., 100.)
        #_ax.minorticks_on()
        #_ax.set_xlim(0., sim_time)

    ylim = ax.get_ylim()
    ax.plot((np.amin(dt_vec), np.amax(dt_vec)), (0, 0), linestyle="--", color="black", linewidth=2, alpha=.5)
    ax.plot((-delay, -delay), ylim, linestyle="--", color="black", linewidth=2, alpha=.5)
    ax.set_ylim(ylim)
    ax.grid(False)
    fig.show()
    plt.show()


# weight_update_test(100.0, 20.)

print("STDP Window for: ", NEURON_MODEL_NAME, " with ", SYNAPSE_MODEL_NAME)
dt_vec, dw_vec, delay = stdp_window(NEURON_MODEL_NAME, SYNAPSE_MODEL_NAME)
plot_stdp_window(dt_vec, dw_vec, delay)


