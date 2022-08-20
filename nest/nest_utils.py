import numpy as np
import matplotlib.pyplot as plt
import nest

import nest_network_v2 as main


def generate_poisson_spiketrain(t_duration, rate) -> list:
    """Generates a list of poisson generated spike times for a given

    - duration **t_duration** (in ms) and
    - firing rate **rate** (in Hz) """
    n = t_duration * rate * 1e-3  # expected number of spikes in [0, t_duration)
    scale = 1 / (rate * 1e-3)  # expected time between spikes
    isi = np.random.exponential(scale, int(np.ceil(n)))  # list of [ceil(n)] input spike intervals
    spikes = np.add.accumulate(isi)
    # Hypothetical position of t_duration in spikes list
    i = np.searchsorted(spikes, t_duration)

    # Add or remove spikes
    extra_spikes = []
    if i == len(spikes):
        t_last = spikes[-1]
        while True:
            t_last += np.random.exponential(scale, 1)[0]
            if t_last >= t_duration:
                break
            else:
                extra_spikes.append(t_last)
        spikes = np.concatenate((spikes, extra_spikes))
    else:
        # Cutoff spike times outside of spike duration
        spikes = np.resize(spikes, (i,))  # spikes[:i]
    a = spikes
    b = list(spikes)
    return spikes.tolist()


def run_simulation(inpgen: main.InputGenerator, t):
    """Pre-generates input patterns for duration of simulation and then runs the simulation"""
    print(nest.biological_time)
    inpgen.generate_input(t, t_origin=nest.biological_time)
    nest.Simulate(t)


def measure_node_collection(nc: main.NodeCollection, inpgen: main.InputGenerator = None, t_sim=300.0) -> None:
    """
    Simulates given **NodeCollection** for **t_sim** and plots the recorded spikes and membrane potential.
    Requires an **InputGenerator** object for pattern input generation.
    """
    multimeter = nest.Create('multimeter')
    multimeter.set(record_from=['V_m'])
    spikerecorder = nest.Create('spike_recorder')
    nest.Connect(multimeter, nc)
    nest.Connect(nc, spikerecorder)

    if inpgen is None:
        nest.Simulate(t_sim)
    else:
        run_simulation(inpgen, t_sim)

    dmm = multimeter.get()
    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(ts, Vms)
    ax1.set_ylabel("Membrane potential (mV)")

    dSD = spikerecorder.get("events")
    evs = dSD["senders"]
    ts = dSD["times"]

    ax2.plot(ts, evs, ".", color='orange')
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("spike events")

    plt.show()
