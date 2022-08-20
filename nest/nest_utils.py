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


def run_simulation(inpgen, t):
    """Pre-generates input patterns for duration of simulation and then runs the simulation"""
    inpgen.generate_input(t, t_origin=nest.biological_time)
    nest.Simulate(t)


def measure_node_collection(nc: main.NodeCollection, inpgen=None, t_sim=5000.0) -> None:
    """
    Simulates given **NodeCollection** for **t_sim** and plots the recorded spikes, membrane potential and presented
    patterns. Requires an **InputGenerator** object for pattern input generation.
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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    fig.set_figwidth(8)
    fig.set_figheight(6)
    # MEMBRANE POTENTIAL
    dmm = multimeter.get()
    Vms = dmm["events"]["V_m"]
    ts = dmm["events"]["times"]

    ax1.plot(ts, Vms)
    ax1.set_ylabel("Membrane potential (mV)")

    # SPIKE EVENTS
    dSD = spikerecorder.get("events")
    evs = dSD["senders"]
    ts = dSD["times"]

    ax2.plot(ts, evs, ".", color='orange')
    ax2.set_ylabel("Spikes")

    # PRESENTED PATTERNS
    time_shift = nest.biological_time - dmm["events"]["times"].size
    if inpgen is not None:
        st = inpgen.spiketrain
        for i in range(len(st)):
            # ax3.scatter(np.add(time_shift, st[i]), [i] * len(st[i]), color=(i / (len(st)), 0.0, i / (len(st))))
            ax3.plot(np.add(time_shift, st[i]), [i]*len(st[i]), ".", color='orange')

    ax3.set_ylabel("Input channels")
    ax3.set_xlabel("time (ms)")

    plt.show()
