import numpy as np
import shared_params
import pickle
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns

np.random.seed(1)


with open('nest_data.pkl', 'rb') as f:
    data_nest = pickle.load(f)
with open('klampfl_data.pkl', 'rb') as f:
    data_klampfl = pickle.load(f, encoding='latin1')

fig, axes = plt.subplots(7, 1, sharex=True)
fig.set_figwidth(9)
fig.set_figheight(12)

c_nest = 'tomato'
c_klampfl = 'cornflowerblue'

cp_nest = sns.color_palette("flare", n_colors=8)
cp_klampfl = sns.color_palette("crest", n_colors=8)

ls = ['-', '--']
lw1 = 2.5
neuron_ids = {'nest': np.arange(1, shared_params.n_neurons * shared_params.n_wta + 1),
              'klampfl': np.arange(0, shared_params.n_neurons * shared_params.n_wta)}
input_ids = {'nest': [14], 'klampfl': [0, 1]}
marker = None

# for idx, key in enumerate(['inp_spikes', 'spikes', 'Vms', 'epsps', 'inp_weights', 'rec_weights',  ][:4]):
#     axes[idx].plot

# Input spikes
for idx, n in enumerate(neuron_ids['klampfl']):
    if idx == 0:
        indices = np.where(data_klampfl['inp_spikes'][n][1])[0]
        times = data_klampfl['inp_spikes'][n][0][indices]
        axes[0].plot(times, [idx] * len(times), 'o', color=c_klampfl)

# for idx, n in enumerate(data_nest['inp_spikes'].shape):
nest_inp_delay = 1.
axes[0].plot(data_nest['inp_spikes'] + nest_inp_delay, [0] * len(times), 'v', color=c_nest)

# Output spikes
for idx, n in enumerate(neuron_ids['nest']):
    times = data_nest['spikes'][n]
    axes[1].plot(times, [idx] * len(times), 'v', ms=12, color=cp_nest[idx], alpha=0.9, label="NEST" if idx==0 else "")
for idx, n in enumerate(neuron_ids['klampfl']):
    indices = np.where(data_klampfl['spikes'][n][1])[0]
    times = data_klampfl['spikes'][n][0][indices]
    axes[1].plot(times, [idx] * len(times), 'o', ms=12, color=cp_klampfl[idx], alpha=0.5, label="Klampfl" if idx==0 else "")
axes[1].legend(handlelength=3.)

# Vms
for idx, (n, n2) in enumerate(zip(neuron_ids['nest'], neuron_ids['klampfl'])):
    axes[2].plot(data_nest['Vms'][n][0], data_nest['Vms'][n][1], marker=marker,
                 color=cp_nest[idx], lw=lw1, ls=ls[0], label="NEST" if idx==0 else "")
    axes[2].plot(data_klampfl['Vms'][n2][0], data_klampfl['Vms'][n2][1], marker=marker,
                 color=cp_klampfl[idx], lw=lw1, ls=ls[1], label="Klampfl" if idx==0 else "")
    # print(f"Vm diff: {mean_squared_error(data_nest['Vms'][n][1], data_klampfl['Vms'][n2][1][:-1])}")
axes[2].legend()

# Input weights
for neuron_idx, (n, n2) in enumerate(zip(neuron_ids['nest'], neuron_ids['klampfl'])):
    for idx, (n_i, n_i2) in enumerate(zip(input_ids['nest'], input_ids['klampfl'])):
        axes[3].plot(data_nest['inp_weights'][n][n_i][0], data_nest['inp_weights'][n][n_i][1],
                     marker=marker, color=cp_nest[neuron_idx], ls=ls[0], lw=lw1, label="NEST" if neuron_idx==0 else "")

    kw = data_klampfl['inp_weights'][1][:, neuron_idx, 0]
    axes[3].plot(data_klampfl['inp_weights'][0], kw, marker=marker,  color=cp_klampfl[neuron_idx], ls=ls[1], lw=lw1, label="Klampfl" if neuron_idx==0 else "")
axes[3].legend()

# Recurrent weights
# for neuron_idx, (n, n2) in enumerate(zip(neuron_ids['nest'], neuron_ids['klampfl'])):

cnt = 0
cnt_klampfl = 0
for wta in range(shared_params.n_wta):
    for k_tgt, v_tgt in data_nest['rec_weights'].items():
        for k_src, v_src in v_tgt.items():
            wta_tgt = (k_tgt - 1) // shared_params.n_wta
            wta_src = (k_src - 1) // shared_params.n_wta

            if len(v_src[0]) > 1 and wta == wta_src:
                # print(f"accepting {k_src} -> {k_tgt}")
                axes[4 + wta].plot(v_src[0], v_src[1],
                             # color=cp_nest[k2 - 1],
                             color=cp_nest[cnt],
                             ls=ls[0], lw=lw1, alpha=0.6, label="NEST" if (k_src==1 and k_tgt==3) or (k_src==3 and k_tgt==1) else "")
                cnt += 1
    print(f"plotted {cnt} NEST rec conns")

    for idx, n_tgt in enumerate(neuron_ids['klampfl']):
        for idx2, n_src in enumerate(neuron_ids['klampfl']):
            wta_src = n_src // shared_params.n_wta
            wta_tgt = n_tgt // shared_params.n_wta
            if n_tgt != n_src:
                # if np.any(data_klampfl['rec_weights'][1][:, n_tgt, n_src]):
                if wta == wta_src and wta_src != wta_tgt:
                    # print(f"accepting {n_src} -> {n_tgt}")
                    axes[4 + wta].plot(data_klampfl['rec_weights'][0], data_klampfl['rec_weights'][1][:, n_tgt, n_src],
                                 # color=cp_klampfl[idx2],
                                 color=cp_klampfl[cnt_klampfl],
                                 ls=ls[1], lw=lw1, label="Klampfl" if (idx==0 and idx2==2) or (idx==2 and idx2==0) else "")
                    cnt_klampfl += 1
    print(f"plotted {cnt_klampfl} Klampfl rec conns")
    axes[4+wta].legend()

################
# EPSPs
for idx, (n, n2) in enumerate(zip(neuron_ids['nest'], neuron_ids['klampfl'])):
    # if idx >= 2:
    #     continue
    for i in neuron_ids['nest']:
        for j, v in data_nest['epsps'][i].items():
            if len(v[0]) == 0:
                continue
            if j == n:
                times = np.array(v[0])
                axes[6].plot(times[1:] - 1, v[1][1:], marker=marker, color=cp_nest[idx], lw=[2.5, 4][i%2],
                             ls=ls[i%2],
                             #label=f'{j - 1} -> {i - 1}',
                             label="NEST" if idx==3 and i == 2 else "")

    axes[6].plot(data_klampfl['epsps'][n2][0], data_klampfl['epsps'][n2][1], marker=marker,
                 color=cp_klampfl[idx], lw=lw1, ls=ls[1], label="Klampfl" if idx==0 else "")
axes[6].legend(handlelength=3.)

axes[0].set_title('Input spikes')
axes[1].set_title('Output spikes')
axes[2].set_title('Voltage trace')
# axes[2].set_ylim([-0.2, 0.1])

axes[3].set_title('Input weights')
axes[4].set_title('Recurrent weights of WTA #1')
axes[5].set_title('Recurrent weights of WTA #2')
axes[6].set_title('Outgoing EPSPs (Y)')

plt.xlim(shared_params.xlim)
plt.tight_layout()
plt.show()