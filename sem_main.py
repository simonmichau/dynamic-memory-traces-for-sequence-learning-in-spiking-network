import numpy
import matplotlib
matplotlib.use('agg')
import pylab
import sys

from sem_liquid import SEMLiquidParams, SEMLiquid


def sem_liquid_pattern4(seed=None, *p):
    strain = 100  # training set length (seconds)
    stest = 3
    params = SEMLiquidParams(task='pattern', nInputs=100, pattern_mode='random_switching', sprob=0.5, nPatterns=1,
                             rin=5,
                             tPattern=[300e-3] * 1, use_priors=False, plot_order_states=False, frac_tau_DS=10,
                             use_dynamic_synapses=True, rNoise=2, use_variance_tracking=True,  # eta=1e-4,
                             use_entropy_regularization=False, pattern_sequences=[[0], [0]],
                             test_pattern_sequences=[[0], [0]], seed=seed, size=(10, 5), pConn=1.0,
                             tNoiseRange=[300e-3, 500e-3], test_tNoiseRange=[300e-3, 800e-3], size_lims=[2, 10],
                             test_time_warp_range=(0.5, 2.))
    liquid = SEMLiquid(params)
    liquid.order_states2 = False
    liquid.sub_neurons = 4
    liquid.isppt = True
    liquid.ext = 'pdf'
    num_train_periods = 1
    test_times = numpy.arange(0, num_train_periods * strain + strain / 2.0, strain)  # aka test_times = [0]

    itest = 0
    test_titlestr = "response to patterns before training"  # test_titlestr = "initial response"

    # Pretrain test
    liquid.test_inpgen.inpgen.time_warp_range = (1., 1.)
    X, I, perf = liquid.test(stest, itest, titlestr=test_titlestr,
                             savestrprefix="sem_liquid_test_pre_train%d_inp" % (itest), plot_readout_spikes=False,
                             train_readouts=False, do_plot=True)
    liquid.test_inpgen.inpgen.time_warp_range = (0.5, 2.)

    for itest in range(0, num_train_periods):
        # Simulation
        liquid.simulate(strain, titlestr="training/simulating phase #%d" % (itest),
                        savestrprefix="sem_liquid_simulation%d" % (itest), do_plot=False)
        # Testing
        test_titlestr = "response to patterns after %ds training/simulation" % ((itest + 1) * strain)
        X, I, perf = liquid.test(stest, itest, titlestr=test_titlestr,
                                 savestrprefix="sem_liquid_test%d" % (itest), plot_readout_spikes=False,
                                 train_readouts=False, do_plot=True)

    pylab.close('all')
    return liquid, test_times


def sem_liquid_pattern4_intervals(seed=None, *p):
    strain = 10  # training set length (seconds)
    stest = 3
    params = SEMLiquidParams(task='pattern', nInputs=100, pattern_mode='random_switching', sprob=0.5, nPatterns=1,
                             rin=5,
                             tPattern=[300e-3] * 1, use_priors=False, plot_order_states=False, frac_tau_DS=10,
                             use_dynamic_synapses=True, rNoise=2, use_variance_tracking=True,  # eta=1e-4,
                             use_entropy_regularization=False, pattern_sequences=[[0], [0]],
                             test_pattern_sequences=[[0], [0]], seed=seed, size=(10, 5), pConn=1.0,
                             tNoiseRange=[300e-3, 500e-3], test_tNoiseRange=[300e-3, 800e-3], size_lims=[2, 10],
                             test_time_warp_range=(0.5, 2.))
    liquid = SEMLiquid(params)
    liquid.order_states2 = False
    liquid.sub_neurons = 4
    liquid.isppt = True
    liquid.ext = 'pdf'
    num_train_periods = 10
    test_times = numpy.arange(0, num_train_periods * strain + strain / 2.0, strain)  # aka test_times = [0]

    itest = 0
    test_titlestr = "response to patterns before training"  # test_titlestr = "initial response"

    # Pretrain test
    liquid.test_inpgen.inpgen.time_warp_range = (1., 1.)
    X, I, perf = liquid.test(stest, itest, titlestr=test_titlestr,
                             savestrprefix="sem_liquid_test_pre_train%d_inp" % (itest), plot_readout_spikes=False,
                             train_readouts=False, do_plot=True)
    liquid.test_inpgen.inpgen.time_warp_range = (0.5, 2.)

    for itest in range(0, num_train_periods):
        # Simulation
        liquid.simulate(strain, titlestr="training/simulating phase #%d" % (itest),
                        savestrprefix="sem_liquid_simulation%d" % (itest), do_plot=False)
        # Testing
        test_titlestr = "response to patterns after %ds training/simulation" % ((itest + 1) * strain)
        X, I, perf = liquid.test(stest, itest, titlestr=test_titlestr,
                                 savestrprefix="sem_liquid_test%d" % (itest), plot_readout_spikes=False,
                                 train_readouts=False, do_plot=True)

    pylab.close('all')
    return liquid, test_times


def sem_liquid_pattern5(seed=None, *p):
    strain = 50
    # stest_train = 3
    stest = 5
    params = SEMLiquidParams(task='pattern', nInputs=100, pattern_mode='random_switching', sprob=0.5, nPatterns=2,
                             rin=5,
                             tPattern=[300e-3] * 2, use_priors=False, plot_order_states=True, frac_tau_DS=5,
                             use_dynamic_synapses=True, rNoise=2, use_variance_tracking=False, eta=5e1,
                             use_entropy_regularization=False, pattern_sequences=[[0], [1]],
                             test_pattern_sequences=[[0], [1]], seed=seed, size=(10, 5),
                             tNoiseRange=[300e-3, 500e-3], test_tNoiseRange=[300e-3, 1000e-3], size_lims=[2, 25],
                             test_time_warp_range=(1., 1.))
    liquid = SEMLiquid(params)
    liquid.order_states2 = False
    liquid.sub_neurons = 4
    liquid.isppt = True
    liquid.ext = 'pdf'
    liquid.initialize_random_weights()
    num_train_periods = 1
    num_test_periods = 10
    test_times = numpy.arange(0, num_train_periods * strain + strain / 2.0, strain)

    itest = 0
    test_titlestr = "response to patterns before training"
    # liquid.use_inputs = False
    X, I, perf = liquid.test(stest, 2 * itest, titlestr=test_titlestr,
                             savestrprefix="sem_liquid_test%da_inp" % (itest), plot_readout_spikes=False,
                             train_readouts=False, do_plot=True)
    X, I, perf = liquid.test(stest, 2 * itest + 1, titlestr=test_titlestr,
                             savestrprefix="sem_liquid_test%db_inp" % (itest), plot_readout_spikes=False,
                             train_readouts=False, do_plot=True)
    # liquid.use_inputs = True

    for isimulation in range(1, num_train_periods + 1):
        # Training
        liquid.simulate(strain, titlestr="training phase #%d" % (isimulation),
                        savestrprefix="sem_liquid_train%d" % (isimulation + 1), do_plot=False)
        test_titlestr = "response to patterns after %ds training" % ((isimulation) * strain)
        # Testing
        for itest in range(0, num_test_periods):
            X, I, perf = liquid.test(stest, 2 * itest, titlestr=test_titlestr,
                                 savestrprefix="sem_liquid_test%da_inp" % (itest), plot_readout_spikes=False,
                                 train_readouts=False, do_plot=True)
            liquid.test_inpgen.tNoiseRange = [5000e-3, 5000e-3]  # deactivate pattern sequences
            X, I, perf = liquid.test(stest, 2 * itest, titlestr=test_titlestr,
                                 savestrprefix="sem_liquid_test%db_no_inp" % (itest), plot_readout_spikes=False,
                                 train_readouts=False, do_plot=True)
            liquid.test_inpgen.tNoiseRange = [300e-3, 500e-3]  # reactivate pattern sequences

    if liquid.ext == 'pdf' and len(liquid.figurelist) > 0:
        liquid.concatenate_pdf(delete=True)
    if len(liquid.figurelist) > 0:
        pylab.close('all')
    # pylab.close('all')
    return liquid, test_times


def sem_liquid_pattern5_intervals(seed=None, *p):
    strain = 5
    # stest_train = 3
    stest = 5
    params = SEMLiquidParams(task='pattern', nInputs=100, pattern_mode='random_switching', sprob=0.5, nPatterns=2,
                             rin=5,
                             tPattern=[300e-3] * 2, use_priors=False, plot_order_states=True, frac_tau_DS=5,
                             use_dynamic_synapses=True, rNoise=2, use_variance_tracking=False, eta=5e1,
                             use_entropy_regularization=False, pattern_sequences=[[0], [1]],
                             test_pattern_sequences=[[0], [1]], seed=seed, size=(10, 5),
                             tNoiseRange=[300e-3, 500e-3], test_tNoiseRange=[300e-3, 1000e-3], size_lims=[2, 25],
                             test_time_warp_range=(1., 1.))
    liquid = SEMLiquid(params)
    liquid.order_states2 = False
    liquid.sub_neurons = 4
    liquid.isppt = True
    liquid.ext = 'pdf'
    liquid.initialize_random_weights()
    num_train_periods = 10
    num_test_periods = 100
    test_times = numpy.arange(0, num_train_periods * strain + strain / 2.0, strain)

    itest = 0
    test_titlestr = "response to patterns before training"
    # liquid.use_inputs = False
    X, I, perf = liquid.test(stest, 2 * itest, titlestr=test_titlestr,
                             savestrprefix="sem_liquid_test0_%ds_inp0" % (itest), plot_readout_spikes=False,
                             train_readouts=False, do_plot=True)
    X, I, perf = liquid.test(stest, 2 * itest + 1, titlestr=test_titlestr,
                             savestrprefix="sem_liquid_test0_%ds_inp1" % (itest), plot_readout_spikes=False,
                             train_readouts=False, do_plot=True)
    # liquid.use_inputs = True

    for itest in range(1, num_train_periods + 1):
        liquid.simulate(strain, titlestr="training phase #%d" % (itest),
                        savestrprefix="sem_liquid_train%d" % (itest + 1), do_plot=False)
        test_titlestr = "response to patterns after %ds training" % (itest * strain)
        X, I, perf = liquid.test(stest, 2 * itest, titlestr=test_titlestr,
                                 savestrprefix="sem_liquid_test%d_%ds_inp0" % (itest, itest * strain), plot_readout_spikes=False,
                                 train_readouts=False, do_plot=True)
        X, I, perf = liquid.test(stest, 2 * itest + 1, titlestr=test_titlestr,
                                 savestrprefix="sem_liquid_test%d_%ds_inp1" % (itest, itest * strain), plot_readout_spikes=False,
                                 train_readouts=False, do_plot=True)
        # liquid.use_inputs = False
        liquid.test_inpgen.tNoiseRange = [5000e-3, 5000e-3]  # deactivate pattern sequences
        X, I, perf = liquid.test(stest, 2 * itest, titlestr=test_titlestr,
                                 savestrprefix="sem_liquid_test%d_%ds_noinp0" % (itest, itest * strain), plot_readout_spikes=False,
                                 train_readouts=False, do_plot=True)
        X, I, perf = liquid.test(stest, 2 * itest + 1, titlestr=test_titlestr,
                                 savestrprefix="sem_liquid_test%d_%ds_noinp1" % (itest, itest * strain), plot_readout_spikes=False,
                                 train_readouts=False, do_plot=True)
        # liquid.use_inputs = True
        liquid.test_inpgen.tNoiseRange = [300e-3, 500e-3]  # reactivate pattern sequences

    if liquid.ext == 'pdf' and len(liquid.figurelist) > 0:
        liquid.concatenate_pdf(delete=True)
    if len(liquid.figurelist) > 0:
        pylab.close('all')
    return liquid, test_times


def sem_liquid_pattern8(seed=None, *p):
    strain = 100
    stest_train = 3
    stest = 3
    params = SEMLiquidParams(task='pattern', nInputs=100, pattern_mode='random_ind', sprob=[0.8, 0.2], nPatterns=3,
                             use_priors=False, tPattern=[200e-3] * 3, pattern_sequences=[[0, 1], [0, 1]],
                             test_time_warp_range=(1., 1.),
                             test_pattern_sequences=[[1], [0, 1], [0]], tNoiseRange=[300e-3, 500e-3],
                             test_tNoiseRange=[300e-3, 800e-3],
                             seed=seed, frac_tau_DS=10, size=(10, 5), size_lims=[2, 10], use_dynamic_synapses=True,
                             use_entropy_regularization=False, plot_order_states=True, swap_order=True)
    liquid = SEMLiquid(params)
    num_train_periods = 1
    liquid.ext = 'pdf'
    liquid.sub_neurons = 4
    test_times = numpy.arange(0, num_train_periods * strain + strain / 2.0, strain)

    itest = -1
    # Pretest
    test_titlestr = "response to patterns before training"
    X, I, perf = liquid.test(stest_train, itest, titlestr=test_titlestr,
                             # titlestr="readout train phase after %ds training" % ((itest+1)*strain),
                             savestrprefix="sem_liquid_pre_train_test", plot_readout_spikes=True,
                             train_readouts=True, do_plot=True)

    liquid.simulate(strain, titlestr="training phase #%d" % (itest + 1),
                    savestrprefix="sem_liquid_train%d" % (itest + 1), do_plot=True)

    for itest in range(num_train_periods):
        test_titlestr = "response to patterns after %ds training" % ((itest + 1) * strain)
        X, I, perf = liquid.test(stest_train, itest, titlestr=test_titlestr,
                                 # titlestr="readout train phase after %ds training" % ((itest+1)*strain),
                                 savestrprefix="sem_liquid_test%d_train_inp" % itest, plot_readout_spikes=True,
                                 train_readouts=True, do_plot=True)
    liquid.plot_weight_distribution()
    if liquid.ext == 'pdf':
        liquid.concatenate_pdf(delete=True)
    pylab.close('all')
    return liquid, test_times

if __name__ == '__main__':

    matplotlib.rcParams['font.size'] = 16.0

    task = sys.argv[1]
    ntrials = 1
    if len(sys.argv) > 2:
        ntrials = (int)(sys.argv[2])
    seeds = [None] * ntrials
    params = [None] * ntrials

    if task == 'pattern4':
        seeds = [13521]
        fcn = sem_liquid_pattern4
    elif task == 'pattern4_intervals':
        seeds = [13521]
        fcn = sem_liquid_pattern4_intervals
    elif task == 'pattern5':
        seeds = [13521]
        fcn = sem_liquid_pattern5
    elif task == 'pattern5_intervals':
        seeds = [13521]
        fcn = sem_liquid_pattern5_intervals
    elif task == 'pattern8':
        seeds = [13521]
        fcn = sem_liquid_pattern8
    else:
        raise Exception("Task %s not implemented" % (task))

    ntrials = len(seeds)
    seeds_dict = dict()
    perfs_dict = dict()
    params_dict = dict()
    for i, seed in enumerate(seeds):
        print
        print "TRIAL %d/%d" % (i + 1, ntrials)
        print
        liquid, test_times = fcn(seed, params[i])
        seeds_dict[liquid.outputdir] = liquid.seed
        perfs_dict[liquid.outputdir] = liquid.test_performance
        params_dict[liquid.outputdir] = params[i]
    print
    keys = numpy.sort(seeds_dict.keys())

    if task == 'xor_pattern' or task == 'speech':  # or task=='pattern12':
        tmp = numpy.asarray(perfs_dict[keys[0]]['stimulus'])
        perf_stim = numpy.zeros((tmp.shape[0], tmp.shape[1], len(keys)))
        tmp = numpy.asarray(perfs_dict[keys[0]]['network'])
        perf_net = numpy.zeros((tmp.shape[0], tmp.shape[1], len(keys)))
        for ki, key in enumerate(keys):
            print key
            print 'seed:', seeds_dict[key]
            print 'params:', params_dict[key]
            print 'performance on stimulus:'
            perf_stim[:, :, ki] = numpy.asarray(perfs_dict[key]['stimulus'])
            print perf_stim[:, :, ki]
            print 'performance on network:'
            perf_net[:, :, ki] = numpy.asarray(perfs_dict[key]['network'])
            print perf_net[:, :, ki]
            print

        if task == 'xor_pattern':
            print "perf_stim_xor"
            # print perf_stim
            print numpy.mean(perf_stim[1::2, :2, :], axis=2)
            # print numpy.std(perf_stim, axis=0)
            # print "perf_stim2"
            # print numpy.mean(perf_stim2, axis=0)
            # print numpy.std(perf_stim2, axis=0)
            print "perf_net_xor"
            # print perf_net
            print numpy.mean(perf_net[1::2, :2, :], axis=2)
            # print numpy.std(perf_net, axis=0)
            print "perf_net_corr"
            print numpy.mean(perf_net[::2, 2:, :], axis=2)
            # print numpy.std(perf_net2, axis=0)
