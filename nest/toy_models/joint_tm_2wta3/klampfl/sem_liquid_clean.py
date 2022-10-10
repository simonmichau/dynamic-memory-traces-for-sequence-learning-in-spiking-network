"""
* randomization of STP params disabled
* all-to-all recurrent connectivity
"""


import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import sys

sys.path.append('../')
import shared_params

matplotlib.use('Agg')
import timeit
import itertools
import scipy.linalg
import datetime
import shutil, os
import logging
import copy

import mdp
#from enthought.mayavi import mlab

from sem_recorder import *
from sem_input import *
#from sem import *

class SEMLiquidParams(object):
    def __init__(self, task, **kwds):
        # random seed
        self.seed = kwds.get('seed',None)
        if self.seed is None:
            self.seed = int(numpy.random.randint(2**16))
        numpy.random.seed(self.seed)

        # size of the network, WTAs are aligned on a 2D grid
        self.size = kwds.get('size', (5, 5))
        self.n, self.m = self.size
        self.npos = numpy.prod(self.size)
        # each WTA has a random number of neurons, drawn from this interval
        self.size_lims = kwds.get('size_lims', [2,10])
        # number of external input channels
        self.nInputs = kwds.get('nInputs', 50)

        # parameter of exponential distance distribution
        self.lam = kwds.get('lambda', 0.088)
        # total number of recurrent connections
        self.nConn = kwds.get('nConn', 1000)
        self.nConn = min(self.nConn, self.npos*(self.npos-1))
        # fraction of connections used (in the end, there will be nConn*pConn connections)
        self.pConn = kwds.get('pConn', 1.0)
        # target fraction of input connections (at most all inputs are fed into each WTA)
        self.Cinscale = kwds.get('Cinscale', 1.0)

        # fraction of WTAs that are trained
        self.train_fraction = kwds.get('train_fraction', 1.0)
        # fraction of WTAs that receive input
        self.input_fraction = kwds.get('input_fraction', 1.0)

        # simulation time step
        self.dt = kwds.get('dt',1e-3)
        # decay time constant of EPSP
        self.tau = kwds.get('tau',20e-3)
        # rise time constant of EPSP
        self.tau2 = kwds.get('tau2',2e-3)
        # initial learning rate
        self.eta = kwds.get('eta', 0.05)
        # maximal firing rate of each WTA neuron
        self.rmax = kwds.get('rmax', 100)
        # noise applied to z-Neurons
        self.Znoise = kwds.get('Znoise', 0.0)

        # use (and adapt) bias/excitability term?
        self.use_priors = kwds.get('use_priors', True)
        # use adaptive learning rate?
        self.use_variance_tracking = kwds.get('use_variance_tracking', True)
        # create any recurrent connections?
        self.use_recurrent_connections = kwds.get('use_recurrent_connections', True)
        # adapt the input connections?
        self.train_inputs = kwds.get('train_inputs', True)
        # use recurrent connections within a single WTA circuit?
        self.use_self_recurrent_connections = kwds.get('use_self_recurrent_connections', False)
        # use dynamic synapses for recurrent connections?
        self.use_dynamic_synapses = kwds.get('use_dynamic_synapses', True)
        # use dynamic synapses for input connections?
        self.use_dynamic_input_synapses = kwds.get('use_dynamic_input_synapses', False)  # INFO: STP DISABLED FOR INPUT SYNAPSES
        # use dynamic synapses for readout connections? (only for WTA readouts)
        self.use_dynamic_readout_synapses = kwds.get('use_dynamic_readout_synapses', False)
        # use individual synapses for each connection, or use a single EPSP instance for each
        # time constants of dynamical synapses are divided by this number
        self.frac_tau_DS = kwds.get('frac_tau_DS', 10)
        # initalize weights randomly (or zero otherwise?)
        self.random_initial_weights = kwds.get('random_initial_weights', False)

        # number of neurons for the WTA readout (if any)
        self.nReadouts = kwds.get('nReadouts', 0)
        # maximum rate of readout WTA (if used)
        self.rmax_rdt = kwds.get('rmax_rdt', 1000)

        # number of time steps between each output update
        self.dt_out = kwds.get('dt_out',1)
        # each this number of time steps a recording is initialized (during training)
        self.dt_rec_spk = kwds.get('dt_rec_spk', 5000)
        # this recording lasts for this number of time steps
        self.nstepsrec = kwds.get('nstepsrec',5000)
        # number of time steps for recording stuff
        self.dt_rec = kwds.get('dt_rec', 100)
        # plot network response ordered by the mean activation time of neurons?
        self.plot_order_states = kwds.get('plot_order_states', True)
        self.swap_order = kwds.get('swap_order', False)

        # 'multi', 'pattern', or 'speech'
        self.task = task

        # for pattern and speech task:

        # input firing rate
        self.rin = kwds.get('rin',5)
        # number of patterns
        self.nPatterns = kwds.get('nPatterns',2)
        # pattern durations
        self.tPattern = kwds.get('tPattern', [400e-3]*self.nPatterns)
        if not hasattr(self.tPattern,'__iter__'):
            self.tPattern = numpy.ones(self.nPatterns)*self.tPattern
        else:
            self.tPattern = numpy.asarray(self.tPattern)
        # the actual input consists of these pattern sequences (even if single patterns are presented)
        self.pattern_sequences = kwds.get('pattern_sequences', [[0],[1]])
        # pattern sequences used for testing
        self.test_pattern_sequences = kwds.get('test_pattern_sequences', [[0],[1]])
        # embed pattern sequences into noise?
        self.embedded = kwds.get('embedded',True)
        # Poisson firing rate that is laid over the patterns
        self.rNoise = kwds.get('rNoise',2.0)
        # noise phase is drawn randomly from this interval
        self.tNoiseRange = kwds.get('tNoiseRange',[100e-3,500e-3])
        # noise phase time range used for testing
        self.test_tNoiseRange = kwds.get('test_tNoiseRange',[100e-3,500e-3])
        # noise phase is drawn randomly from this interval
        self.time_warp_range = kwds.get('time_warp_range',[1.0, 1.0])
        # noise phase time range used for testing
        self.test_time_warp_range = kwds.get('test_time_warp_range',[1.0,1.0])
        # mode of presenting patterns:
        # random_ind: pattern sequences are independently chosen from the prob. dist. defined by "sprob"
        # random_switching: pattern sequences are switched after each presentation with probability "sprob"
        # alternating: pattern sequences are presented in order
        self.pattern_mode = kwds.get('pattern_mode', 'random_ind')
        # can be a single probability value or vector, see above
        self.sprob = kwds.get('sprob', [0.8, 0.2])
        # probability with which noise is presented after each pattern sequence
        self.nprob = kwds.get('nprob', 1.0)

        # for MULTI task:

        # number of different input rates
        self.nInputGroups = kwds.get('nInputGroups', 2)
        # rates are chosen randomly from this interval
        self.in_rate_lims = kwds.get('in_rate_lims', (10,80))
        # duration during which rate stays constant, in units of EPSP time constant
        self.tau_multiple = kwds.get('tau_multiple', 5)
        self.tau_multi = kwds.get('tau_multi', 20e-3)

        # create input generators (for training)
        if self.task == 'pattern':
            self.inpgen_seq = SequentialPatternInputGenerator(self.nInputs, self.dt, self.rin,
                self.nPatterns, self.tPattern, self.pattern_sequences, self.rNoise, self.pattern_mode, self.sprob, self.time_warp_range)
            if self.embedded:
                self.inpgen = EmbeddedPatternInputGenerator(self.inpgen_seq, tNoiseRange=self.tNoiseRange, prob=self.nprob)
            else:
                self.inpgen = self.inpgen_seq
        elif self.task == 'multi':
            self.inpgen = RateInputGenerator(self.nInputs, self.dt, self.tau_multiple*self.tau_multi, self.tau,
                self.nInputGroups, self.in_rate_lims)
        elif self.task == 'xor':
            inpgen1 = SequentialPatternInputGenerator(self.nInputs/2, self.dt, self.rin,
                self.nPatterns, self.tPattern, self.pattern_sequences, self.rNoise, self.pattern_mode, self.sprob)
            inpgen2 = SequentialPatternInputGenerator(self.nInputs-self.nInputs/2, self.dt, self.rin,
                self.nPatterns, self.tPattern, self.pattern_sequences, self.rNoise, self.pattern_mode, self.sprob)
            self.inpgen = CombinedInputGenerator([inpgen1,inpgen2])
        elif self.task == 'speech':
            # use only a part of the utterances for training
            tmp = numpy.random.permutation(10)[:1].tolist()
            tmp = [1,2,5,6,7][numpy.random.randint(5)]
            if self.digits is None:
                self.digits = numpy.random.permutation(10)[:2].tolist()
            if self.speakers is None:
                self.speakers = numpy.take([1,2,5,6,7], numpy.random.permutation(5)[:1])
            self.inpgen_speech = PreprocessedSpeechInputGenerator(self.nInputs, self.dt, self.rin, self.rNoise,
                scale=50, poisson=False, digits=self.digits, rev_digits=self.rev_digits, utterances=self.utterances[:7],
                speakers=self.speakers, sprob=self.sprob)
            if self.embedded:
                self.inpgen = EmbeddedPatternInputGenerator(self.inpgen_speech, tNoiseRange=self.tNoiseRange,
                    prob=self.nprob)
            else:
                self.inpgen = self.inpgen_speech
        elif self.task == 'xor_pattern':
            inpgen1 = SequentialPatternInputGenerator(self.nInputs/2, self.dt, self.rin,
                self.nPatterns, self.tPattern[:self.nPatterns], self.pattern_sequences[0], self.rNoise, self.pattern_mode, self.sprob)
            inpgen2 = SequentialPatternInputGenerator(self.nInputs-self.nInputs/2, self.dt, self.rin,
                self.nPatterns, self.tPattern[:self.nPatterns], self.pattern_sequences[0], self.rNoise, self.pattern_mode, self.sprob)
            self.inpgen_xor = CombinedInputGenerator([inpgen1,inpgen2])
            self.inpgen_seq = SequentialPatternInputGenerator(self.nInputs, self.dt, self.rin,
                self.nPatterns, self.tPattern[self.nPatterns:], self.pattern_sequences[1], self.rNoise, self.pattern_mode, self.sprob)
            if self.embedded:
                self.inpgen_pat = EmbeddedPatternInputGenerator(self.inpgen_seq, tNoiseRange=self.tNoiseRange, prob=self.nprob)
            else:
                self.inpgen_pat = self.inpgen_seq
            self.inpgen = SwitchableInputGenerator([self.inpgen_xor, self.inpgen_pat])

        # create input generator for testing
        self.test_inpgen = copy.deepcopy(self.inpgen) # deepcopy is used here because python just creates a binding between object and target when assignments are used
        if self.task=='pattern':
            # for testing, always switch through patterns
            if self.embedded:
                self.test_inpgen.inpgen.pattern_sequences = self.test_pattern_sequences
                self.test_inpgen.inpgen.mode = 'alternating'
                self.test_inpgen.inpgen.sprob = 1.0
                self.test_inpgen.inpgen.time_warp_range = self.test_time_warp_range
                self.test_inpgen.tNoiseRange = self.test_tNoiseRange
            else:
                self.test_inpgen.pattern_sequences = self.test_pattern_sequences
                self.test_inpgen.mode = 'alternating'
                self.test_inpgen.sprob = 1.0
                self.test_inpgen.time_warp_range = self.test_time_warp_range
        elif self.task == 'speech':
            # use remaining utterances for testing
            if self.embedded:
                self.test_inpgen.inpgen.sprob = 1.0
                self.test_inpgen.inpgen.utterances = self.utterances[7:]
                self.test_inpgen.tNoiseRange = self.test_tNoiseRange
                self.test_inpgen.inpgen.time_warp_range = self.test_time_warp_range
            else:
                self.test_inpgen.sprob = 1.0
                self.test_inpgen.utterances = self.utterances[7:]
                self.test_inpgen.time_warp_range = self.test_time_warp_range

class SEMLiquid(object):
    def __init__(self, params, **kwds):

        self.__dict__.update(params.__dict__)

        #print numpy.random.rand()
        simpars = "Simulation parameters:\n"
        for key in numpy.sort(self.__dict__.keys()):
            simpars += " "+key+"="+str(self.__dict__[key])+"\n"
        print simpars

        # draw sizes of WTA circuits
        self.sizes = numpy.random.randint(self.size_lims[0], self.size_lims[1]+1, self.size)
        # total number of neurons
        self.tsize = numpy.sum(self.sizes)
        # membrane potential
        self.u = numpy.zeros(self.tsize+self.nReadouts)
        # membrane potential resulting only from input connections
        self.u_inp = numpy.zeros(self.tsize+self.nReadouts)
        # membrane potential resulting only from recurrent connections
        self.u_rec = numpy.zeros(self.tsize+self.nReadouts)
        # trace of membrane potentials during simulation (for plotting)
        self.u_trace = []
        # trace of input membrane potentials during simulation (for plotting)
        self.u_inp_trace = []
        self.spk_out_trace = []
        self.epsp_trace = []
        self.epsp2_trace = []
        # Trace of udyn and rdyn
        self.isi_trace = []
        self.rdyn_trace = []
        self.udyn_trace = []
        self.dEPSP_trace = []
        # trace of input and recurrent weight during simulation
        self.w_inp_trace = []
        self.w_rec_trace = []
        # temporary variables for storing postsynaptic epsp traces
        self.epsp = numpy.zeros(self.nInputs+self.tsize)
        self.epsp2 = numpy.zeros(self.nInputs+self.tsize)
        # make nInputs accessible in SEMLiquid functions
        self.nInputs = self.nInputs

        # vector for output spikes
        self.Z = numpy.zeros(self.tsize+self.nReadouts)
        # vector for output spike probability
        self.Zp = numpy.ones(self.tsize+self.nReadouts)
        # firing rate of the network
        self.r = self.rmax*self.Zp/numpy.sum(self.Zp)  # Zp = e^{u(t)} (initialize firing rates uniformly)
        # state of the network (low-pass filtered network spike trains)
        self.state = numpy.zeros(self.nInputs+self.tsize)
        # weights
        self.W = numpy.zeros((self.tsize+self.nReadouts, 1+self.nInputs+self.tsize))
        # weight change
        self.dW = numpy.zeros(self.W.shape)

        # binary matrices indicating recurrent, input, bias, and readout weights
        self.rec_W = numpy.zeros(self.W.shape, dtype='int')
        self.rec_W[:self.tsize,1+self.nInputs:] = 1
        self.inp_W = numpy.zeros(self.W.shape, dtype='int')
        self.inp_W[:self.tsize,1:1+self.nInputs] = 1
        self.pr_W = numpy.zeros(self.W.shape, dtype='int')
        self.pr_W[:self.tsize,0] = 1
        self.rdt_W = numpy.zeros(self.W.shape, dtype='int')
        self.rdt_W[self.tsize:,1+self.nInputs:] = 1

        self.poiss_corr = numpy.zeros(self.tsize+self.nReadouts)
        self.alpha = 1.0
        self.beta = 1.0
        self.rho = 1.0  # TODO revert
        # self.rho = 0.0

        self.step = 0

        self.initialize_dynamic_synapses()

        # variance tracking parameters
        if self.use_variance_tracking:
            self.S = numpy.zeros(self.W.shape)
            self.Q = numpy.ones(self.W.shape)
            self.dS = numpy.zeros(self.W.shape)
            self.dQ = numpy.zeros(self.W.shape)

        # scale epsp to fit maximum or area to 1
        tau_frac = self.tau/self.tau2
        #self.epsp_scale = tau_frac**(self.tau2/(self.tau2 - self.tau)) - tau_frac**(self.tau/(self.tau2 - self.tau))
        #self.epsp_scale = (self.tau - self.tau2)/self.sigma
        self.epsp_scale = 1.0

        self.pca_dim = 3
        self.colors = ['r','g','b','m','c','y','k']
        self.ext = 'pdf'
        self.W_readout = dict()
        self.b_readout = dict()
        self.train_performance = dict()
        self.test_performance = dict()

        self.use_inputs = True
        self.use_noise = False
        self.do_train = True
        self.order_states = (self.task=='pattern' or self.task=='speech')
        self.order_states2 = False
        self.sub_neurons = 1
        self.sub_neurons_input = 1
        self.sub_time = 20
        self.isppt = True
        self.ms = 3
        self.plot_weights = True  # False

        self.filename = 'sem_liquid_v2.py'
        self.outputdir = os.environ["PWD"] + '/data/'
        if hasattr(datetime,'today'):
            today = datetime.today()
        else:
            today = datetime.datetime.today()
        self.outputdir += str(today.strftime("%Y-%m-%d_%H%M%S"))
        self.outputdir += self.task
        self.outputdir += "_%s_%d/" % (os.uname()[1].split('.')[0],self.seed)
        try:
            os.makedirs(self.outputdir)
        except:
            pass
        # shutil.copy(self.filename,self.outputdir+self.filename)
        logging.basicConfig(filename=self.outputdir+"output.log",
                            level=logging.DEBUG,
                            format="%(asctime)s %(message)s")
        logging.debug(simpars)
        self.figurelist = []

        print "Generating liquid of size %dx%d..." % (self.n, self.m)
        self.generate_connections()
        liquidstr = "%d SEMS with %d neurons, min size: %d, max size: %d, avg size: %f" % (self.n*self.m, self.tsize,
            numpy.min(self.sizes), numpy.max(self.sizes), numpy.mean(self.sizes))
        print liquidstr
        print self.sizes
        logging.debug(liquidstr)
        logging.debug("SEM sizes:\n"+str(self.sizes))
        self.analyze_connections()
        #print numpy.random.rand()
        #print numpy.random.rand()

    def ind2pos(self, i):
        if hasattr(i,'__iter__'):
            return [numpy.unravel_index(ii, self.size) for ii in i]
        else:
            return numpy.unravel_index(i, self.size)

    def pos2ind(self, pos):
        return pos[0]*self.size[1] + pos[1]

    def distance_probability(self, d):
        dmin = 1.0
        dmax = numpy.linalg.norm(numpy.array(self.size))
        dd = 0.01
        ds = numpy.arange(dmin, dmax, dd)
        p1 = self.lam*numpy.exp(-self.lam*ds)
        p2 = 2*numpy.pi*ds
        pd = p1/p2
        pd = pd/numpy.sum(pd)
        i = numpy.searchsorted(ds, d)
        return pd[i]

    def initialize_dynamic_synapses(self):
        if self.use_dynamic_synapses:
            self.U_mean = 0.5
            self.D_mean = 1.1
            self.F_mean = 0.05
            self.D_mean /= self.frac_tau_DS
            self.F_mean /= self.frac_tau_DS
            self.U = self.U_mean + self.U_mean/2*numpy.random.randn(self.nInputs+self.tsize) * 0  # TODO remove *0
            self.D = self.D_mean + self.D_mean/2*numpy.random.randn(self.nInputs+self.tsize) * 0
            self.F = self.F_mean + self.F_mean/2*numpy.random.randn(self.nInputs+self.tsize) * 0
            # TODO check the meaning of this: is STP disabled on input synapses?
            # TODO it appears so..
            if not self.use_dynamic_input_synapses:
                self.U[:self.nInputs] = 1.0
                self.D[:self.nInputs] = 0.0
                self.F[:self.nInputs] = 0.0
            self.rdyn = numpy.ones(self.nInputs+self.tsize)

            self.U = numpy.maximum(self.U, 0.0)
            self.D = numpy.maximum(self.D, self.dt)
            self.F = numpy.maximum(self.F, self.dt)
            self.udyn = self.U
            self.isi = numpy.zeros(self.nInputs+self.tsize)
        else:
            self.udyn = self.rdyn = 1
            self.isi = numpy.zeros(self.nInputs+self.tsize)

    def initialize_random_weights(self):
        tmp = numpy.random.rand(self.tsize+self.nReadouts,self.nInputs+self.tsize)*self.W_map
        tmp[tmp>0] = numpy.log(tmp[tmp>0])
        self.W[:,1:] = tmp

    def generate_connections(self):
        npos = self.npos
        self.distances = numpy.zeros((npos,npos))
        self.conn = numpy.zeros((npos,npos)).astype(int)
        self.inpconn = numpy.zeros((self.nInputs,npos)).astype(int)
        self.idmap = numpy.empty(self.size, dtype='object')
        self.group_map = []
        self.groups = numpy.zeros((self.tsize+self.nReadouts, self.tsize+self.nReadouts)).astype(int)
        self.W_map = numpy.zeros((self.tsize+self.nReadouts, self.nInputs+self.tsize)).astype(int)

        printf("Generating ids...\n")
        cur = 0
        for i in range(self.n):
            for j in range(self.m):
                self.idmap[i,j] = numpy.arange(cur, cur+self.sizes[i,j])
                self.group_map += [self.pos2ind((i,j))]*self.sizes[i,j]
                self.groups[cur:cur+self.sizes[i,j],cur:cur+self.sizes[i,j]] = 1
                cur += self.sizes[i,j]
        self.group_map = numpy.asarray(self.group_map)

        if self.use_recurrent_connections:
            printf("Calculating distances...  0%%")
            for i in range(npos):
                for j in range(npos):
                    pos1 = self.ind2pos(i)
                    pos2 = self.ind2pos(j)
                    # self.distances[i,j] = distance(pos1, pos2)
                    self.distances[i,j] = 0 # TODO @zbarni - this ensures an all to all conn
                    printf("\b\b\b\b%3d%%" % ((i*npos+j+1)*100.0/npos**2))
            printf("\b\b\b\b100%\n")

            printf("Generating %d recurrent connections...  0%%" % (self.nConn))
            eps = 1.0
            count = 0
            while count<self.nConn:
                d = numpy.random.exponential(scale=1.0/self.lam)
                I,J = numpy.where((self.distances<=d+eps) & (self.distances>d-eps))
                if len(I)>0:
                    i = numpy.random.randint(len(I))
                    if I[i]!=J[i]:
                        self.conn[I[i],J[i]] = 1
                        count = numpy.sum(self.conn)
                        printf("\b\b\b\b%3d%%" % (count*100.0/self.nConn))
            printf("\b\b\b\b100%\n")
            #self.conn[numpy.eye(npos).astype(int)] = 0
            for i,j in zip(*numpy.where(self.conn)):
                from_start, from_end = tuple(self.idmap[self.ind2pos(i)][[0,-1]])
                to_start, to_end = tuple(self.idmap[self.ind2pos(j)][[0,-1]])
                self.W_map[to_start:to_end+1, self.nInputs+from_start:self.nInputs+from_end+1] = 1  # ensures that autapses are not considered
                print "connect: %d-%d/%d-%d" % (to_start,to_end,self.nInputs+from_start,self.nInputs+from_end)

        input_sems = numpy.where(numpy.random.rand(self.npos)<=self.input_fraction)[0]
        printf("Generating input connections for %d inputs to %d SEMs...  0%%" % (self.nInputs, len(input_sems)))
        if self.Cinscale>=1.0:
            pcin = 1.0
        else:
            frac = self.Cinscale/(1.0-self.Cinscale)
            #nrec = float(self.nConn)/float(self.npos)
            nrec = numpy.mean(numpy.dot(self.conn[input_sems,:].T, self.sizes.flatten()[input_sems]))
            pcin = nrec*frac/float(self.nInputs)
            #print "pcin",pcin
        for i in range(self.nInputs):
            for j in input_sems:
                if numpy.random.rand()<=pcin:
                    self.inpconn[i,j] = 1
                    printf("\b\b\b\b%3d%%" % ((i*npos+j+1)*100.0/(self.nInputs*npos)))
        printf("\b\b\b\b100%\n")
        for i,j in zip(*numpy.where(self.inpconn)):
            to_start, to_end = tuple(self.idmap[self.ind2pos(j)][[0,-1]])
            self.W_map[to_start:to_end+1, i] = 1
            #print "connect: %d-%d/%d" % (to_start,to_end,i)

        if self.pConn<1.0:
            printf("Deleting recurrent connections (p=%f)...\n" % (self.pConn))
            for i,j in zip(*numpy.where(self.W_map[:self.tsize,self.nInputs:])):
                if numpy.random.rand()<=(1.0-self.pConn):
                    self.W_map[:self.tsize,self.nInputs:][i,j] = 0
                    #print "remove: %d/%d" % (i,self.nInputs+j)

        if self.use_self_recurrent_connections:
            for i in range(npos):
                from_start, from_end = tuple(self.idmap[self.ind2pos(i)][[0,-1]])
                to_start, to_end = tuple(self.idmap[self.ind2pos(i)][[0,-1]])
                self.W_map[to_start:to_end+1, self.nInputs+from_start:self.nInputs+from_end+1] = 1
                #print "connect: %d-%d/%d-%d" % (to_start,to_end,self.nInputs+from_start,self.nInputs+from_end)

        trained = numpy.zeros(self.tsize).astype(bool)
        trained_sems = numpy.where(numpy.random.rand(self.npos)<=self.train_fraction)[0]
        if len(trained_sems)>0:
            trained_ids = numpy.concatenate([self.idmap[self.ind2pos(pos)] for pos in trained_sems])
            trained[trained_ids] = True
        untrained = numpy.invert(trained)
        tmp = numpy.random.rand(untrained.sum(),self.nInputs+self.tsize)*self.W_map[untrained,:]
        #tmp = tmp-3.
        #tmp = (tmp.T/numpy.sum(tmp,axis=1)).T
        tmp[tmp>0] = numpy.log(tmp[tmp>0])
        #tmp[tmp>0] = (-5)*tmp[tmp>0]
        self.W[untrained,1:] = tmp
        #tmp = numpy.random.rand(untrained.sum(),self.nInputs)*self.W_map[untrained,:self.nInputs]
        #tmp = (-5)+1*tmp
        #self.W[untrained,1:1+self.nInputs] = tmp
        self.untrained = untrained
        #print "untrained:",self.untrained
        if not self.train_inputs:
            tmp = numpy.random.rand(self.W.shape[0],self.nInputs)
            tmp[tmp>0] = numpy.log(tmp[tmp>0])
            self.W[:,1:1+self.nInputs] = tmp
        if self.random_initial_weights:
            tmp = numpy.random.rand(*self.W_map.shape)*self.W_map
            tmp[tmp>0] = numpy.log(tmp[tmp>0])
            self.W[:,1:] = tmp

        printf("Generating readout connections...\n")
        self.W_map[self.tsize:,self.nInputs:] = 1
        #print "connect: %d-%d/%d-%d" % (self.tsize,self.tsize+self.nReadouts,self.nInputs,self.nInputs+self.tsize)
        self.groups[self.tsize:,self.tsize:] = 1

    def analyze_connections(self):
        npos = self.npos
        tmpstr = "total recurrent connections: %d\n" % (numpy.sum(self.W_map[:,self.nInputs:]))
        tmpstr += "total input connections: %d\n" % (numpy.sum(self.W_map[:,:self.nInputs]))
        #fanins = [numpy.dot(self.conn[:,i],self.sizes.flatten())+numpy.sum(self.inpconn[:,i]) for i in range(npos)]
        fanins = [numpy.sum(self.W_map[i,:]) for i in range(self.tsize)]
        tmpstr += "fan-ins: min %d, max %d, mean %f\n" % (numpy.min(fanins),numpy.max(fanins),numpy.mean(fanins))
        #fanouts = [numpy.dot(self.conn[i,:],self.sizes.flatten()) for i in range(npos)]
        fanouts = [numpy.sum(self.W_map[:self.tsize,self.nInputs+i]) for i in range(self.tsize)]
        tmpstr += "fan-outs: min %d, max %d, mean %f\n" % (numpy.min(fanouts),numpy.max(fanouts),numpy.mean(fanouts))
        #nin = [numpy.dot(self.conn[:,i],self.sizes.flatten())+numpy.sum(self.inpconn[:,i]) for i in range(npos)]
        #inpfracs = [float(numpy.sum(self.inpconn[:,i]))/float(nin[i]) for i in range(npos) if nin[i]>0]
        inpfracs = [float(numpy.sum(self.W_map[i,:self.nInputs]))/float(numpy.sum(self.W_map[i,:]))
            for i in range(self.tsize)]
        tmpstr += "input fractions: min %f, max %f, mean %f\n" % (numpy.min(inpfracs),numpy.max(inpfracs),numpy.mean(inpfracs))
        tmpstr += "%d/%d inputs used\n" % (numpy.sum(numpy.any(self.inpconn, axis=1)),self.nInputs)

        print tmpstr
        logging.debug('analyzing connections\n'+tmpstr)


    def reset_epsp(self):
        #old_epsp = self.epsp
        #old_epsp2 = self.epsp2
        #self.epsp[:self.nInputs] = numpy.mean(old_epsp[:self.nInputs])
        #self.epsp2[:self.nInputs] = numpy.mean(old_epsp2[:self.nInputs])
        #mean_epsp = numpy.sum(old_epsp[self.nInputs:])/self.npos
        #mean_epsp2 = numpy.sum(old_epsp2[self.nInputs:])/self.npos
        #self.epsp[self.nInputs:] = mean_epsp/numpy.sum(self.groups[:self.tsize,:self.tsize], axis=0)
        #self.epsp2[self.nInputs:] = mean_epsp2/numpy.sum(self.groups[:self.tsize,:self.tsize], axis=0)
        self.epsp = numpy.zeros(self.epsp.shape)
        self.epsp2 = numpy.zeros(self.epsp2.shape)
        self.state = numpy.zeros(self.state.shape)
        if self.use_dynamic_synapses:
            self.udyn = self.U
            self.rdyn = numpy.ones(self.rdyn.shape)
            self.isi = numpy.zeros(self.nInputs+self.tsize)

    # calculate input and recurrent entropy
    def calculate_entropies(self):
        u_inp = numpy.dot(self.W[:self.tsize,1:1+self.nInputs]* \
        self.W_map[:self.tsize,:self.nInputs]*self.rec_mod[:self.tsize,1:1+self.nInputs], self.Y[:self.nInputs])
        expU_inp = numpy.exp(u_inp - numpy.max(u_inp))
        self.Zp_inp = expU_inp / numpy.dot(self.groups[:self.tsize,:self.tsize], expU_inp)
        u_rec = numpy.dot(self.W[:self.tsize,1+self.nInputs:]* \
        self.W_map[:self.tsize,self.nInputs:]*self.rec_mod[:self.tsize,1+self.nInputs:], self.Y[self.nInputs:])
        expU_rec = numpy.exp(u_rec - numpy.max(u_rec))
        self.Zp_rec = expU_rec / numpy.dot(self.groups[:self.tsize,:self.tsize], expU_rec)
        self.H_inp = entropy(self.Zp_inp)
        self.H_rec = entropy(self.Zp_rec)
        #print self.H_inp, self.H_rec, self.Zp_inp.shape, self.Zp_rec.shape

    # this is the main method that advances the simulation by one time step
    # x is the input, t is the optional target vector for the readout WTAs
    def process_input(self, x, t=None):
        if t is not None:
            assert(len(t) == self.nReadouts)
        else:
            t = numpy.ones(self.nReadouts, dtype='bool')

        # calculate EPSP (unweighted!)
        self.Y = (self.epsp - self.epsp2)/self.epsp_scale
        # calculate membrane potential (priors + weighted epsps)
        self.rec_mod = self.inp_W + self.rdt_W + self.rho*self.rec_W
        self.u = self.W[:,0] + numpy.dot(self.W[:,1:]*self.W_map*self.rec_mod[:,1:], self.Y)
        #if self.u != [0.]:
        #    print "Breakpoint: self.u != 0."

        # for supervised WTA readout learning
        if t.sum()>1:
            self.u[self.tsize:][numpy.invert(t)] = -numpy.infty
        # calculate output probabilities
        expU = numpy.exp(self.u - numpy.max(self.u)) # TODO test exp(u) instead of exp(u-max(u))
        self.Zp = expU / numpy.dot(self.groups, expU)
        #self.Zp[:self.tsize] = (1-self.Znoise) * self.Zp[:self.tsize] + \
        #    self.Znoise/numpy.sum(self.groups[:self.tsize,:self.tsize], axis=0)
        # draw output spikes
        self.r = self.rmax*self.Zp  # normalizes firing rate
        self.r[self.tsize:] *= self.rmax_rdt*self.Zp[self.tsize:]
        self.Z = (numpy.random.exponential(1.0/self.r) <= self.dt).astype(int)

        # TODO revert  /remove - manual recurrent spikes
        if shared_params.use_fixed_spike_times:
            if isinstance(shared_params.output_spikes[0], list):
                # for each cell, set corresponding entry in self.Z to 1
                for neuron_idx, out_spikes in enumerate(list(itertools.chain(*shared_params.output_spikes))):
                    self.Z[neuron_idx] = int(self.step in out_spikes)
            else:
                if self.step in shared_params.output_spikes:  # TODO revert  /remove
                    self.Z = numpy.ones(self.Z.shape)
                else:
                    self.Z = numpy.zeros(self.Z.shape)

        self.calculate_entropies()

        # for supervised WTA readout learning
        if t.sum()==1:
            self.Z[self.tsize:] = t

        # update EPSPs
        y = numpy.concatenate((x,self.Z[:self.tsize]))
        if any(y) or any(x):
            print "Breakpoint"
        #y = numpy.concatenate((x,self.Zp[:self.tsize]))
        # calculate postsynaptic traces of presynaptic spikes
        #print y.shape, self.epsp.shape, self.rdyn.shape, self.udyn.shape
        # TODO @note STP contribution added directly to EPSP?
        self.epsp += y*self.rdyn*self.udyn - self.epsp*self.dt/self.tau
        self.epsp2 += y*self.rdyn*self.udyn - self.epsp2*self.dt/self.tau2
        self.state += y - self.state*self.dt/self.tau
        #self.state = (self.epsp - self.epsp2)/self.epsp_scale

        # calculate and apply the weight update
        if self.do_train:
            #if self.use_entropy_regularization:
            #    self.rho -= self.eta_rho*(self.H_inp - self.H_rec)
            self.W_mod = self.beta*self.inp_W + self.rdt_W + self.alpha*self.rec_W + self.pr_W
            if self.use_variance_tracking:
                self.etas = self.eta*(self.Q-self.S**2)/(numpy.exp(-self.S)+1)
            else:
                self.etas = self.eta*numpy.ones(self.W.shape)/self.step  # TODO why the division???
            P = numpy.exp(self.W)  # just takes the exponential of the weights: e^(-w_{ki})
            truncP = numpy.maximum(P,self.etas)
            self.dW[:,1:] = (self.Z * (self.Y - P[:,1:]).T).T / truncP[:,1:]

            if not self.use_inputs or not self.train_inputs:
                self.dW[:,1:self.nInputs+1] = 0
            self.dW[self.untrained,:] = 0

            # apply the weight update
            self.W += self.etas * self.W_mod * self.dW
            # print('etas: ', self.etas[0][-1])
            # print("from Y(t): \t\t{}".format(self.Y))

            # update adaptive learning rate
            if self.use_variance_tracking:
                # assert False
                self.dS[:,1:] = (self.Z * (self.W[:,1:] - self.S[:,1:]).T).T
                self.dQ[:,1:] = (self.Z * (self.W[:,1:]**2 - self.Q[:,1:]).T).T
                if self.use_priors:
                    self.dS[:,0] = self.W[:,0] - self.S[:,0]
                    self.dQ[:,0] = self.W[:,0]**2 - self.Q[:,0]
                if not self.use_inputs:
                    self.dS[:,1:self.nInputs+1] = 0
                    self.dQ[:,1:self.nInputs+1] = 0
                self.S += self.etas * self.W_mod * self.dS
                self.Q += self.etas * self.W_mod * self.dQ

        # simulate dynamic synapses
        if self.use_dynamic_synapses:
            #if self.use_multiple_synapses:
            #    r = self.rdyn[:,y==1]; u = self.udyn[:,y==1]; isi = self.isi[y==1]
            #    D = self.D[:,y==1]; U = self.U[:,y==1]; F = self.F[:,y==1]
            #    self.rdyn[:,y==1] = 1 + (r - r*u - 1)*numpy.exp(-isi/(D+eps))
            #    self.udyn[:,y==1] = U + u*(1-U)*numpy.exp(-isi/(F+eps))
            #    self.isi += self.dt
            #    self.isi[y==1] = 0
            #else:
            r = self.rdyn[y==1]; u = self.udyn[y==1]; isi = self.isi[y==1]
            D = self.D[y==1]; U = self.U[y==1]; F = self.F[y==1]
            self.rdyn[y==1] = 1 + (r - r*u - 1)*numpy.exp(-isi/(D+eps))  # TODO is this important? u_{k-1}
            self.udyn[y==1] = U + u*(1-U)*numpy.exp(-isi/(F+eps))
            self.isi += self.dt
            self.isi[y==1] = 0
        else:
            self.isi += self.dt
            self.isi[y == 1] = 0

        self.u_trace.append(copy.deepcopy(self.u))
        self.u_inp_trace.append(x)
        self.isi_trace.append(copy.deepcopy(self.isi))
        self.rdyn_trace.append(copy.deepcopy(self.rdyn))
        self.udyn_trace.append(copy.deepcopy(self.udyn))
        self.dEPSP_trace.append(copy.deepcopy(self.udyn*self.rdyn))
        self.epsp_trace.append(copy.deepcopy(self.epsp))
        self.epsp2_trace.append(copy.deepcopy(self.epsp2))
        self.w_inp_trace = numpy.append(self.w_inp_trace, numpy.array(self.W[0, 1:self.nInputs + 1]), axis=0)
        self.w_rec_trace = numpy.append(self.w_rec_trace, numpy.array(self.W[0, 1 + self.nInputs:]), axis=0)
        self.spk_out_trace.append(copy.deepcopy(self.Z))

        # print("")
        # print("Weights:")
        # print(self.W[:, 1:])
        # print('epsp', self.epsp)
        # print('epsp2', self.epsp2)

    def plotInfo(self, ax, fs=20, dstr=None):
        if dstr is None:
            dstr = "%s" % (self.outputdir)
        bbox = ax.get_position()
        pylab.figtext(bbox.xmax, bbox.ymax+0.005, dstr, fontsize=fs, ha='right', va='baseline')

    def update_parameters_train(self, step, nsteps):
        pass
        #nsteps_seq = int(self.tPattern*len(self.pattern_sequences[0])/self.dt)
        #if (step%nsteps_seq)==0:
            #self.reset_epsp()
        #if (step%nsteps_seq)<=self.alpha_reset_steps:
            #self.alpha=0.0
        #else:
            #self.alpha=1.0

    def get_order(self, p, I, t, r, tstart, nsteps):
        pt = pt2 = 0
        order = numpy.arange(self.tsize)
        times = numpy.zeros(order.shape)
        order2 = numpy.arange(self.tsize)
        times2 = numpy.zeros(order2.shape)
        if self.order_states:
            for pti,ptt in enumerate(p[::-1]):
                pt = ptt - tstart
                pi = numpy.max(I[pt])
                pl = t[-pti-1]
                if pi<0:
                    continue
                #pl = int(self.tPattern[pi]/self.dt)
                if pt+pl<=nsteps:
                    break
            Tord = numpy.arange(pt,pt+pl).astype(int)
            tmp = numpy.sum(r[Tord,:].T*numpy.exp(numpy.arange(pl)/(pl/(2*numpy.pi))*1j), axis=1)
            tmp /= numpy.sum(r[Tord,:].T, axis=1)
            angles = numpy.angle(tmp)
            angles[angles<0] += 2*numpy.pi
            weighted_rates_max_time = angles/(2*numpy.pi/pl)
            assert(weighted_rates_max_time.shape == (self.tsize,))
            order = numpy.argsort(weighted_rates_max_time)
            times = pt+weighted_rates_max_time[order]

            for pti2,ptt2 in enumerate(p[::-1]):
                pt2 = ptt2 - tstart
                pi2 = numpy.max(I[pt2])
                pl2 = t[-pti2-1]
                if pi2<0:
                    continue
                if pi2==pi:
                    continue
                #pl = int(self.tPattern[pi]/self.dt)
                if pt2+pl2<=nsteps:
                    break
            Tord2 = numpy.arange(pt2,pt2+pl2).astype(int)
            #print r[Tord,:]
            #print r[Tord2,:]
            tmp = numpy.sum(r[Tord2,:].T*numpy.exp(numpy.arange(pl2)/(pl2/(2*numpy.pi))*1j), axis=1)
            tmp /= numpy.sum(r[Tord2,:].T, axis=1)
            angles = numpy.angle(tmp)
            angles[angles<0] += 2*numpy.pi
            weighted_rates_max_time2 = angles/(2*numpy.pi/pl2)
            assert(weighted_rates_max_time2.shape == (self.tsize,))
            order2 = numpy.argsort(weighted_rates_max_time2)
            times2 = pt2+weighted_rates_max_time2[order2]
        else:
            Tord = numpy.arange(pt,pt).astype(int)
            Tord2 = numpy.arange(pt2,pt2).astype(int)
        return order,times,Tord,order2,times2,Tord2

    def simulate(self, Tsim, titlestr="learning", savestrprefix="sem_liquid_train", do_plot=True):
        nsteps = (int)(Tsim/self.dt)
        print "\n%s..." % (titlestr)
        logging.debug("%s...\n" % (titlestr))
        Ydict = dict(); Zdict = dict(); Idict = dict(); Idict2 = dict(); ZRdict = dict();
        rdict = dict(); ldict = dict(); pdict = dict(); tdict = dict(); stimdict = dict();
        rend = numpy.zeros((self.nstepsrec,self.tsize))
        spk_rec_times = range(0, nsteps-self.nstepsrec+1, self.dt_rec_spk)
        if (nsteps-self.nstepsrec) not in spk_rec_times:
            spk_rec_times.append(nsteps-self.nstepsrec)
        rec_start = 0

        num_weights_to_rec = self.tsize  # 50
        inp_weight_ids = numpy.where(self.W_map[:self.tsize,:self.nInputs].flatten())[0]
        # inp_weights_to_rec = numpy.random.permutation(inp_weight_ids)[:num_weights_to_rec]
        inp_weights_to_rec = inp_weight_ids[:num_weights_to_rec]
        num_fields_to_rec = num_weights_to_rec
        if self.use_recurrent_connections:
            rec_weight_ids = numpy.where(self.W_map[:self.tsize,self.nInputs:].flatten())[0]
            # rec_weights_to_rec = numpy.random.permutation(rec_weight_ids)[:num_weights_to_rec]
            rec_weights_to_rec = rec_weight_ids[:num_weights_to_rec]
            num_fields_to_rec += num_weights_to_rec
        if self.use_priors:
            prior_ids = numpy.arange(self.tsize)
            prior_to_rec = numpy.random.permutation(prior_ids)[:num_weights_to_rec]
            num_fields_to_rec += num_weights_to_rec
        if self.nReadouts > 0:
            rdt_weight_ids = numpy.where(self.W_map[self.tsize:,self.nInputs:].flatten())[0]
            rdt_weights_to_rec = numpy.random.permutation(rdt_weight_ids)[:num_weights_to_rec]
            num_fields_to_rec += num_weights_to_rec
        recorder = Recorder(nsteps/self.dt_rec+1, num_fields_to_rec)
        recorder.record(numpy.zeros(num_fields_to_rec))

        self.do_train = True
        self.inpgen.reset()
        start_time = timeit.default_timer()
        # for step in range(nsteps):  # TODO revert
        for step in range(shared_params.sim_time):
            #print numpy.random.rand()
            self.step += 1
            self.update_parameters_train(step, nsteps)  # does nothing (pass)
            t = None
            if not self.use_inputs:
                x = numpy.zeros(self.nInputs, dtype='int')
            else:
                assert shared_params.use_fixed_spike_times and isinstance(shared_params.input_spikes, list)
                if shared_params.use_fixed_spike_times:
                    x = numpy.zeros(self.nInputs, dtype='int')
                    assert self.nInputs == shared_params.n_inp_channels
                    for ch in range(self.nInputs):
                        x[ch] = int(step in shared_params.input_spikes[ch])
                    # x = [1] if step in shared_params.input_spikes else [0]  # manually set input spikes for testing
                else:
                    x = self.inpgen.generate(step)
                if not hasattr(self.inpgen.idx,'__iter__'):
                    t = numpy.asarray(numpy.arange(self.nReadouts)==self.inpgen.idx, 'bool')
                #t = numpy.atleast_1d(t)
                #t = None
            self.process_input(x, t)
            if (step+1)%self.dt_out==0:
                ratio = float(step+1)/float(nsteps)
                time = timeit.default_timer() - start_time
                printf('\r\n\nsimulating step %d/%d (%d%%) ETA %s...' % ((step+1)/self.dt_out,
                    nsteps/self.dt_out, ratio*100, format_time((1.0/ratio - 1)*time)))
            #if (step+1)%self.dt_rec==0:
            recorder.record(self.W[:self.tsize,1:1+self.nInputs].flatten()[inp_weights_to_rec])
            # TODO manual recording of input weights here
            recorder.continuous_inp_rec.append(copy.deepcopy(self.W[:self.tsize,1:1+self.nInputs]))

            if self.use_recurrent_connections:
                recorder.record(self.W[:self.tsize,1+self.nInputs:].flatten()[rec_weights_to_rec])
                # TODO manual recording of recurrent weights here
                # recorder.continuous_rec_rec[step, :num_weights_to_rec] = \
                #     self.W[:self.tsize,1+self.nInputs:].flatten()[rec_weights_to_rec]
                recorder.continuous_rec_rec.append(copy.deepcopy(self.W[:self.tsize,1+self.nInputs:]))

            if self.use_priors:
                recorder.record(self.W[:self.tsize,0].flatten()[prior_to_rec])
            if self.nReadouts > 0:
                recorder.record(self.W[self.tsize:,1+self.nInputs:].flatten()[rdt_weights_to_rec])
            recorder.stop_step()
            if step in spk_rec_times:
                if self.task=='multi':
                    self.reset_epsp()
                Ydict[step] = [[] for i in range(self.nInputs)]
                Zdict[step] = [[] for i in range(self.tsize)]
                ZRdict[step] = [[] for i in range(self.nReadouts)]
                #Idict[step] = numpy.zeros(self.nstepsrec, dtype='int')
                Idict[step] = []
                Idict2[step] = []
                rdict[step] = numpy.zeros((self.nstepsrec,self.nReadouts))
                ldict[step] = numpy.zeros((self.nstepsrec,4))
                pdict[step] = []
                tdict[step] = []
                stimdict[step] = []
                rec_start = step
            if step >= rec_start and step < rec_start + self.nstepsrec:
                t = step-rec_start
                for i in numpy.where(x)[0]:
                    Ydict[rec_start][i].append(t)
                for i in numpy.where(self.Z[:self.tsize])[0]:
                    Zdict[rec_start][i].append(t)
                for i in numpy.where(self.Z[self.tsize:])[0]:
                    ZRdict[rec_start][i].append(t)
                #Idict[rec_start][t] = self.inpgen.get_idx()
                Idict[rec_start].append(self.inpgen.idx)
                Idict2[rec_start].append(self.inpgen.get_idx())
                rdict[rec_start][t,:] = self.Zp[self.tsize:]
                ldict[rec_start][t,:] = [self.H_inp,self.H_rec,kld(self.Zp_rec,self.Zp_inp),self.rho]
                if self.task=='xor_pattern':
                    if hasattr(self.inpgen.inpgen,'last_pattern_start') and step==self.inpgen.inpgen.last_pattern_start:
                        pdict[rec_start].append(step)
                        tdict[rec_start].append(self.inpgen.inpgen.inpgen.get_current_pattern_sequence_length())
                if hasattr(self.inpgen,'last_pattern_start') and step==self.inpgen.last_pattern_start:
                    pdict[rec_start].append(step)
                    if self.embedded:
                        if self.task=='pattern':
                            tdict[rec_start].append(self.inpgen.inpgen.get_current_pattern_sequence_length())
                        if self.task=='speech':
                            tdict[rec_start].append(self.inpgen.inpgen.get_current_pattern_length())
                            stimdict[rec_start].append(self.inpgen.inpgen.stimulus.file)
                    else:
                        if self.task=='pattern':
                            tdict[rec_start].append(self.inpgen.get_current_pattern_sequence_length())
                        if self.task=='speech':
                            tdict[rec_start].append(self.inpgen.get_current_pattern_length())
                            stimdict[rec_start].append(self.inpgen.stimulus.file)
            if step>=nsteps-self.nstepsrec:
                t = step - (nsteps-self.nstepsrec)
                rend[t,:] = self.Zp[:self.tsize]
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        printf("\nsimulated %ds in %s real time\n" % (Tsim, format_time(elapsed_time)))
        logging.debug("simulated %ds in %s real time" % (Tsim, format_time(elapsed_time)))
        for key in Idict.keys():
            Idict[key] = numpy.asarray(Idict[key])
            Idict2[key] = numpy.asarray(Idict2[key])

        recorder.stop_recording()
        # inp_weights = recorder.read(num_weights_to_rec)
        # inp_weights = recorder.continuous_inp_rec[:, :num_weights_to_rec] # TODO
        inp_weights = numpy.array(recorder.continuous_inp_rec)

        if self.use_recurrent_connections:
            # rec_weights = recorder.read(num_weights_to_rec)
            # rec_weights = recorder.continuous_rec_rec[:, :num_weights_to_rec] # TODO
            rec_weights = numpy.array(recorder.continuous_rec_rec)
        if self.use_priors:
            priors = recorder.read(num_weights_to_rec)
        if self.nReadouts > 0:
            rdt_weights = recorder.read(num_weights_to_rec)
        ts = numpy.arange(0,Tsim+self.dt_rec*self.dt/2.0,self.dt_rec*self.dt)

        if do_plot:
            order, times, Tord, order2, times2, Tord2 = self.get_order(pdict[spk_rec_times[-1]], Idict[spk_rec_times[-1]],
                tdict[spk_rec_times[-1]], rend, spk_rec_times[-1], self.nstepsrec)
            sub_time = 20
            sub_neurons = 1
            for rec_step in spk_rec_times:
                pylab.figure(figsize=(12,9))
                ax = pylab.axes([0.125,0.73,0.775,0.17])
                plot_spike_trains2(Ydict[rec_step], self.nstepsrec, pylab.gca(), Idict2[rec_step])
                pylab.ylabel('input')
                pylab.title('%s: %ds-%ds' % (titlestr, rec_step*self.dt, (rec_step+self.nstepsrec)*self.dt))
                pylab.gca().set_xticklabels([])
                pylab.yticks(pylab.yticks()[0][1:])
                if self.task=='speech':
                    p = pdict[rec_step]; t = tdict[rec_step]; stims = stimdict[rec_step]; I = Idict[rec_step];
                    for ptt,pl,s in zip(p,t,stims):
                        if I[ptt-rec_step]>=0:
                            pylab.text(ptt-rec_step+0.5*pl, self.nInputs, s, ha="center", va="top", fontsize=10,
                                color=self.colors[I[ptt-rec_step]], weight='bold')
                pylab.axes([0.125,0.36,0.775,0.35])
                plot_spike_trains2([Zdict[rec_step][i] for i in order], self.nstepsrec, pylab.gca(),
                    self.group_map[order], sub_neurons=sub_neurons)
                if self.order_states and rec_step==spk_rec_times[-1]:
                    pylab.plot(times, numpy.arange(0,self.tsize), 'k')
                pylab.ylabel('network')
                pylab.gca().set_xticklabels([])
                pylab.yticks(pylab.yticks()[0][1:])
                pylab.ylim([0,self.tsize])
                pylab.axes([0.125,0.1,0.775,0.25])
                pylab.hold(True)
                pylab.plot(ldict[rec_step][:,0], 'r')
                pylab.plot(ldict[rec_step][:,1], 'b')
                pylab.plot(ldict[rec_step][:,2], 'k')
                pylab.ylabel('entr. [bit]')
                pylab.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
                ax2 = pylab.twinx()
                pylab.plot(ldict[rec_step][:,3], 'k--')
                pylab.ylim([-0.1,1.1])
                pylab.yticks([0.0,1.0])
                ax2.yaxis.tick_right()
                pylab.xlabel('time [ms]')
                pylab.ylabel(r'$\rho$')
                self.plotInfo(ax, fs=12)
                self.save_figure(self.outputdir+'%s_%02d.%s' % (savestrprefix,rec_step*self.dt,self.ext))

        if self.plot_weights and False:
            co = ['b','g','r','y','c','m','k']
            pylab.figure()
            pylab.subplots_adjust(top=0.9, bottom=0.15)
            pylab.hold(True)
            for i in range(inp_weights.shape[1]):
                pylab.plot(ts, inp_weights[:,i], c=co[i%len(co)])
            pylab.title('input weights')
            pylab.xlabel('time [s]')
            self.save_figure(self.outputdir+'%s_inp_weights.%s' % (savestrprefix,self.ext))

            if self.use_recurrent_connections:
                pylab.figure()
                pylab.subplots_adjust(top=0.9, bottom=0.15)
                for i in range(rec_weights.shape[1]):
                    pylab.plot(ts, rec_weights[:,i], c=co[i%len(co)])
                pylab.title('recurrent weights')
                pylab.xlabel('time [s]')
                self.save_figure(self.outputdir+'%s_rec_weights.%s' % (savestrprefix,self.ext))


        # Plot membrane potentials during simulation
        # print self.u_trace
        co = ['b','g','r','y','c','m','k']

        #print(len(self.u_trace[0]))
        n_neurons = len(self.u_trace[0])
        # fig, axes = plt.subplots(nrows=9, ncols=n_neurons, figsize=(25, 25))
        fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(10, 35))
        n_inp_neurons = 1
        results = {x: {} for x in ['Vms', 'spikes', 'inp_weights', 'rec_weights', 'epsps', 'inp_spikes']}

        for neuron in range(n_neurons):
            # if neuron + 1 not in shared_params.plot_neuron_ids:
            #     continue
            # plotting_vars = {'u_trace': [], 'u_inp_trace': [], 'spk_out_trace': [], 'epsp1': [], 'epsp2': []}
            plotting_vars = {'u_trace': [], 'u_inp_trace': [], 'spk_out_trace': [], 'epsp': [], 'udyn': [],
                             'rdyn': [], 'dEPSP': [], 'isi': []}
            for i in range(len(self.u_trace)):
                plotting_vars['u_trace'].append(self.u_trace[i][neuron])  # Voltage trace
                # plotting_vars['u_inp_trace'].append(self.u_inp_trace[i][neuron])  # Voltage trace
                plotting_vars['u_inp_trace'].append(self.u_inp_trace[i][0])  # input spikes
                plotting_vars['spk_out_trace'].append(self.spk_out_trace[i][neuron])  # output spikes
                # plotting_vars['epsp1'].append(self.epsp_trace[i][n_inp_neurons + neuron])  # EPSP1 of this neuron
                # plotting_vars['epsp2'].append(self.epsp2_trace[i][n_inp_neurons + neuron])  # EPSP2 of this neuron
                plotting_vars['epsp'].append(self.epsp_trace[i][n_inp_neurons + neuron] -
                                             self.epsp2_trace[i][n_inp_neurons + neuron])  # EPSP of this neuron (y(t))
                if self.use_dynamic_synapses:
                    plotting_vars['udyn'].append(self.udyn_trace[i][n_inp_neurons + neuron]) # R trace
                    plotting_vars['rdyn'].append(self.rdyn_trace[i][n_inp_neurons + neuron]) # u trace
                    plotting_vars['dEPSP'].append(self.dEPSP_trace[i][n_inp_neurons + neuron]) # calculated delta EPSP (u*R)
                else:
                    plotting_vars['udyn'].append(self.udyn)  # R trace
                    plotting_vars['rdyn'].append(self.rdyn)  # u trace
                    plotting_vars['dEPSP'].append(self.udyn*self.rdyn) # calculated delta EPSP (u*R)
                plotting_vars['isi'].append(self.isi_trace[i][n_inp_neurons + neuron])  # ISI trace

            axes_idx = 0
            time_axis = numpy.arange(1, len(plotting_vars['u_trace']) + 1, 1)

            results['Vms'][neuron] = (time_axis, plotting_vars['u_trace'])
            results['inp_spikes'][neuron] = (time_axis, plotting_vars['u_inp_trace'])
            results['spikes'][neuron] = (time_axis, plotting_vars['spk_out_trace'])
            results['epsps'][neuron] = (time_axis, plotting_vars['epsp'])


            for k in sorted(plotting_vars.keys()):
                if k in ['dEPSP', 'isi', 'udyn', 'rdyn']:
                    continue

                v = plotting_vars[k]
                ax = axes[axes_idx]
                axes_idx += 1

                ax.plot(time_axis, v, '-')
                if 'epsp' not in k:
                    title = k
                else:
                    title = 'Out EPSPs (y_i(t) of this)'
                #     ax.plot(time_axis, v)
                # else:
                #     trace_idx = int(k[-1]) - 1
                #     ax.plot(time_axis, v, color=['k', 'r'][trace_idx], ls=['-', '--'][trace_idx], label=k)
                #     if v == 'epsp2':
                #         ax.legend()

                ax.set_title(title)
                ax.set_xlim([0, shared_params.sim_time])
    
            # axes[axes_idx][neuron].plot(ts * 1e3, rec_weights[:, neuron], c=co[neuron % len(co)])
            for src in range(n_neurons):
                axes[axes_idx].plot(numpy.arange(1, len(rec_weights[:, neuron, src]) + 1),
                                            # ensure that only valid connections are plotted
                                            rec_weights[:, neuron, src]  * self.W_map[neuron, self.nInputs + src],
                                            c=co[src % len(co)], label='{} -> {}'.format(src, neuron))
            # axes[axes_idx].legend()
            axes[axes_idx].set_title('Recurrent weights')
            axes[axes_idx].set_xlim([0, shared_params.sim_time])
            axes_idx += 1
            # axes[axes_idx][neuron].plot(ts * 1e3, inp_weights[:, neuron], c=co[neuron % len(co)])
            # axes[axes_idx][neuron].set_title('Input weights')
            # axes[axes_idx][neuron].set_xlim([0, shared_params.sim_time])

            # for i in range(1): # range(inp_weights.shape[1]):
            #     pylab.plot(ts * 1e3, inp_weights[:, i], c=co[i % len(co)])
            # pylab.title('Input weights')
            # axes.append(ax)

        results['inp_weights'] = (time_axis, inp_weights)
        # ensure that we only take into account the valid weights / connections
        # results['rec_weights'] = (time_axis, np.multiply(rec_weights, self.W_map[:, self.nInputs:]))
        results['rec_weights'] = (time_axis, rec_weights)

        import pickle
        with open('../klampfl_data.pkl', 'wb') as f:
            pickle.dump(results, f)

        fig.tight_layout()
        fig.savefig('voltage_trace.pdf')
        # plt.show()

    def update_parameters_test(self, step, nsteps):
        pass
        #if self.reset_input_test:
            #self.use_inputs = (step<=1000)
            #self.test_inpgen.set_noise(not (step<=400))
        #if step<=self.alpha_reset_steps:
            #self.alpha=0.0
        #else:
            #self.alpha=1.0

    def test(self, Tsim, itest, start=0, plot_readout_spikes=False, train_readouts=True,
             titlestr="testing", savestrprefix="sem_liquid_test", do_plot=True, do_save=False):
        self.do_train = False
        nsteps = (int)(Tsim/self.dt)
        print "\n%s..." % (titlestr)
        logging.debug("%s...\n" % (titlestr))
        Y = [[] for i in range(self.nInputs)] # create [[], [], ..., []] (nInputs-times)
        Z = [[] for i in range(self.tsize)] # create [[], [], ..., []] (tsize-times)
        ZR = [[] for i in range(self.nReadouts)] # create [[], [], ..., []] (nReadouts-times)
        I = []
        Is = []
        I2 = []
        r = numpy.zeros((nsteps,self.tsize)) # create a nsteps x tsize matrix
        ri = numpy.zeros((nsteps,self.nInputs))
        X = numpy.zeros((nsteps,self.tsize))
        Xs = numpy.zeros((nsteps,self.nInputs))
        rzr = numpy.zeros((nsteps,self.nReadouts))
        p = []; t = []; stims = [];

        #if self.task=='multi':
            #self.reset_epsp()
        self.reset_epsp()
        #if itest>0:
            #self.test_inpgen.reset()
        self.test_inpgen.reset()
        if self.task=='speech':
            self.test_inpgen.inpgen.next_idx = 1-(itest % 2)
        elif self.task=='pattern' and self.embedded:
            self.test_inpgen.inpgen.next_idx = (itest % 2)
            #self.test_inpgen.inpgen.next_idx = 0
        start_time = timeit.default_timer()
        for step in range(nsteps):
            #if self.task=='multi' and (step%self.nstepsrec)==0:
                #self.reset_epsp()
            #self.step += 1
            self.update_parameters_test(step, nsteps)  # just passes, ignore
            if not self.use_inputs:
                x = numpy.zeros(self.nInputs, dtype='int')
            else:
                x = self.test_inpgen.generate(step)
            self.process_input(x)
            if (step+1)%self.dt_out==0:
                ratio = float(step+1)/float(nsteps)
                time = timeit.default_timer() - start_time
                printf('\rtesting step %d/%d (%d%%) ETA %s...' % ((step+1)/self.dt_out,
                    nsteps/self.dt_out, ratio*100, format_time((1.0/ratio - 1)*time)))
            for i in numpy.where(x)[0]: # mpy.where(x)[0] returns the first index where x is True
                Y[i].append(step)
            for i in numpy.where(self.Z[:self.tsize])[0]:
                Z[i].append(step)
            for i in numpy.where(self.Z[self.tsize:])[0]:
                ZR[i].append(step)
            # I.append(self.test_inpgen.idx)
            I.append(self.test_inpgen.get_idx())  # @change
            Is.append(self.test_inpgen.get_idx())
            if self.task=='pattern':
                I2.append(self.test_inpgen.get_sidx())
            else:
                I2.append(self.test_inpgen.get_idx())
            r[step,:] = self.r[:self.tsize]
            if self.task=='multi':
                ri[step,:] = self.test_inpgen.num_spikes
                #ri[step,:] = self.test_inpgen.current_rates
            X[step,:] = self.state[self.nInputs:]
            Xs[step,:] = self.state[:self.nInputs]
            rzr[step,:] = self.Zp[self.tsize:]
            if hasattr(self.test_inpgen,'last_pattern_start') and step==self.test_inpgen.last_pattern_start:
                p.append(step)
                if self.embedded:
                    if self.task=='pattern':
                        t.append(self.test_inpgen.inpgen.get_current_pattern_sequence_length())
                    if self.task=='speech':
                        t.append(self.test_inpgen.inpgen.get_current_pattern_length())
                        stims.append(self.test_inpgen.inpgen.stimulus.digit)
                else:
                    if self.task=='pattern':
                        t.append(self.test_inpgen.get_current_pattern_sequence_length())
                    if self.task=='speech':
                        t.append(self.test_inpgen.get_current_pattern_length())
                        stims.append(self.test_inpgen.stimulus.digit)
            if self.task=='xor_pattern' and hasattr(self.test_inpgen.inpgen,'last_pattern_start') and step==self.test_inpgen.inpgen.last_pattern_start:
                p.append(step)
                t.append(self.test_inpgen.inpgen.inpgen.get_current_pattern_sequence_length())
        end_time = timeit.default_timer()
        elapsed_time = end_time - start_time
        printf("\ntested %ds in %s real time\n" % (Tsim, format_time(elapsed_time)))
        logging.debug("tested %ds in %s real time" % (Tsim, format_time(elapsed_time)))
        I = numpy.asarray(I)
        I2 = numpy.asarray(I2)
        Is = numpy.asarray(Is)

        target = {'speech':I2,'pattern':I2,'xor':I2,'xor_pattern':I2,'multi':ri}[self.task]
        T,rdt_out,trs,perf = self.apply_readouts("network", train_readouts, X, target)
        T_s,rdt_out_s,trs_s,perf_s = self.apply_readouts("stimulus", train_readouts, Xs, target)
        num_rdt = len(perf)

        if (self.task=='xor_pattern' or self.task=='pattern') and not train_readouts:
            spike_corr = spike_correlation(r, p, t, I2, r_thresh=90)
            self.test_performance["network"][-1].append(spike_corr[0])
            self.test_performance["network"][-1].append(spike_corr[1])
            print self.test_performance["network"][-1]

        #print trs, trs.shape

        sub_time, sub_neurons, sub_neurons_input = self.sub_time, self.sub_neurons, self.sub_neurons_input
        ms = self.ms
        isppt = self.isppt
        d = dict()
        if do_save:
            d = shelve.open(self.outputdir+'%s.shelve' % (savestrprefix))
            d["X"] = X
            d["Y"] = Y
            d["I"] = I
            d["Is"] = Is
            d["I2"] = I2
            d["Z"] = Z
            d["group_map"] = self.group_map
            d["r"] = r
            d["rzr"] = rzr
            d["T"] = T
            d["rdt_out"] = rdt_out
            d["p"] = p
            d["t"] = t
        if do_plot:
            order, times, Tord, order2, times2, Tord2 = self.get_order(p, I2, t, r, 0, nsteps)

            if do_save:
                d["order"] = order
                d["times"] = times
                d["order2"] = order2
                d["times2"] = times2
            #pts = []
            #for pti,ptt in enumerate(p[::-1]):
                #pt = ptt
                #pi = numpy.max(I2[pt])
                #pl = t[-pti-1]
                #if pi<0:
                    #continue
                #if pt+pl<=nsteps:
                    #pts.append(numpy.arange(pt,pt+pl).astype(int))
                #if len(pts)==2:
                    #break
            #resp1 = r[pts[0],:]
            #resp2 = r[pts[1],:]
            #d = shelve.open("corr_test.shelve")
            #d["response%02d_lim1"%(itest)] = resp1
            #d["response%02d_lim2"%(itest)] = resp2
            #d["response%02d"%(itest)] = r
            #d["spikes%02d"%(itest)] = Z
            #d["order%02d"%(itest)] = order
            #d["Tord%02d"%(itest)] = pts[0]
            #d["Tord2%02d"%(itest)] = pts[1]
            #d.close()

            if self.swap_order:
                order, order2, times, times2 = order2, order, times2, times
            #pylab.figure(figsize=(12,9))
            if isppt:
                pylab.figure(figsize=(8,4))
                ax = pylab.axes([0.125,0.67,0.775,0.2])
            else:
                #pylab.figure(figsize=(12,9))
                pylab.figure(figsize=(8,6))
                ax = pylab.axes([0.125,0.73,0.775,0.17])
            #pylab.subplot(4,1,1)
            plot_spike_trains2(Y, nsteps, pylab.gca(), Is, ms=ms, sub_neurons=sub_neurons_input, patterns=(p,t))
            if isppt:
                pylab.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(2))
            pylab.gca().set_xticklabels([])
            if self.task=='multi':
                pylab.gca().set_yticklabels([])
            pylab.ylabel('input')
            pylab.yticks(pylab.yticks()[0][1:])
            #pylab.title(titlestr)
            pylab.figtext(0.5, 0.95, titlestr, va='center', ha='center')
            if self.task=='speech':
                for pt,pl,s in zip(p,t,stims):
                    if I[pt]>=0 and pt+pl/2<=nsteps:
                        pylab.text(pt+0.5*pl, self.nInputs, "digit %d"%(s), ha="center", va="bottom", fontsize=16,
                            color=self.colors[I[pt]], weight='bold')
            if not self.task=='multi':
                if isppt:
                    pylab.axes([0.125,0.17,0.775,0.48])
                else:
                    if self.order_states2:
                        pylab.axes([0.125,0.42,0.775,0.3])
                    else:
                        pylab.axes([0.125,0.37,0.775,0.35])
                pylab.hold(True)
                plot_spike_trains2([Z[i] for i in order], nsteps, pylab.gca(), self.group_map[order],
                    sub_neurons=sub_neurons, ms=ms)
                if self.plot_order_states:
                    pylab.plot(times, numpy.arange(self.tsize), 'k')
                pylab.ylabel('network')
                if isppt:
                    pylab.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
                pylab.yticks(pylab.yticks()[0][1:])
                pylab.ylim([0,self.tsize])
            if not isppt:
                pylab.gca().set_xticklabels([])
                if self.task=='multi':
                    #pylab.axes([0.125,0.17,0.775,0.48])
                    #pylab.axes([0.125,0.1,0.775,0.25])
                    #pylab.hold(True)
                    labels = [r"$r_1(t)$", r"$r_2(t)$", r"$(r_1\cdot{}r_2)(t)$", r"$(r_1\cdot{}r_2)(t-\tau)$"]
                    pylab.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.7)
                    for ri in range(num_rdt):
                        pylab.subplot(num_rdt,1,ri+1)
                        pylab.hold(True)
                        pylab.plot(trs, T[trs,ri], 'k--')
                        #pylab.plot(trs, T_s[trs,ri], 'k-.')
                        pylab.plot(trs, rdt_out[:,ri], 'k-')
                        pylab.plot(trs, rdt_out_s[:,ri], 'k:')
                        pylab.ylabel(labels[ri])
                        #if ri==2:
                            #pylab.ylim([0,self.in_rate_lims[1]**2])
                        #else:
                            #pylab.ylim([0,self.in_rate_lims[1]+10])
                        pylab.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
                        #pylab.gca().set_yticklabels([])
                        if not ri==num_rdt-1:
                            pylab.gca().set_xticklabels([])
                    #pylab.yticks(numpy.arange(T.shape[1])+0.5, labels) #[r"$f_%d(t)$" % (i+1) for i in range(T.shape[1])])
                    #pylab.ylim([-0.5,T.shape[1]+0.5])
                elif self.order_states2:
                    #print "order",order
                    #print "order2",order2
                    pylab.axes([0.125,0.1,0.775,0.3])
                    pylab.hold(True)
                    plot_spike_trains2([Z[i] for i in order2], nsteps, pylab.gca(), self.group_map[order2],
                        sub_neurons=sub_neurons, ms=ms)
                    if self.plot_order_states:
                        pylab.plot(times2, numpy.arange(self.tsize), 'k')
                    pylab.ylabel('network')
                    if isppt:
                        pylab.gca().yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
                    pylab.yticks(pylab.yticks()[0][1:])
                    pylab.ylim([0,self.tsize])
                elif self.task=='pattern' or self.task=='speech' or self.task=='xor':
                    if self.nReadouts > 0:
                        pylab.axes([0.125,0.24,0.775,0.12])
                        pylab.hold(True)
                        #hlines=range(0,self.nReadouts,self.nReadouts_per_pattern)
                        #hlines=numpy.cumsum(self.nReadouts_per_pattern)
                        pylab.gca().set_xticklabels([])
                        #pylab.yticks(hlines)
                        readout_rates = numpy.mean(numpy.reshape(rzr, (nsteps/sub_time,sub_time,self.nReadouts)), axis=1)
                        #print readout_rates.shape
                        pylab.pcolor(numpy.arange(0,nsteps+1,sub_time), numpy.arange(self.nReadouts+1), readout_rates.T,
                            cmap=matplotlib.cm.binary, vmin=0, vmax=1)
                        #pylab.plot(numpy.arange(0,nsteps,sub_time),readout_rates)
                        if plot_readout_spikes:
                            plot_spike_trains2(ZR, nsteps, pylab.gca(), I)
                        #else:
                            #for hl in hlines:
                                #pylab.axhline(hl, 0, 1, c='k', ls=':')
                        pylab.ylabel('SEM')
                        pylab.yticks(numpy.arange(self.nReadouts)+0.5, range(1,self.nReadouts+1))
                        pylab.ylim([0,self.nReadouts])
                        #pylab.ylim([0,1])
                        pylab.xlim([0,nsteps])
                        #pylab.yticks([0,1])
                        #pylab.axes([0.125,0.1,0.775,0.15])
                        ##rdt_range = numpy.arange(numpy.sum(self.nReadouts_per_pattern[:itest]),
                                                ##numpy.sum(self.nReadouts_per_pattern[:itest+1])).astype(int)
                        #rdt_range = numpy.arange(numpy.sum(self.nReadouts_per_pattern[self.pattern_sequences[0]]),
                                                #self.nReadouts)
                        #pylab.plot(numpy.sum(r2[:,rdt_range], axis=1)/numpy.sum(r2, axis=1))
                        #pylab.ylim([-0.1,1.1])
                        #pylab.yticks([0.0,1.0])
                        #pylab.ylabel(r'$P_{class}$')
                        pylab.axes([0.125,0.1,0.775,0.12])
                    else:
                        pylab.axes([0.125,0.1,0.775,0.25])
                    pylab.hold(True)
                    for ri in range(num_rdt):
                        #trs = numpy.arange(nsteps)
                        pylab.plot(trs, ri+0.4*T[trs,ri], ls='--', c='k')
                        pylab.plot(trs, ri+0.4*rdt_out[trs,ri]/numpy.abs(rdt_out[trs,ri]).max(), ls='-', c='k')
                        #pylab.plot(trs, ri+0.4*rdt_out[trs,ri], ls='-', c='k')
                        #pylab.plot(trs, ri+0.4*rdt_out_s[trs,ri]/numpy.abs(rdt_out[trs,ri]).max(), ls='-.', c='k')
                        pylab.plot([trs[0],trs[-1]], [ri,ri], 'k:')
                    if self.task=='speech':
                        pylab.ylabel('lin. reg.')
                        if len(self.rev_digits)==0:
                            pylab.yticks([-0.4,0,0.4],["d%d" % (self.digits[0]), "0", "d%d" % (self.digits[1])])
                        else:
                            pylab.yticks([-0.4,0,0.4],["d%d" % (self.digits[0]), "0", "d%drev" % (self.rev_digits[0])])
                        #pylab.yticks(numpy.arange(T.shape[1]), ["digit %d" % (d) for d in self.digits])
                        ids = [0,-1,1]
                        for i,label in zip(ids,pylab.gca().get_yticklabels()):
                            label.set_color(self.colors[i])
                    #else:
                        #pylab.yticks(numpy.arange(T.shape[1]), [r"$f_%d(t)$" % (i+1) for i in range(T.shape[1])])
                    pylab.ylim([-0.5,num_rdt-0.5])
            pylab.xlabel('time [ms]')
            #self.plotInfo(ax, fs=12)
            self.save_figure(self.outputdir+'%s.%s' % (savestrprefix,self.ext))

        if do_save:
            d.close()

        # Plot Input
        # fig = plt.figure()
        # fig, ax = plt.subplots()
        # plt_list = []
        # for i in range(len(self.epsp_trace)):
        #     # Get maximum spike input at time i and append to list
        #     max_inp = 0.0
        #     for j in range(len(self.u_inp_trace[0])):
        #         if self.epsp_trace[i][j] * self.u_inp_trace[i][j] > max_inp:
        #             max_inp = self.epsp_trace[i][j] * self.u_inp_trace[i][j]
        #     if max_inp > 0.0:
        #         plt_list.append(max_inp)
        # #ax.plot(plt_list)
        # print self.u_trace
        # ax.plot(self.u_trace)
        # #plt.plot(plt_list1)
        # #plt.plot(plt_list2)
        # plt.show()

        return X,I,perf

    def save_figure(self, savestr):
        pylab.savefig(savestr)
        self.figurelist.append('%s' % (savestr))

    def concatenate_pdf(self, output="figures.pdf", delete=True):
        delete=False  # @change - don't remove any figure
        #pdffiles = [f for f in dircache.listdir(self.outputdir) if f[:-4]=='.pdf']
        pdffiles = ' '.join(self.figurelist)
        outputfile = self.outputdir+output
        # command = "ssh figipc66 pdftk %s cat output %s" % (pdffiles,outputfile)
        command = "pdftk %s cat output %s" % (pdffiles,outputfile)  # @change
        try:
            print command
            os.system(command)
            if delete:
                for file in self.figurelist:
                    command = "rm %s" % (file)
                    print command
                    os.system(command)
        except Exception as e:
            print("Error: " + str(e))


    def plot_weight_distribution(self, savestrprefix="sem_liquid"):
        rec_weight_ids = numpy.where(self.W_map[:self.tsize,self.nInputs:].flatten())[0]
        weights = self.W[:self.tsize,1+self.nInputs:].flatten()[rec_weight_ids]
        pylab.figure()
        pylab.hist(weights, bins=numpy.arange(-5,0,0.1))
        pylab.title("histogram of recurrent weights")
        self.save_figure(self.outputdir+'%s_recwhist.%s' % (savestrprefix,self.ext))
        inp_weight_ids = numpy.where(self.W_map[:self.tsize,:self.nInputs].flatten())[0]
        weights = self.W[:self.tsize,1:1+self.nInputs].flatten()[inp_weight_ids]
        pylab.figure()
        pylab.hist(weights, bins=numpy.arange(-5,0,0.1))
        pylab.title("histogram of input weights")
        pylab.title("histogram of input weights")
        self.save_figure(self.outputdir+'%s_inpwhist.%s' % (savestrprefix,self.ext))
        pylab.show()


# returns liquid computing model and test times for a given seed (seed) and a number of parameters (*p)
def sem_liquid_pattern1(seed=None, *p):
    strain = 1  # 100  # training set length (seconds)
    # strain = 10
    stest_train = 3
    stest = 3
    params = SEMLiquidParams(task='pattern',
                             nInputs=1,
                             pattern_mode='random_switching',
                             sprob=0.5,
                             nPatterns=1,
                             rin=5,
                             tPattern=[300e-3]*1,
                             use_priors=False,
                             plot_order_states=False,
                             frac_tau_DS=10,
                             ###########################################################################################
                             # Toggle this for testing STP, also remind to set spiketimes in shared_params so that STP is visible
                             use_dynamic_synapses=False,  # meaning: use_stp
                             # NOTE: fixed spiketimes from shared_params don't work
                             ###########################################################################################
                             rNoise=0,
                             use_noise=False,
                             use_variance_tracking=shared_params.use_variance_tracking, # meaning: use_variance_tracking
                             pattern_sequences=[[0],[0]],
                             test_pattern_sequences=[[0],[0]],
                             seed=seed,
                             # seed=99476,
                             size=(1,2),
                             pConn=1.0,
                             tNoiseRange=[300e-3,500e-3],
                             test_tNoiseRange=[300e-3, 800e-3],
                             size_lims=[shared_params.n_neurons, shared_params.n_neurons],
                             test_time_warp_range=(0.5,2.),
                             nstepsrec=strain*1000, # number of steps the recording lasts for
                             dt_rec=1,  # recording rate/Abtastrate of recorder
                             plot_weights=False,
                             )
    liquid = SEMLiquid(params)
    liquid.order_states2 = False
    liquid.sub_neurons = 4
    liquid.isppt = True
    liquid.ext = 'pdf'

    # shared_params.sim_time = strain * 1000
    strain = max(1, int(shared_params.sim_time / 1000))

    num_train_periods = 1
    num_test_periods = 10
    num_trial_periods = 1  # 100
    test_times = numpy.arange(0,num_train_periods*strain+strain/2.0,strain) # aka test_times = [0]

    itest = 0
    #test_titlestr = "response to patterns before training"
    #liquid.test_inpgen.inpgen.time_warp_range=(1.,1.)
    #X,I,perf = liquid.test(stest, 2*itest+1, titlestr=test_titlestr,
    #    savestrprefix="sem_liquid_test_pre_train%d_inp"%(2*itest+1), plot_readout_spikes=False, train_readouts=False, do_plot=True, do_save=True)
    #X,I,perf = liquid.test(stest, 2*itest+2, titlestr=test_titlestr,
    #    savestrprefix="sem_liquid_test_pre_train%d_inp"%(2*itest+2), plot_readout_spikes=False, train_readouts=False, do_plot=True, do_save=True)
    liquid.test_inpgen.inpgen.time_warp_range=(0.5,2.)

    for itest in range(1,num_train_periods+1):
        liquid.simulate(strain, titlestr="/simulating phase #%d" %(itest+1),
            savestrprefix="sem_liquid_train%d" % (itest+1), do_plot=False)
        #liquid.plot_weight_distribution()

        states = []
        ids = []
        test_titlestr = "response to patterns after %ds training" % ((itest+1)*strain)
        # X,I,perf = liquid.test(stest_train, itest, titlestr="readout train phase after %ds training" % ((itest+1)*strain),
        #     savestrprefix="sem_liquid_test%d_train_inp"%(itest), plot_readout_spikes=False, train_readouts=True, do_plot=True, do_save=True)
        exit()
    for itest in range(1,num_test_periods):
        X,I,perf = liquid.test(stest, 2*itest+1, titlestr=test_titlestr,
            savestrprefix="sem_liquid_test%d_inp"%(2*itest+1), plot_readout_spikes=False, train_readouts=False, do_plot=True, do_save=True)
        #states.append(X)
        #ids.append(I)
        X,I,perf = liquid.test(stest, 2*itest+2, titlestr=test_titlestr,
            savestrprefix="sem_liquid_test%d_inp"%(2*itest+2), plot_readout_spikes=False, train_readouts=False, do_plot=True, do_save=True)
        #states.append(X)
        #ids.append(I)

    liquid.test_inpgen.inpgen.time_warp_range=(1.,1.)
    for itest in range(num_trial_periods):
        X,I,perf = liquid.test(stest, itest, titlestr="Trial #%d"%(itest),
            savestrprefix="sem_liquid_trials%d"%(itest), plot_readout_spikes=False, train_readouts=False, do_plot=True, do_save=True)
    #states = numpy.concatenate(states)
    #ids = numpy.concatenate(ids)
    #liquid.trainPCA(states)
    #liquid.plotPCA(states,ids, savestrprefix="sem_liquid_test")
    liquid.plot_weight_distribution()
    if liquid.ext=='pdf':
        liquid.concatenate_pdf(delete=True)
    #pylab.show()
    pylab.close('all')
    return liquid, test_times


def stdp_window(seed=None, *p):
    pass
    # TODO: 1) Set pre_spike_time fixed to 100
    #  2) iterate post_spike_time from 25 to 175
    #  3) run_network for each combination of pre_spike_time-post_spike_time


if __name__ == '__main__':

    matplotlib.rcParams['font.size'] = 16.0

    task = sys.argv[1]
    ntrials = 1
    if len(sys.argv)>2:
        ntrials = (int)(sys.argv[2])
    seeds = [None]*ntrials
    params = [None]*ntrials

    if task == 'pattern1':
        seeds = [13521]
        #seeds = [49856] orig
        fcn = sem_liquid_pattern1
    else:
        raise Exception("Task %s not implemented" % (task))

    ntrials = len(seeds)
    seeds_dict = dict()
    perfs_dict = dict()
    params_dict = dict()
    for i,seed in enumerate(seeds):
        print
        print "TRIAL %d/%d" % (i+1,ntrials)
        print
        liquid,test_times = fcn(seed, params[i])
        seeds_dict[liquid.outputdir] = liquid.seed
        perfs_dict[liquid.outputdir] = liquid.test_performance
        params_dict[liquid.outputdir] = params[i]
    print
    keys = numpy.sort(seeds_dict.keys())

