
import numpy

class Recorder(object):
    def __init__(self, num_steps, num_fields):
        self.num_fields = num_fields  # Dimension of one slice of recorded data
        self.num_steps = num_steps  # Number of steps to record
        self.rec = numpy.zeros((num_steps,num_fields))
        self.idx = 0
        self.step = 0

        self.continuous_inp_rec = []
        self.continuous_rec_rec = []

    def record(self, values):
        values = numpy.asarray(values)
        if values.ndim==0:
            nvalues=1
        else:
            assert(values.ndim == 1)
            nvalues = len(values)
        if self.idx+nvalues>self.num_fields:
            values = values[:self.num_fields-self.idx]
            nvalues = len(values)
        self.rec[self.step, self.idx:self.idx+nvalues] = values
        self.idx += nvalues
        if self.idx >= self.num_fields:
            self.idx = 0
            self.step += 1

    def stop_step(self):
        self.idx = 0

    def stop_recording(self):
        self.idx = 0
        self.step = 0

    def read(self, nvalues):
        result = self.rec[:, self.idx:self.idx+nvalues]
        self.idx += nvalues
        return result
