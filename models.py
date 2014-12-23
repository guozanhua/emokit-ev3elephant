from utils import *
import cPickle
import numpy as np
import theano
import theano.tensor as T

# MODEL

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                np.random.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):

    def __init__(self, n_in, layer_sizes, n_out, shift=0, activation=None):
        self.input = T.fmatrix()
        self.target = T.fmatrix()
        self.inputs = [self.input]


        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function

        n_previous = n_in
        previous_output = self.input
        self.layers = []

        for n_hidden in layer_sizes:
            self.layers.append(HiddenLayer(
                input=previous_output,
                n_in=n_previous,
                n_out=n_hidden,
                activation=activation
            ))
            n_previous=n_hidden
            previous_output = self.layers[-1].output


        # The logistic regression layer gets as input the hidden units of the hidden layer
        self.outputLayer = HiddenLayer(
            input=previous_output,
            n_in=n_previous,
            n_out=n_out,
            activation=activation
        )

        self.output = self.outputLayer.output
        self.layers.append(self.outputLayer)

        self.params = sum([l.params for l in self.layers], []) # self.hiddenLayer.params + self.outputLayer.params

        ## We need to rescale to get correct MAPE error
        self.cost = T.mean(T.mean(T.reshape(abs(self.target - self.output) / abs(self.target+ shift),(-1,)), axis=0))

        self.monitors = []

    def get_params(self):
        return [p.get_value(borrow=True) for p in self.params]

    def set_params(self, params):
        for i,p in enumerate(params):
            self.params[i].set_value(p)

    def save_params(self, filename):
        cPickle.dump(self.get_params(), open(filename, "wb"))

    def load_params(self, filename):
        params = cPickle.load(open(filename, "rb"))
        self.set_params(params)

# TRAINERS

class SGD(object):
    '''This is a base class for all trainers.'''

    def __init__(self, network, profile=False, lr=0.3, L1=0, momentum=0.4, L2=0.00005, compile=True):
        self.profile = profile

        self.network = network

        self.lr = np.float32(lr)
        self.momentum = np.float32(momentum)

        self.params = network.params

        self.cost = network.cost + np.float32(L2)*T.sum([(p**2).sum() for p in self.params]) + np.float32(L1)*T.sum([abs(p).sum() for p in self.params])
        self.grads = T.grad(self.cost, self.params)

        # Expressions evaluated for training
        self.cost_exprs = [self.cost, network.cost]
        self.cost_names = ['L2 cost', "Network cost"]
        for name, monitor in network.monitors:
            self.cost_names.append(name)
            self.cost_exprs.append(monitor)

        self.shapes = [p.get_value(borrow=True).shape for p in self.params]
        self.counts = [np.prod(s) for s in self.shapes]
        self.starts = np.cumsum([0] + self.counts)[:-1]
        self.dtype = self.params[0].get_value().dtype

        self.best_cost = 1e100
        self.best_iter = 0
        self.best_params = [p.get_value().copy() for p in self.params]

        mode = "FAST_RUN"
        if self.profile:
            mode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())


        if compile:
            self.f_eval = theano.function(
                network.inputs, network.output)

            self.f_learn = theano.function(
                network.inputs + [network.target],
                self.cost_exprs,
                updates=list(self.learning_updates()),mode=mode)

            self.f_cost =theano.function(
                network.inputs + [network.target],
                self.cost_exprs)



    def flat_to_arrays(self, x):
        x = x.astype(self.dtype)
        return [x[o:o+n].reshape(s) for s, o, n in
                zip(self.shapes, self.starts, self.counts)]

    def arrays_to_flat(self, arrays):
        x = np.zeros((sum(self.counts), ), self.dtype)
        for arr, o, n in zip(arrays, self.starts, self.counts):
            x[o:o+n] = arr.ravel()
        return x

    def set_params(self, targets):
        for param, target in zip(self.params, targets):
            param.set_value(target)

    def learning_updates(self):
        # This code computes updates only for given R, so it drops last dimension. Plus soe theano magic to circumvent its graph comp.
        grads = self.grads
        for i, param in enumerate(self.params):
            delta = self.lr * grads[i]
            velocity = theano.shared(
                np.zeros_like(param.get_value(), dtype=theano.config.floatX), name=param.name +'_vel')

            yield velocity, T.cast(self.momentum * velocity - delta, theano.config.floatX)
            yield param, param + velocity


    def train_minibatch(self, x):
        costs_vals = self.f_learn(*x)
        costs = list(zip( self.cost_names, costs_vals))
        return costs

import scipy

class Scipy(SGD):
    '''General trainer for neural nets using `scipy.optimize.minimize`.'''

    METHODS = ('l-bfgs-b', 'cg', 'dogleg', 'newton-cg', 'trust-ncg')

    def __init__(self, network, L2=0.0001, L1=0,  method = 'l-bfgs-b'):

        SGD.__init__(self,network=network, L2=L2, L1=L1, compile= False )

        self.method = method

        if compile:
            self.f_eval = theano.function(
                network.inputs, network.output)

            self.f_cost = theano.function(
                network.inputs+ [network.target], self.cost_exprs)


            self.f_grad = theano.function(network.inputs+ [network.target], T.grad(self.cost, self.params))


    def function_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x.astype(np.float32)))
        return np.mean([self.f_cost(*y)[0]  for y in train_set]).astype(np.float64) #lbfgs fortran code wants float64. meh

    def gradient_at(self, x, train_set):
        self.set_params(self.flat_to_arrays(x.astype(np.float32)))
        grads = [[] for _ in range(len(self.params))]
        for y in train_set:
            for i, g in enumerate(self.f_grad(*y)):
                grads[i].append(np.asarray(g))
        G = self.arrays_to_flat([np.mean(g, axis=0) for g in grads]).astype(np.float64) #lbfgs fortran code wants float64. meh
        return G

    def train_minibatch(self, x):
        def display(p):
            pass
            #self.set_params(self.flat_to_arrays(p.astype(np.float32)))
            #costs = self.f_cost(*x)
            #cost_desc = ' '.join(
            #    '%s=%.6f' % el for el in zip(self.cost_names, costs))
            #print('scipy %s %s' %
            #      (self.method, cost_desc))
            #sys.stdout.flush()


        try:
            res = scipy.optimize.minimize(
                fun=self.function_at,
                jac=self.gradient_at,
                x0=self.arrays_to_flat(self.best_params),
                args=([x], ),
                method=self.method,
                callback=display,
                options=dict(maxiter=2),
            )
        except KeyboardInterrupt:
            print('interrupted!')

        @timed
        def set():
            params = self.flat_to_arrays(res.x.astype(np.float32))
            self.set_params(params)

        return []

class Rprop(SGD):
    '''Trainer for neural nets using resilient backpropagation.
    The Rprop method uses the same general strategy as SGD (both methods are
    make small parameter adjustments using local derivative information). The
    difference is that in Rprop, only the signs of the partial derivatives are
    taken into account when making parameter updates. That is, the step size for
    each parameter is independent of the magnitude of the gradient for that
    parameter.
    To accomplish this, Rprop maintains a separate learning rate for every
    parameter in the model, and adjusts this learning rate based on the
    consistency of the sign of the gradient of J with respect to that parameter
    over time. Whenever two consecutive gradients for a parameter have the same
    sign, the learning rate for that parameter increases, and whenever the signs
    disagree, the learning rate decreases. This has a similar effect to
    momentum-based SGD methods but effectively maintains parameter-specific
    momentum values.
    The implementation here actually uses the "iRprop-" variant of Rprop
    described in Algorithm 4 from Igel and Huesken (2000), "Improving the Rprop
    Learning Algorithm." This variant resets the running gradient estimates to
    zero in cases where the previous and current gradients have switched signs.
    '''

    def __init__(self, network, **kwargs):
        self.step_increase = kwargs.get('rprop_increase', 1.01)
        self.step_decrease = kwargs.get('rprop_decrease', 0.99)
        self.min_step = kwargs.get('rprop_min_step', 0.)
        self.max_step = kwargs.get('rprop_max_step', 100.)
        super(Rprop, self).__init__(network, **kwargs)

    def learning_updates(self):
        step = self.lr
        self.grads = []
        self.steps = []
        for param in self.params:
            v = param.get_value()
            n = param.name
            self.grads.append(theano.shared(np.zeros_like(v), name=n + '_grad'))
            self.steps.append(theano.shared(np.zeros_like(v) + step, name=n + '_step'))
        for param, step_tm1, grad_tm1 in zip(self.params, self.steps, self.grads):
            grad = T.grad(self.J, param)
            test = grad * grad_tm1
            same = T.gt(test, 0)
            diff = T.lt(test, 0)
            step = T.minimum(self.max_step, T.maximum(self.min_step, step_tm1 * (
                T.eq(test, 0) +
                same * self.step_increase +
                diff * self.step_decrease)))
            grad = grad - diff * grad
            yield param, param - T.sgn(grad) * step
            yield grad_tm1, grad
            yield step_tm1, step
