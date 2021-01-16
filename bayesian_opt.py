import numpy as np
from main import train_model
from model.model import CNN_classifier
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
from skopt import gp_minimize
import math
import pickle
import logging
logging.basicConfig(level=logging.ERROR)

'''
from https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html#sphx-glr-auto-examples-bayesian-optimization-py
'''

#model = CNN_classifier("efficientnet-b0", 2)

dim_epochs = Integer(low = 10, high = 20, name = "epochs")
dim_batch = Integer(low = 8, high = 16, name = "batch")
dim_lr = Real(low = 1e-6, high = 1e-2, prior = "log-uniform", name = "lr")
dim_weight_decay = Real(low = 1e-6, high = 1e-2, prior = "log-uniform", name = "weight_decay")

#dim_freeze = Categorical(2)

dimensions = [dim_epochs, dim_batch, dim_lr, dim_weight_decay]
default_parameters = [10, 8, 1e-5, 1e-5]

@use_named_args(dimensions = dimensions)
def fitness(epochs, batch, lr, weight_decay):

    params = { "epochs" : epochs,
            "batch_size" : batch,
            "lr" : lr,
            "weight_decay" : weight_decay}

    model_name = "efficientnet-b2"
    acc = train_model(model_name , **params)
    
    return -acc

search_result = gp_minimize(func = fitness, 
                            dimensions = dimensions,
                            acq_func = "EI",
                            n_calls = 20,
                            x0 = default_parameters)

print(search_result.x)

with open("./eff_b2.pickle", "wb") as f:
    pickle.dump(search_result, f)