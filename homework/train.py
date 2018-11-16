import argparse, pickle
import numpy as np
import os
from itertools import cycle

import torch
from torch import nn, optim

from .main import model
from .models import *

import ray
from .policy_eval import PolicyEvaluator

ray.init()
dirname = os.path.dirname(os.path.abspath(__file__))


def train(epoch):
    '''
    This is the main training function. You need to fill in this function to complete the assignment
    '''


    levels = {
        '01 - Welcome to Antarctica.stl' : 0.2,
        '02 - The Journey Begins.stl': 0.2,
        '03 - Via Nostalgica.stl' : 0.2,
        '04 - Tobgle Road.stl' : 0.2,
        '05 - The Somewhat Smaller Bath.stl' : 0.2,
    }
    print ("Num of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

    gaussians = {}
    survivors = []
    alpha_model = model
    # print(alpha_model.named_parameters())
    for k,v in alpha_model.named_parameters():
        gaussians[k] = (0,3)

    for t in range(epoch):

        '''
        Your code here : use your favorite method here
        '''

        '''
        Your code here : optionally, you can print diagnostics of your model below
        '''

        models_n = 20
        keep_models = 4

        models = [Model() for i in range(0,models_n)]
        for m in models:
            params = {}
            for k,v in alpha_model.named_parameters():
                train_parameters[k] = v + numpy.random_normal(gaussians[k][0], gaussians[k][1])
            m.load_state_dict(params)

        n_workers = 3
        H=500
        evaluators = [PolicyEvaluator.remote(level, iterations) for _ in range(n_workers)]
        rewards = ray.get([
        	evaluator.eval.remote(m, H) for m, evaluator in zip(models, evaluators)
        ])

        indices = np.argsort(rewards)[:keep_models]
        survivors = [models[i] for i in indices]
        compute_new = {}
        sum_mean = 0
        for survivor in survivors:
            for k,v in survivor.named_parameters():
                if k in compute_new:
                    compute_new[k] = (0,2)
                else:
                    compute_new[k][0] += v
        for k,v in compute_new:
            compute_new[k][0] /= 4

        gaussians = compute_new

        # Print diagnostics
        print ('====== Iter: %d ======' % t)
        print ("Mean reward: %.5f" % np.mean(rewards))
        print ("Std reward: %.5f" % np.std(rewards))
        print ("Min reward: %.5f" % np.min(rewards))
        print ("Max reward: %.5f" % np.max(rewards))


    # Save model
    torch.save(model.state_dict(), os.path.join(dirname, 'model.th')) # Do NOT modify this line




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--epoch', type=int, default=50)

    args = parser.parse_args()

    print ('[I] Start training')
    train(args.epoch)
    print ('[I] Training finished')
