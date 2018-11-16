import argparse, pickle
import numpy as np
import os
from itertools import cycle
import random
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
    trainable = {}
    for name,p in alpha_model.named_parameters():
        if p.requires_grad:
            trainable[name] = p
    # for k,v in trainable.items():
    #     gaussians[k] = (0,3)

    for t in range(epoch):

        '''
        Your code here : use your favorite method here
        '''

        '''
        Your code here : optionally, you can print diagnostics of your model below
        '''
        print("epoch", t)
        models_n = 20
        keep_models = 4

        models = [Model() for i in range(0,models_n)]
        for m in models:
            params1 = {}
            subdict = {}
            for i in range(200):
                k = random.choice(list(trainable))
                subdict[k] = trainable[k]
            for k,v in trainable.items():
                params1[k] = v
            for k,v in subdict.items():
                params1[k] = v + np.random.normal(0,2)
            m.load_state_dict(params1)

        n_workers = 4
        H=500
        iterations = 5
        level = random.choice(list(levels))
        evaluators = [PolicyEvaluator.remote(level, iterations) for _ in range(n_workers)]
        rewards = ray.get([
        	evaluator.eval.remote(m, H) for m, evaluator in zip(models, evaluators)
        ])

        indices = np.argsort(rewards)[:keep_models]
        survivors = [models[i] for i in indices]
        # compute_new_gauss = {}
        params = {}
        sum_mean = 0
        for survivor in survivors:
            for k,v in trainable.items():
                if k not in params:
                    params[k] = v
                else:
                    params[k] = torch.add(params[k],v)
        for k,v in params.items():
            # compute_new_gauss[k][0] /= 4
            params[k] = torch.div(params[k],4)
        m = Model()
        m.load_state_dict(params)
        alpha_model = m
        # gaussians = compute_new_gausss

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
