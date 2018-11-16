import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .models import *
import ray
from .policy_eval import PolicyEvaluator

ray.init()
dirname = os.path.dirname(os.path.abspath(__file__))

levels = [
    '01 - Welcome to Antarctica.stl',
    '02 - The Journey Begins.stl',
    '03 - Via Nostalgica.stl',
    '04 - Tobgle Road.stl',
    '05 - The Somewhat Smaller Bath.stl',
]

#policy_evaluators = [ PolicyEvaluator(lvl) for lvl in levels ]

def get_action(logits):
	probs = 1. / (1. + np.exp(-logits.detach().numpy()))
	bits = np.array([np.random.uniform()<=p for p in probs]).astype(int)
	return int(np.packbits(bits)[0] >> 2)

def test(iterations, H=500):

    policy_evaluators = [ PolicyEvaluator.remote(lvl, iterations=iterations) for lvl in levels ]

    # Load the model
    model = Model()
    model.load_state_dict(torch.load(os.path.join(dirname, 'model.th')))

    model.eval()

    positions = {level : [] for level in levels}


    episode_p = ray.get([ evaluator.eval.remote(model, H) for evaluator in policy_evaluators ])

    print ('===== Evaludation Finished =====')
    print ("Mean rewards: %.5f"%np.mean(episode_p))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int, default=10)

    args = parser.parse_args()

    test(iterations=args.iterations, H=2000)
