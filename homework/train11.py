import argparse, pickle
import numpy as np
import os
from itertools import cycle

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import *
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

def cycle(seq):
	while True:
		for elem in seq:
			yield elem

def train(max_iter, batch_size=16, log_dir=None):
	'''
	This is the main training function, feel free to modify some of the code, but it
	should not be required to complete the assignment.
	'''

	"""
	Load the training data
	"""

	train_dataloader = load('train', num_workers=0, crop=20, batch_size=batch_size)
	valid_dataloader = load('val', num_workers=0, crop=20, batch_size=batch_size)

	train_dataloader_iterator = cycle(train_dataloader)
	valid_dataloader_iterator = cycle(valid_dataloader)

	model = Model()##.cuda()
	print (model)

	print ("Num of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

	log = None
	if log_dir is not None:
		from .utils import SummaryWriter
		log = SummaryWriter(log_dir)

	# If your model does not train well, you may swap out the optimizer or change the lr

	optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)
	#optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay=1e-4)

	loss = nn.BCEWithLogitsLoss(
		pos_weight=torch.as_tensor([13.97403691 ,  3.64247702,  31.81070587,  20.31615018,   2.62973018, 5.16822194])##.cuda()
	)

	for t in range(max_iter):
		batch_obs, batch_actions = next(train_dataloader_iterator)
		batch_obs = batch_obs.float().permute(0,1,4,2,3)#.cuda()
		batch_actions = batch_actions.float().permute(0,2,1)#.cuda()

		model.train()

		# zero the gradients (part of pytorch backprop)
		optimizer.zero_grad()

		# Compute the model output and loss (view flattens the input)
		model_outputs = model(batch_obs)

		# Compute the loss
		t_loss_val = loss(model_outputs, batch_actions)

		# Compute the gradient
		t_loss_val.backward()

		# Update the weights
		optimizer.step()

		if t % 10 == 0:
			model.eval()

			batch_obs, batch_actions = next(valid_dataloader_iterator)
			batch_obs = batch_obs.float().permute(0,1,4,2,3)#.cuda()
			batch_actions = batch_actions.float().permute(0,2,1)#.cuda()

			model_outputs = model(batch_obs)

			v_loss_val = loss(model_outputs, batch_actions)

			print('[%5d]  t_loss = %f   v_loss = %f'%(t, t_loss_val,v_loss_val))

			if log is not None:
				log.add_scalar('train/loss', t_loss_val, t)
				log.add_scalar('val/loss', v_loss_val, t)

	# Save the trained model
	torch.save(model.state_dict(), os.path.join(dirname, 'model.th')) # Do NOT modify this line

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--max_iter', type=int, default=5000)
	parser.add_argument('-l', '--log_dir')
	args = parser.parse_args()

	print ('[I] Start training')
	train(args.max_iter, log_dir=args.log_dir)
	print ('[I] Training finished')
