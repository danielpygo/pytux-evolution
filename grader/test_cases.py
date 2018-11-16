from . import base
import torch, os
import numpy as np

from torch import nn
from pytux import Tux


def get_action(logits):
	probs = 1. / (1. + np.exp(-logits.detach().numpy()))
	bits = np.array([np.random.uniform()<=p for p in probs]).astype(int)
	return int(np.packbits(bits)[0] >> 2)


class Grader(base.Grader):
	TOTAL_SCORE = 100

	def __init__(self, module):
		import torch
		self.module = module
		self.model = module.model#.cuda()
		if self.model is None:
			print( "Failed to load model. Did you train one?" )
			exit(1)
		self.model.eval()
		self.verbose = False

		self.levels = [
			'01 - Welcome to Antarctica.stl',
			'02 - The Journey Begins.stl',
			'03 - Via Nostalgica.stl',
			'04 - Tobgle Road.stl',
			'05 - The Somewhat Smaller Bath.stl',
		]

	def op_check(self):
		pass

	def grade(self):
		import numpy as np

		torch.manual_seed(0)

		positions = {level : [] for level in self.levels}
		
		torch.set_num_threads(1)

		for it in range(10):
			for level in self.levels:
				p = 0.0
				T = Tux('data/levels/world1/%s'%level, 128, 128, acting=True, visible=True, synchronized=True)

				# Restart Tux
				T.restart()
				if not T.waitRunning():
					exit(0)

				fid, act, state, obs = T.step(0)
				policy = self.model.policy()

				for t in range(2000):
					tux_mask = (obs['label'] % 8) == 4

					xs, ys = np.argwhere(tux_mask).T
					try:
						x = int(xs.mean())
						y = int(ys.mean())
					except:
						x = 64
						y = 64

					img = obs['image']
					img = np.pad(img, ((32,32), (32,32), (0,0)), mode='constant', constant_values=127)
					img = img[x:x+64,y:y+64]
					img = (img - [81.715, 109.922, 132.204]) / [50.216, 65.347, 77.755]
					img = torch.as_tensor(img).float()
					logits = policy(img.permute(2,0,1))
					a = get_action(logits)
					try:
						fid, act, state, obs = T.step(a)
					except TypeError as e:
						print (e)
						break
					if state['is_dying']:
						break
					p = max(p, state['position'])
				positions[level].append(p)


		mean_reward = np.mean([np.mean(v) for v in positions.values()])

		with self.SECTION('Testing mean rewards'):
			for k in np.linspace(0.2, 0.35, 100):
				self.CASE(mean_reward >= k)
