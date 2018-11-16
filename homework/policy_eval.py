import ray
from pytux import Tux
import numpy as np
import torch

def get_action(logits):
	probs = 1. / (1. + np.exp(-logits.detach().numpy()))
	bits = np.array([np.random.uniform()<=p for p in probs]).astype(int)
	return int(np.packbits(bits)[0] >> 2)

@ray.remote
class PolicyEvaluator():

	def __init__(self, level, iterations):
		
		self.level = level 
		self.iterations = iterations

	def eval(self, model, H):

		torch.set_num_threads(1)

		ps = []

		for it in range(self.iterations):

			p = 0.0
			T = Tux('data/levels/world1/%s'%self.level, 128, 128, acting=True, visible=True, synchronized=True)

			# Restart Tux
			T.restart()
			if not T.waitRunning():
				exit(0)

			fid, act, state, obs = T.step(0)
			policy = model.policy()

			for t in range(H):
				
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

			ps.append(p)

		return np.mean(ps)