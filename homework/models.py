from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

def get_seq_mask(input_seq_lens, max_seq_len):
    return torch.as_tensor(np.asarray([[1 if j < input_seq_lens.data[i].item() else 0 for j in range(0, max_seq_len)] for i in range(0, input_seq_lens.shape[0])]), dtype=torch.float)#.cuda()


class Policy:
	def __init__(self, model):
		self.model = model
		self.hist = []

	def __call__(self, obs):
		self.hist.append(obs)
		if len(self.hist) > self.model.width:
			self.hist = self.hist[-self.model.width:]
		x = torch.stack(self.hist, dim=0)[None]
		return self.model(x)[0,-1,:]

class CheatModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.width = 1

	def forward(self, hist):
		b,s,c,h,w = hist.size()
		action = torch.tensor([-10,1,-10,-10,1,-10]).float()#.cuda()
		actions = torch.stack([action]*b,dim=0)
		actions = torch.stack([actions]*s,dim=1)
		return actions

	def policy(self):
		return Policy(self)

class CNNModel(nn.Module):
	def __init__(self):
		super().__init__()

		self.width = 1

		self.conv = nn.Sequential(
			nn.Conv2d(3,16,5,3,1),
			nn.ReLU(True),
			nn.Conv2d(16,32,5,3,1),
			nn.ReLU(True),
			nn.Conv2d(32,64,5,3,1),
			nn.ReLU(True),
		)

		self.fc = nn.Sequential(
			nn.Linear(256,64),
			nn.ReLU(True),
			nn.Linear(64,6),
		)

	def forward(self, hist):
		b,s,c,h,w = hist.size()
		hist = hist.view(b*s,c,h,w)
		h = self.conv(hist).view(-1,256)
		actions = self.fc(h)
		actions = actions.view(b,s,6)
		return actions

	def policy(self):
		return Policy(self)

class LSTMCNNModel(nn.Module):
	def __init__(self):
		super().__init__()

		self.width = 20

		self.conv = nn.Sequential(
			nn.Conv2d(3,16,5,3,1),
			nn.ReLU(True),
			nn.Conv2d(16,32,5,3,1),
			nn.ReLU(True),
			nn.Conv2d(32,64,5,3,1),
			nn.ReLU(True),
		)

		self.fc = nn.Sequential(
			nn.Linear(256,64),
			nn.ReLU(True),
			nn.Linear(64,32),
		)

		self.lstm = nn.LSTM(32,32, batch_first=True)

		self.out = nn.Linear(32,6)

	def forward(self, hist):
		b,s,c,h,w = hist.size()
		hist = hist.view(b*s,c,h,w)
		h = self.conv(hist).view(-1,256)
		h = self.fc(h).view(b,s,32)

		lstm_out, hidden = self.lstm(h)
		actions = self.out(lstm_out.contiguous().view(b*s,32)).view(b,s,6)
		return actions

	def policy(self):
		return Policy(self)


class TempCNNCNNModel(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv = nn.Sequential(
			nn.Conv2d(3,16,5,3,1),
			nn.ReLU(True),
			nn.Conv2d(16,32,5,3,1),
			nn.ReLU(True),
			nn.Conv2d(32,64,5,3,1),
			nn.ReLU(True),
		)

		self.fc = nn.Sequential(
			nn.Linear(256,64),
			nn.ReLU(True),
			nn.Linear(64,32),
		)

		self.t_cnn = nn.Sequential(
			nn.Conv1d(32,16,5),
			nn.ReLU(True),
			nn.Conv1d(16,16,5),
			nn.ReLU(True),
			nn.Conv1d(16,16,5),
			nn.ReLU(True),
			nn.Conv1d(16,6,5),
		)

		self.width = 4*(5-1)

	def forward(self, hist):
		b,s,c,h,w = hist.size()
		hist = hist.view(b*s,c,h,w)
		h = self.conv(hist).view(-1,256)
		h = self.fc(h).view(b,s,32)
		h = h.permute(0,2,1)
		h = F.pad(h, (self.width,0))
		actions = self.t_cnn(h)
		actions = actions.permute(0,2,1)
		return actions

	def policy(self):
		return Policy(self)


Model = TempCNNCNNModel
