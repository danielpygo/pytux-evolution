from .models import *
import os, torch

dirname = os.path.dirname(os.path.abspath(__file__))

try:
	model = Model()
	model.load_state_dict(torch.load(os.path.join(dirname, 'model.th'), map_location=lambda storage, loc: storage))
except:
	model = Model()