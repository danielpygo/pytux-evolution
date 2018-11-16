from pytux import Tux
from time import sleep, time

t0 = time()
#T = Tux("data/levels/world1/03 - Via Nostalgica.stl", 600, 400, True, True)
T = Tux("data/levels/world1/03 - Via Nostalgica.stl", 200, 150, acting=False, visible=False, lrW=11, lrH=7, lrS=10)
t1 = time()
T.restart()
t2 = time()
print( T.waitRunning() )
t3 = time()
print( T.running, t1-t0, t2-t1, t3-t2 )

t0, t1 = 0, 0
s = 0
for it in range(10):
	t0 -= time()
	T.restart()
	T.waitRunning()
	t0 += time()
	t1 -= time()
	for i in range(1000):
		s += T.step(2) is not None
	t1 += time()
print( s / (t1+t0), t0, t1 )

T.restart()
print( T.waitRunning() )
im, lbl = None, None
for i in range(10000):
	fid, act, state, obs = T.step(2)
	from pylab import *
	ion()   
	#figure()
	#subplot(1,2,1)
	#if im is None:
		#im = imshow(obs['image'])
	#else:
		#im.set_data(obs['image'])
	#subplot(1,2,2)
	if lbl is None:
		lbl = imshow(obs['label_lr'] & 0xf, interpolation='nearest')
		colorbar()
	else:
		lbl.set_data(obs['label_lr'] & 0xf)
	draw()
	pause(0.001)
	print( fid, act, state, list(obs) )
