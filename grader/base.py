class CheckFailed(Exception):
	def __init__(self, why):
		self.why = why
	
	def __str__(self):
		return "[E] Grading failed! %s"%self.why

class ContextManager:
	def __init__(self, on, off):
		self.on = on
		self.off = off
	
	def __enter__(self):
		self.on()
	
	def __exit__(self, exc_type, exc_value, traceback):
		self.off()

class Grader:
	def __init__(self, module):
		import torch
		self.m = module
		self.g = torch.jit.get_trace_graph(module, self.INPUT_EXAMPLE)[0].graph()
		self.verbose = False
	
	def CHECK_SHAPE(self, t, s, name=None):
		S = t.shape
		if name is None: name = t.name
		assert len(S) == len(s), "Shape mismatch for tensor '%s' expected %s got %s"%(name, str(s), str(S))
		for a,b in zip(s,S):
			assert b is None or a == b, "Shape mismatch for tensor '%s' expected %s got %s"%(name, str(s), str(S))
	
	section = ""
	section_score = 0
	section_max = 0
	def BEGIN_SECTION(self, section):
		self.group = None
		self.section = section
		self.section_score = 0
		self.section_max = 0

	def END_SECTION(self):
		from sys import stdout
		if self.verbose: stdout.write('\r'+80*' ')
		
		self.section_score, self.section_max = round(self.section_score), round(self.section_max)
		print( '\r * %-50s	[ %3d / %3d ]'%(self.section, self.section_score, self.section_max) )
		self.score += self.section_score
		self.section_score = 0
		self.section_max = 0
	
	def SECTION(self, section):
		def on(): self.BEGIN_SECTION(section)
		def off(): self.END_SECTION()
		return ContextManager(on, off)
	
	group = None
	group_ok = False
	def BEGIN_GROUP(self):
		self.group = 0
		self.group_ok = True
	
	def END_GROUP(self):
		if self.group is not None and not self.group_ok:
			self.section_score -= self.group
		self.group = None
	
	def GROUP(self):
		def on(): self.BEGIN_GROUP()
		def off(): self.END_GROUP()
		return ContextManager(on, off)
	
	def CASE(self, passed, score=1):
		from sys import stdout
		if passed:
			if self.verbose: stdout.write('+')
			self.section_score += score
		else:
			if self.verbose: stdout.write('-')
		self.section_max += score
		if self.group is not None:
			if not passed:
				self.group_ok = False
			else:
				self.group += score
	
	BLACK_LIST = []
	ALLOWED_CONST = None
	def const_check(self, v):
		if self.ALLOWED_CONST is None: return True
		try:
			for a in v:
				if not self.const_check(a):
					return False
			return True
		except TypeError:
			return v in self.ALLOWED_CONST
	
	def find_const(self, n):
		import collections
		r = []
		for i in n.attributeNames():
			k = n.kindOf(i)
			if k == 't':
				v = n.t(i).tolist()
				if isinstance(v, collections.Iterable):
					r.extend( v )
				else:
					r.append( v )
		return r
	
	def op_check(self):
		for n in self.g.nodes():
			k = n.kind()
			if 'aten::' in k:
				k = k.replace('aten::','')
			else:
				raise CheckFailed("Unknown operation '%s'"%k)
			
			if k in self.BLACK_LIST:
				raise CheckFailed("Operation '%s' not allowed in this assignment!"%k)
			
			if self.ALLOWED_CONST is not None:
				for v in self.find_const(n):
					if not self.const_check(v):
						raise CheckFailed("Only constants allowed are %s got %s!"%(str(self.ALLOWED_CONST), str(v)))		
			
	def io_check(self):
		pass
	
	def __call__(self):
		try:
			self.op_check()
		except CheckFailed as e:
			print( str(e) )
			return 0
		
		try:
			self.io_check()
		except CheckFailed as e:
			print( str(e) )
			return 0
		except AssertionError as e:
			print( "[E] Grading failed! " + str(e) )
			return 0
		
		self.score = 0
		self.grade()
		return self.score
