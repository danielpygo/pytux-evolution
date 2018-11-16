from glob import glob
from os import path

THIS_DIR = path.dirname(path.abspath(__file__))
PARENT_DIR = path.dirname(THIS_DIR)
DATA_DIR = path.join(PARENT_DIR, 'data')
LEVELS_DIR = path.join(DATA_DIR, 'levels')

levels = sorted(glob(path.join(LEVELS_DIR,'*/*.stl')))
