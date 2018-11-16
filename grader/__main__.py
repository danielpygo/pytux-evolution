import argparse, importlib

parser = argparse.ArgumentParser('Grade an assignment')
parser.add_argument('assignment', default='homework')
args = parser.parse_args()

print( 'Loading assignment ...' )
assignment = importlib.import_module(args.assignment+'.main')

print( 'Loading grader ...' )
from .test_cases import Grader
g = Grader(assignment)

print( 'Grading locally ...' )
score = g()
print()
print( 'total score locally                                        %3d / %3d'%(score, g.TOTAL_SCORE) )
