import glob, argparse, zipfile
from os import path

BLACKLIST = ['__pycache__', '.pyc', '/bundle.py', '/test.py', '/play.py']
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('utid')
	args = parser.parse_args()
	
	this_path = path.dirname(path.realpath(__file__))
	files = []
	for f in glob.glob(path.join(this_path,'**')):
		if all(b not in f for b in BLACKLIST):
			files.append(f)
	
	zf = zipfile.ZipFile(args.utid+'.zip', 'w', compression=zipfile.ZIP_DEFLATED)
	for f in files:
		zf.write(f, f.replace(this_path,args.utid))
	zf.close()
