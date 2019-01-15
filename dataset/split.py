import os, json, random, shutil
from PIL import Image

def dirpath(str):
	return os.path.join(os.path.normpath(str), '')

def ensure_dir(path):
	path = dirpath(path)
	d = os.path.dirname(path)
	if not os.path.exists(d):
		os.makedirs(d)
	return d

data = json.loads(open('cookster-new.json', 'r').read())

keys = [k for k in data]
random.shuffle(keys)

train_len = int(len(keys) * 0.7)
train = keys[:train_len]
val = keys[train_len:]

train_dir = ensure_dir('train')
val_dir = ensure_dir('val')

def cpimgs(keys, dest):
	for k in keys:
		filename = data[k]['filename']
		shutil.copyfile(os.path.join('images', filename), os.path.join(dest, filename))

cpimgs(train, train_dir)
cpimgs(val, val_dir)

train_data = {}
for k in train:
	train_data[k] = data[k]

val_data = {}
for k in val:
	val_data[k] = data[k]

open(os.path.join(train_dir, 'cookster.json'), 'w').write(json.dumps(train_data))
open(os.path.join(val_dir, 'cookster.json'), 'w').write(json.dumps(val_data))