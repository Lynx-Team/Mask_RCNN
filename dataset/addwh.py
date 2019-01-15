import os, json
from PIL import Image

data = json.loads(open('cookster.json', 'r').read())

ignore = {}

for _, content in data.items():
	filename = content['filename']
	try:
		(w, h) = Image.open(os.path.join('images', filename)).size
		content['Width'] = w
		content['Height'] = h
	except:
		ignore[_] = 1

for k in ignore:
	del data[k]

open('cookster-new.json', 'w').write(json.dumps(data))