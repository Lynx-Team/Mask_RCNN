import json
import os

file = open('cookster.json', 'r').read()

jsonf = json.loads(file)

i = 0
for key, item in jsonf.items():
  print(key.split('.')[0] + '.jpg')
  jsonf[key]['filename'] = key.split('.')[0] + '.jpg'

file = open('cookster2.json', 'w')
file.write(json.dumps(jsonf))