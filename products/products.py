import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import PIL
from mrcnn.config import Config # pylint: disable=all
from mrcnn import model as modellib, utils, visualize # pylint: disable=all
import matplotlib.pyplot as plt
from flask import jsonify

ROOT_DIR = os.path.abspath('.')
sys.path.append(ROOT_DIR)
COCO_WEIGHTS_PATH = 'mask_rcnn_coco.h5'
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
PRODUCTS = ['Apple', 'Banana', 'Tomato', 'Cucumber', 'Egg']
CWD = os.getcwd()

class ProductConfig(Config):
	NAME = 'Products'
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 5
	STEPS_PER_EPOCH = 100
	DETECTION_MIN_CONFIDENCE = 0.7

class ProductDataset(utils.Dataset):
	def load_product(self, dataset_dir, subset):
		for i in range(len(PRODUCTS)):
			self.add_class('Products', i+1, PRODUCTS[i])
		assert subset in ['train', 'test']
		annotations = json.load(open(os.path.join(dataset_dir, subset, 'cookster.json')))
		annotations = list(annotations.values())
		annotations = [a for a in annotations if a['regions']]
		for a in annotations:
			if type(a['regions']) is dict:
				polygons = [{'shape_attributes': r['shape_attributes'], 'class': r['region_attributes']['class']} for r in a['regions'].values()]
			else:
				polygons = [
					{
						'xs': r['shape_attributes']['all_points_x'],
						'ys': r['shape_attributes']['all_points_y'],
						'class': r['region_attributes']['class']
					} for r in a['regions']
				]
			image_path = os.path.join(dataset_dir, subset, a['filename'])
			height, width = a['Height'], a['Width']
			self.add_image('Products', image_id=a['filename'], path=image_path, width=width, height=height, polygons=polygons)

	def load_mask(self, image_id):
		info = self.image_info[image_id]
		mask = np.zeros([info['height'], info['width'], len(info['polygons'])], dtype=np.uint8)
		class_ids = []
		for i, p in enumerate(info['polygons']):
			rr, cc = skimage.draw.polygon(p['ys'], p['xs'])
			class_ids.append(self.class_names.index(p['class']))
			mask[rr, cc, i] = class_ids[-1]
		return mask, np.array(class_ids).astype(np.int32)

	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']

class EvalResult:
	def __init__(self, data, classes, image):
		self.data = data
		self.image = image
		self.classes = classes

	def save(self, image_path):
		image_path = fixpath(image_path)
		_, ax = plt.subplots(1, figsize=(16, 16))
		visualize.display_instances(
			image=self.image,
			boxes=self.data['rois'],
			masks=self.data['masks'],
			class_ids=self.data['class_ids'],
			class_names=self.classes,
			scores=self.data['scores'],
			ax=ax)
		plt.savefig(image_path)

	def to_json(self):
		# todo
		return jsonify({'products': [{'apple': 1}, {'orange': 2}]})

class CooksterNN:
	def __init__(self, is_training=False, load_last_checkpoint=True):
		self.config = ProductConfig()
		self.mode = 'training' if is_training else 'inference'
		self.model = modellib.MaskRCNN(mode=self.mode, config=self.config, model_dir=DEFAULT_LOGS_DIR)
		if load_last_checkpoint:
			weights_path = self.model.find_last()
			print('Loading weights ', weights_path)
			self.model.load_weights(weights_path, by_name=True)
		else:
			print('Loading weights ', COCO_WEIGHTS_PATH)
			self.model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
				'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])

	def train(self, num_epochs, dataset_dir):
		dataset_train = ProductDataset()
		dataset_train.load_product(dataset_dir, 'train')
		dataset_train.prepare()
		dataset_val = ProductDataset()
		dataset_val.load_product(dataset_dir, 'test')
		dataset_val.prepare()
		self.model.train(dataset_train, dataset_val, learning_rate=self.config.LEARNING_RATE, epochs=num_epochs, layers='heads')

	def eval(self, input_image):
		input_image = fixpath(input_image)
		image = skimage.io.imread(input_image)
		return EvalResult(self.model.detect([image])[0], ['bg'] + PRODUCTS, image)

def fixpath(p):
	return os.path.abspath(os.path.normpath(p))