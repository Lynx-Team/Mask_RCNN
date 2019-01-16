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
from imgaug import augmenters as iaa

def fixpath(p):
	return os.path.abspath(os.path.normpath(p))

ROOT_DIR = fixpath('.') # '/var/www/cookster-nn/'
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
PRODUCTS = ['Apple', 'Banana', 'Tomato', 'Cucumber', 'Egg']

class ProductConfig(Config):
	NAME = 'Products'
	IMAGES_PER_GPU = 1
	NUM_CLASSES = 1 + 5
	STEPS_PER_EPOCH = 100
	DETECTION_MIN_CONFIDENCE = 0.77

class ProductDataset(utils.Dataset):
	def load(self, dataset_dir):
		for i in range(len(PRODUCTS)):
			self.add_class('Products', i+1, PRODUCTS[i])
		annotations = json.load(open(os.path.join(dataset_dir, 'cookster.json')))
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
			image_path = os.path.join(dataset_dir, a['filename'])
			height, width = a['Height'], a['Width']
			self.add_image('Products', image_id=a['filename'], path=image_path, width=width, height=height, polygons=polygons)
		return self

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
		products = {}
		for i in range(len(self.data['scores'])):
			product_id = self.data['class_ids'][i]
			product = self.classes[product_id]
			if product in products:
				products[product] += 1
			else:
				products[product] = 1
		return jsonify(products=products)

class Weights:
	COCO = 0
	LAST = 1
	IMAGENET = 2

class CooksterNN:
	def __init__(self, training=False, weights=Weights.LAST):
		self.config = ProductConfig()
		self.mode = 'training' if training else 'inference'
		self.model = modellib.MaskRCNN(mode=self.mode, config=self.config, model_dir=DEFAULT_LOGS_DIR)

		if weights == Weights.COCO or weights == Weights.IMAGENET:
			weights_path = COCO_WEIGHTS_PATH if weights == Weights.COCO else self.model.get_imagenet_weights()
			print('Loading weights ', weights_path)
			self.model.load_weights(weights_path, by_name=True, exclude=[
				'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
		elif weights == Weights.LAST:
			weights_path = self.model.find_last()
			print('Loading weights ', weights_path)
			self.model.load_weights(weights_path, by_name=True)

	def train(self, num_epochs, train_path, val_path):
		train = ProductDataset().load(fixpath(train_path))
		train.prepare()
		val = ProductDataset().load(fixpath(val_path))
		val.prepare()

		aug = iaa.SomeOf((0, 2), [
			iaa.Fliplr(0.5),
			iaa.Flipud(0.5),
			iaa.OneOf([iaa.Affine(rotate=90), iaa.Affine(rotate=180), iaa.Affine(rotate=270)]),
			iaa.GaussianBlur(sigma=(0.0, 2.0)),
			iaa.Multiply((0.8, 1.5)),
		])

		self.model.train(
			train_dataset=train,
			val_dataset=val,
			learning_rate=self.config.LEARNING_RATE,
			epochs=num_epochs,
			layers='heads',
			augmentation=aug)

	def eval(self, input_image):
		input_image = fixpath(input_image)
		image = skimage.io.imread(input_image)
		return EvalResult(self.model.detect([image])[0], ['bg'] + PRODUCTS, image)
