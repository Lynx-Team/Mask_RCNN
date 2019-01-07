import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import PIL
from mrcnn.config import Config # pylint: disable=all
from mrcnn import model as modellib, utils, visualize # pylint: disable=all

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


def ldmodel(is_training, load_last):
	config = ProductConfig()
	mode = 'training' if is_training else 'inference'
	print('Mode ' + mode)
	model = modellib.MaskRCNN(mode=mode, config=config, model_dir=DEFAULT_LOGS_DIR)
	if load_last:
		weights_path = model.find_last()
		print('Loading weights ', weights_path)
		model.load_weights(weights_path, by_name=True)
	else:
		#print('Loading weights ', model.get_imagenet_weights())
		print('Loading weights ', COCO_WEIGHTS_PATH)
		model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=[
			'mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
	return model

def train(model, dataset_path):
	config = ProductConfig()
	dataset_train = ProductDataset()
	dataset_train.load_product(dataset_path, 'train')
	dataset_train.prepare()
	dataset_val = ProductDataset()
	dataset_val.load_product(dataset_path, 'test')
	dataset_val.prepare()
	print('Training network heads')
	model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='heads')

def test(model, img_path):
	image = skimage.io.imread(img_path)
	results = model.detect([image], verbose=1)
	r = results[0]
	prd = ['bg'] + PRODUCTS
	visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], prd, r['scores'])

# model = ldmodel(is_training=False, load_last=True)
# test(model, 'images\\products.jpg')

#model = ldmodel(is_training=True, load_last=False)
#train(model, 'some dataset')