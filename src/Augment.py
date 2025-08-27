import os
import argparse
import yaml
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import random
random.seed(42)

augment_code_map = {
	'random': 'random',
	'all': 'all',
	'f': 'Flip',
	'r': 'Rotate',
	's': 'Shear',
	'c': 'Contrast',
	'z': 'Zoom',
	'd': 'Distortion'
}

def select_files(n, files_list):
	length = len(files_list)
	if n <= length:
		# Sample without replacement (unique elements)
		return random.sample(files_list, n)
	else:
		# Sample with replacement (repeats allowed)
		return random.choices(files_list, k=n)

def check_and_print_image_metadata(file_path):
	try:
		with Image.open(file_path) as img:
			img.verify()  # Verify if it is an image
		with Image.open(file_path) as img:
			print(f"Image format: {img.format}")
			print(f"Image size: {img.size}")
			print(f"Image mode: {img.mode}")
			print(f"Image info metadata: {img.info}")
			print("========")
	except (IOError, SyntaxError) as e:
		print("File is not a valid image or cannot be opened.")

def run_augments(params):
	dir = Path(params['dir'])
	mode = params['mode']
	count_req = params['count']

	if mode not in augment_code_map.keys():
		raise ValueError(f"unknown mode {mode}: expected {augment_code_map.keys()}")

	# assume that the user always provides a valid file path
	existing_files = os.listdir(dir)
	
	# do nothing if requirement is fulfilled
	if len(existing_files) >= count_req:
		return

	n_files_left = count_req - len(existing_files)

	# for modes except 'all', they will need to pick from the existing files to fulfill all requirements
	aug_filters = {k: v for k, v in augment_code_map.items() if k not in ('random', 'all')}
	n_aug_filters = len(aug_filters.keys())
	source_files = select_files(n_aug_filters * (n_files_left // n_aug_filters) if mode == 'all' else n_files_left, existing_files)
	
	while n_files_left > 0:
		source = os.path.abspath(f"{dir}/{source_files[n_files_left - 1]}")
		# NOTE: assumes there is only 1 '.' in filename
		tokens = source.split(".")

		# mode is 'all', use all filters and add n_aug_filters to progress
		if mode == 'all':
			image = Image.open(source) 
			image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
			image_rot = image.rotate(90)
			image_shear = image.transform(image.size, Image.AFFINE, (1, 0.1, 0, 0, 1, 0))
			image_contrast = ImageEnhance.Contrast(image).enhance(2.0)
			image_zoom = image.crop((10, 10, image.width - 10, image.height - 10)).resize(image.size, Image.Resampling.LANCZOS)
			image_distort = image.filter(ImageFilter.SHARPEN)
			
			image_flip.save('.'.join([tokens[0] + "_Flip", tokens[1]]))
			image_rot.save('.'.join([tokens[0] + "_Rotate", tokens[1]]))
			image_shear.save('.'.join([tokens[0] + "_Shear", tokens[1]]))
			image_contrast.save('.'.join([tokens[0] + "_Contrast", tokens[1]]))
			image_zoom.save('.'.join([tokens[0] + "_Zoom", tokens[1]]))
			image_distort.save('.'.join([tokens[0] + "_Distort", tokens[1]]))
			
			n_files_left -=  n_aug_filters
		else:
			image = Image.open(source) 
			aug = aug_filters[mode]
			if mode == 'random':
				aug = random.choice(list(aug_filters.values()))
			
			if aug == 'Flip':
				image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
				image_flip.save('.'.join([tokens[0] + "_Flip", tokens[1]]))
			elif aug == 'Rotate':
				image_rot = image.rotate(90)
				image_rot.save('.'.join([tokens[0] + "_Rotate", tokens[1]]))
			elif aug == 'Shear':
				image_shear = image.transform(image.size, Image.AFFINE, (1, 0.1, 0, 0, 1, 0))
				image_shear.save('.'.join([tokens[0] + "_Shear", tokens[1]]))
			elif aug == 'Contrast':
				image_contrast = ImageEnhance.Contrast(image).enhance(2.0)
				image_contrast.save('.'.join([tokens[0] + "_Contrast", tokens[1]]))
			elif aug == 'Zoom':
				image_zoom = image.crop((10, 10, image.width - 10, image.height - 10)).resize(image.size, Image.Resampling.LANCZOS)
				image_zoom.save('.'.join([tokens[0] + "_Zoom", tokens[1]]))
			elif aug == 'Distortion':
				image_distort = image.filter(ImageFilter.SHARPEN)
				image_distort.save('.'.join([tokens[0] + "_Distort", tokens[1]]))
			else:
				print(aug)
				raise ValueError("Why are you cooking me bro?")

			n_files_left -= 1
	
def get_args():
	root_path = os.path.dirname(__file__)
	default_config_path = f"{root_path}/Augment.yaml"

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_cfg', help='input config', type=str, default=default_config_path)
	return parser.parse_args()

def validate_params(data):
	required_keys = ['dir', 'mode', 'count']
	missing = [key for key in required_keys if key not in data]
	if missing:
		raise ValueError(f"{missing} is missing in config")

def main():
	args = get_args()
	param_file = args.input_cfg

	params = None
	with open(param_file, 'r') as file:
		params = yaml.safe_load(file)

	validate_params(params)
	path = params['dir']
	run_augments(params)

	print(f"Images are balanced at {path}")

main()