import os
import argparse
import yaml
import cv2
from pathlib import Path
from plantcv import plantcv as pcv
import pickle

from .classes.classifier import Classifier

def get_args():
	root_path = os.path.dirname(__file__)
	default_config_path = f"{root_path}/Train.yaml"

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_cfg', help='input config', type=str, default=default_config_path)
	parser.add_argument('-id', '--id', help='experiment id for reports and cache', type=str, required=True)
	parser.add_argument(
		"-d",
		"--data",
		help="File path for data (features) pickle",
		type=str,
		default="transformed/data.pkl",
	)
	parser.add_argument(
		"-o",
		"--outcomes",
		help="Outcomes file path",
		type=str,
		default="transformed/target.pkl",
	)
	parser.add_argument(
		"-cm",
		"--classmap",
		help="classmap file path",
		type=str,
		default="transformed/classmap.pkl",
	)


	return parser.parse_args()

def validate_params(data):
	required_keys = ['src'] # TODO: fill this up
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

	df = None
	root_path = os.path.dirname(__file__)
	labelled_cache_path = f"{root_path}/../cache/{args.id}/{args.data}"
	with open(labelled_cache_path, "rb") as handle:
		df = pickle.load(handle)

	targets = None
	outcomes_cache_path = f"{root_path}/../cache/{args.id}/{args.outcomes}"
	with open(outcomes_cache_path, "rb") as handle:
		targets = pickle.load(handle)

	classmap = None
	classmap_cache_path = f"{root_path}/../cache/{args.id}/{args.classmap}"
	with open(classmap_cache_path, "rb") as handle:
		classmap = pickle.load(handle)

	print(df)
	print(targets)
	print(classmap)
	classifier = Classifier(params)
	res = classifier.run()

main()