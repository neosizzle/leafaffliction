import os
import argparse
import yaml
from pathlib import Path
import pickle
import json
import numpy as np
import copy
import joblib

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
	required_keys = ['hyper_params'] # TODO: fill this up
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

	# TODO: Pixel Intensity Distribution Analysis

	data = None
	root_path = os.path.dirname(__file__)
	labelled_cache_path = f"{root_path}/../cache/{args.id}/{args.data}"
	with open(labelled_cache_path, "rb") as handle:
		data = pickle.load(handle)

	targets = None
	outcomes_cache_path = f"{root_path}/../cache/{args.id}/{args.outcomes}"
	with open(outcomes_cache_path, "rb") as handle:
		targets = pickle.load(handle)

	classmap = None
	classmap_cache_path = f"{root_path}/../cache/{args.id}/{args.classmap}"
	with open(classmap_cache_path, "rb") as handle:
		classmap = pickle.load(handle)

	classifier = Classifier(params, data, targets, classmap)
	(report, pipeline) = classifier.run_train()
	print(report)

	report_out_path = f"{root_path}/../reports/{args.id}/model_train/"
	report_out_path = os.path.abspath(report_out_path)
	Path(report_out_path).mkdir(parents=True, exist_ok=True)
	def numpy_converter(obj):
		if isinstance(obj, np.generic):
			return obj.item()
		raise TypeError(f"Object of type {obj.__class__} is not JSON serializable")

	with open(f"{report_out_path}/model_performance.json", "w") as f:
		json.dump(report, f, indent=4, default=numpy_converter)

	# NOTE: KerasClassifier is not compatible with joblib.dump.
	# So we save everything except the KerasClassifier model with joblib.dump
	# And save the KerasClassifier model separately
	model_cache_out_path = f"{root_path}/../cache/{args.id}/keras_model.keras"
	model_cache_out_path = os.path.abspath(model_cache_out_path)
	model = pipeline.named_steps["neural"].model_  # type: ignore
	model.save(model_cache_out_path)

	# Now remove the keras model from the pipeline
	pipeline = copy.deepcopy(pipeline)
	# Must remove ALL references to the Keras model
	# Otherwise ImportError ddl will occur when loading the pipeline (in predict.py)
	pipeline.named_steps["neural"].model_ = None  # type: ignore
	pipeline.named_steps["neural"].model = None  # type: ignore
	# pipeline.named_steps["neural"].history_ = None  # remove fit history # type: ignore

	pipeline_cache_path = f"{root_path}/../cache/{args.id}/model.pkl"
	pipeline_cache_path = os.path.abspath(pipeline_cache_path)
	with open(pipeline_cache_path, "wb") as f:
		joblib.dump(pipeline, f)
main()