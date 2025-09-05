import os
import argparse
import yaml
from pathlib import Path
import pickle
import json
import numpy as np
import copy
import joblib
from plantcv import plantcv as pcv
import tensorflow as tf
import sys

from .classes.classifier import Classifier
from .classes.transformer import Transformer

def load_keras_pipeline(pipeline_cache_path, keras_model_cache_path):
	"""
	Load the Keras model from the pipeline cache and return it.
	"""

	import joblib

	pipeline = joblib.load(pipeline_cache_path)

	keras_model = tf.keras.models.load_model(keras_model_cache_path)  # type: ignore

	pipeline.named_steps["neural"].model_ = keras_model
	return pipeline

def get_args():
	root_path = os.path.dirname(__file__)
	default_config_path = f"{root_path}/Predict.yaml"

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_cfg', help='input config', type=str, default=default_config_path)
	parser.add_argument('-id', '--id', help='experiment id for reports and cache', type=str, required=True)
	parser.add_argument(
		"-cm",
		"--classmap",
		help="classmap file path",
		type=str,
		default="transformed/classmap.pkl",
	)
	parser.add_argument(
		"-m",
		"--model",
		help="File path for model pickle",
		type=str,
		default="model.pkl",
	)
	parser.add_argument(
		"-km",
		"--keras_model",
		help="File path for Keras model",
		type=str,
		default="keras_model.keras",
	)


	return parser.parse_args()

def validate_params(data):
	required_keys = ['transformer_params']
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

	transformer = Transformer(params=params['transformer_params'])
	transformed = transformer.run()

	root_path = os.path.dirname(__file__)
	classmap = None
	classmap_cache_path = f"{root_path}/../cache/{args.id}/{args.classmap}"
	with open(classmap_cache_path, "rb") as handle:
		classmap = pickle.load(handle)

	features = []
	targets = []
	public_img_path = f"{root_path}/../public/images"
	public_img_path = os.path.abspath(public_img_path)
	Path(public_img_path).mkdir(parents=True, exist_ok=True)
	id = 0
	for transformed_root in transformed.keys():
		for entry in transformed[transformed_root]:
			features.append(entry['final'])
			targets.append(classmap[transformed_root[1:]])
			pcv.print_image(entry['input'], filename=f"{public_img_path}/input_{id}.png")
			pcv.print_image(entry['final'], filename=f"{public_img_path}/transformed_{id}.png")
			id += 1

	# Load Model Sequentially
	pipeline_cache_path = os.path.abspath(
		f"{root_path}/../cache/{args.id}/{args.model}"
	)

	keras_model_cache_path = os.path.abspath(
		f"{root_path}/../cache/{args.id}/{args.keras_model}"
	)

	model = load_keras_pipeline(pipeline_cache_path, keras_model_cache_path)


	classifier = Classifier(params, features, targets, classmap, pred_model=model)
	(report, predictions) = classifier.run_predict()
	print(report)

	report_out_path = f"{root_path}/../reports/{args.id}/model_predict/"
	report_out_path = os.path.abspath(report_out_path)
	Path(report_out_path).mkdir(parents=True, exist_ok=True)
	def numpy_converter(obj):
		if isinstance(obj, np.generic):
			return obj.item()
		raise TypeError(f"Object of type {obj.__class__} is not JSON serializable")

	with open(f"{report_out_path}/model_performance.json", "w") as f:
		json.dump(report, f, indent=4, default=numpy_converter)

	render_json_out_path = f"{root_path}/../public"
	render_json_out_path = os.path.abspath(render_json_out_path)
	Path(render_json_out_path).mkdir(parents=True, exist_ok=True)
	with open(f"{render_json_out_path}/classmap.json", "w") as f:
		json.dump(classmap, f, indent=4, default=numpy_converter)

	with open(f"{render_json_out_path}/pred.json", "w") as f:
		json.dump({'pred': predictions.tolist()}, f, indent=4, default=numpy_converter)

	os.execve(sys.executable, [sys.executable, "-m", "http.server", "8080", "--directory", "public"], os.environ)
	# # NOTE: KerasClassifier is not compatible with joblib.dump.
	# # So we save everything except the KerasClassifier model with joblib.dump
	# # And save the KerasClassifier model separately
	# model_cache_out_path = f"{root_path}/../cache/{args.id}/keras_model.keras"
	# model_cache_out_path = os.path.abspath(model_cache_out_path)
	# model = pipeline.named_steps["neural"].model_  # type: ignore
	# model.save(model_cache_out_path)

	# # Now remove the keras model from the pipeline
	# pipeline = copy.deepcopy(pipeline)
	# # Must remove ALL references to the Keras model
	# # Otherwise ImportError ddl will occur when loading the pipeline (in predict.py)
	# pipeline.named_steps["neural"].model_ = None  # type: ignore
	# pipeline.named_steps["neural"].model = None  # type: ignore
	# # pipeline.named_steps["neural"].history_ = None  # remove fit history # type: ignore

	# pipeline_cache_path = f"{root_path}/../cache/{args.id}/model.pkl"
	# pipeline_cache_path = os.path.abspath(pipeline_cache_path)
	# with open(pipeline_cache_path, "wb") as f:
	# 	joblib.dump(pipeline, f)
main()