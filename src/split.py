import os
import argparse
import yaml
from pathlib import Path
import shutil
import random
random.seed(42)

def ft_split(params):
	main_root = Path(params['src'])
	# traverse all folders
	for root, dirs, files in os.walk(main_root):
		# create train and test directories for curr path
		rel_path = root.replace(main_root.as_posix(), "")
		final_train_path = f"{params['out_train']}/{rel_path}"
		Path(final_train_path).mkdir(parents=True, exist_ok=True)
		final_val_path = f"{params['out_val']}/{rel_path}"
		Path(final_val_path).mkdir(parents=True, exist_ok=True)

		# split the contents
		ratio = params['ratio']
		files_copy = files[:]  # Copy to avoid modifying original
		random.shuffle(files_copy)
		split_index = int(len(files_copy) * ratio)
		train_data = files_copy[:split_index]
		val_data = files_copy[split_index:]

		# copy the files
		for filename in val_data:
			shutil.copy2(f"{root}/{filename}", final_val_path)
		for filename in train_data:
			shutil.copy2(f"{root}/{filename}", final_train_path)

	print("OK")

def get_args():
	root_path = os.path.dirname(__file__)
	default_config_path = f"{root_path}/split.yaml"

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_cfg', help='input config', type=str, default=default_config_path)
	return parser.parse_args()

def validate_params(data):
	required_keys = ['src', 'out_train', 'out_val', 'ratio']
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
	ft_split(params)

main()