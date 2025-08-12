import os
import argparse
import yaml
import cv2
from pathlib import Path
from plantcv import plantcv as pcv
import pickle

from .classes.transformer import Transformer, generate_landmarks

def get_args():
	root_path = os.path.dirname(__file__)
	default_config_path = f"{root_path}/Transform.yaml"

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input_cfg', help='input config', type=str, default=default_config_path)
	parser.add_argument('-id', '--id', help='experiment id for reports and cache', type=str, required=True)
	parser.add_argument('-o', '--output_path', help='optional output path', type=str, default=None)
	parser.add_argument('-ni', '--no_img', help='flag to omit image', action='store_true')
	parser.add_argument('-nh', '--no_histogram', help='flag to omit histogram', action='store_true')

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

	transformer = Transformer(params)	
	transformed = transformer.run()
	root_path = os.path.dirname(__file__)
	
	cache_out_path = f"{root_path}/../cache/{args.id}/transformed/" 
	if args.output_path is not None:
		cache_out_path = args.output_path
	cache_out_path = os.path.abspath(cache_out_path)		
	Path(cache_out_path).mkdir(parents=True, exist_ok=True)
	out_cache_data = []
	out_cache_target = []
	class_map = {}

	# create one hot encoding
	i = 0
	for transformed_root in transformed.keys():		
		if transformed_root == '':
			continue
		class_map[transformed_root[1:]] = i
		i += 1

	for transformed_root in transformed.keys():		
		# path create
		final_out_path = f"{cache_out_path}/{transformed_root}"
		Path(final_out_path).mkdir(parents=True, exist_ok=True)

		for entry in transformed[transformed_root]:
			# print(entry.keys())
			filename = entry['filename']
			landmark_img = generate_landmarks(entry['final'].copy(), entry['landmarks'])
			out_cache_data.append(entry['final'])
			# one hot encoding target
			out_cache_target.append(class_map[transformed_root[1:]])


			if not args.no_img:
				# export images
				pcv.print_image(entry['input'], filename=f"{final_out_path}/{filename}")
				pcv.print_image(entry['blur'], filename=f"{final_out_path}/blur_{filename}")
				pcv.print_image(entry['masked'], filename=f"{final_out_path}/mask_{filename}")
				pcv.print_image(entry['final'], filename=f"{final_out_path}/final_{filename}")
				pcv.print_image(entry['size'], filename=f"{final_out_path}/size_{filename}")
				cv2.imwrite(f"{final_out_path}/landmark_{filename}", cv2.cvtColor(landmark_img, cv2.COLOR_RGB2BGR))
				cv2.imwrite(f"{final_out_path}/roi_{filename}", entry['roi'])

			if not args.no_histogram:
				entry['hist_before'].save(f'{final_out_path}/hist_before_{filename}.png')
				entry['hist_after'].save(f'{final_out_path}/hist_after_{filename}.png')
		
	with open(f"{cache_out_path}/data.pkl", 'wb') as handle:
		pickle.dump(out_cache_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open(f"{cache_out_path}/target.pkl", 'wb') as handle:
		pickle.dump(out_cache_target, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open(f"{cache_out_path}/classmap.pkl", 'wb') as handle:
		pickle.dump(class_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
main()