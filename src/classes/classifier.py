import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
from pathlib import Path
from .transformer import Transformer
	
PIXEL_MAX = 255

class Classifier:
	def __init__(self, params, data, targets, classmap):
		self.params = params
		self.data = data
		self.targets = targets
		self.classmap = classmap

	def run(self):
		# min max normalization with known max value
		input_data = np.divide(self.data, PIXEL_MAX)
		print(input_data)
		train_data_root = self.params['src']

		# convert into tensors
		pass