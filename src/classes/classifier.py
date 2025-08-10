import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
from pathlib import Path
from .transformer import Transformer
	
class Classifier:
	def __init__(self, params):
		self.params = params


	def run(self):
		# load from labelled cache and target cache
		train_data_root = self.params['src']

		# convert into tensors
		pass