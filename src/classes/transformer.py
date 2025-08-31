import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
import threading
from pathlib import Path


def add_landmark(img, landmark, color):
	x = landmark[0][0]
	y = landmark[0][1]
	cv2.circle(img, (int(x), int(y)), radius=5, color=color, thickness=-1)

def generate_landmarks(img_with_landmarks, landmark_obj):
	for landmark in landmark_obj['top_landmarks']:
		add_landmark(img_with_landmarks, landmark, (0, 0, 255))

	for landmark in landmark_obj['btm_landmarks']:
		add_landmark(img_with_landmarks, landmark, (0, 255, 0))

	for landmark in landmark_obj['center_w']:
		add_landmark(img_with_landmarks, landmark, (255, 0, 0))

	for landmark in landmark_obj['left_landmarks']:
		add_landmark(img_with_landmarks, landmark, (0, 0, 255))

	for landmark in landmark_obj['right_landmarks']:
		add_landmark(img_with_landmarks, landmark, (0, 255, 0))

	for landmark in landmark_obj['center_h']:
		add_landmark(img_with_landmarks, landmark, (255, 0, 0))

	return img_with_landmarks
	
class Transformer:
	def __init__(self, params):
		self.params = params

	# VIS (Visible Light Image) image processing
	# PSII Image (Photosystem II Fluorescence Image) not supported 
	# https://plantcv.readthedocs.io/en/v3.2.0/vis_tutorial/
	def image_process(self, image_path):
		res = {}
				
		img, path, filename = pcv.readimage(image_path, mode='rgb')
		res['filename'] = filename
		res['input'] = img


		# transforming the image to grayscale
		# is needed to make a binary image, which is required for object segmentation (bg removal)
		# this method is converting RGB to HSV and extract the saturation channel
		s = pcv.rgb2gray_hsv(img, 's')

		# binarize the image here, the threshold alpha can be tuned
		threshold_alpha = self.params['thres_a']
		s_thresh = pcv.threshold.binary(s, threshold_alpha, 'light')

		# add some blur to smoothen noise
		gaussian_k = self.params['gaus_k']
		s_mblur = pcv.gaussian_blur(s_thresh, (gaussian_k, gaussian_k))

		res['blur'] = s_mblur

		# # Use another channel (LAB) to do grayscaling and join with the origninal
		# # grayscale method if we missed any features... needed?
		# b = pcv.rgb2gray_lab(img, 'b')
		# b_thresh = pcv.threshold.binary(b, 160, 255, 'light')
		# bs = pcv.logical_or(s_mblur, b_thresh)

		# remove background, yayyyy
		masked = pcv.apply_mask(img, s_mblur, 'white')

		res['masked'] = masked

		# repeat something similar to clear up and remaining
		# background artifacts
		masked_a = pcv.rgb2gray_lab(masked, 'a')
		masked_b = pcv.rgb2gray_lab(masked, 'b')

		artifact_thres_a = self.params['artifact_thres_a']
		artifact_thres_a2 = self.params['artifact_thres_a2']
		artifact_thres_a3 = self.params['artifact_thres_a3']

		maskeda_thresh = pcv.threshold.binary(masked_a, artifact_thres_a, 'dark')
		maskeda_thresh1 = pcv.threshold.binary(masked_a, artifact_thres_a2, 'light')
		maskedb_thresh = pcv.threshold.binary(masked_b, artifact_thres_a3, 'light')

		# join the masked images together
		ab1 = pcv.logical_or(maskeda_thresh, maskedb_thresh)
		ab = pcv.logical_or(maskeda_thresh1, ab1)

		# fill in small objects, we are confident enough that the
		# pixels filled are insignificant now after all the filtering
		fill_n = self.params['fill_n']
		ab_fill = pcv.fill(ab, fill_n)

		# now we are manually tagging the objects 
		roi_x = self.params['roi_x']
		roi_y = self.params['roi_y']
		roi_width = self.params['roi_width']
		roi_height = self.params['roi_height']
		roi = pcv.roi.rectangle(img=ab_fill, x=roi_x, y=roi_y, h=roi_height, w=roi_width)
		roi_mask = pcv.roi.filter(mask=ab_fill, roi=roi, roi_type='partial')

		# reapply mask
		masked2 = pcv.apply_mask(masked, roi_mask, 'white')

		res['final'] = masked2

		# examine roi rectangle
		img_with_roi = masked2.copy()
		cv2.rectangle(img_with_roi, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), color=(255, 0, 0), thickness=2)
		res['roi'] = img_with_roi

		# this is to make validate our work so far.. the borders
		# should match the target item we are trying to analyze
		size_analysis = pcv.analyze.size(img=img, labeled_mask=roi_mask)
		res['size'] = size_analysis

		# what landmarks are https://pmc.ncbi.nlm.nih.gov/articles/PMC5713628/
		left_landmarks, right_landmarks, center_h = pcv.homology.y_axis_pseudolandmarks(img=masked2, mask=roi_mask)
		top_landmarks, btm_landmarks, center_w = pcv.homology.x_axis_pseudolandmarks(img=masked2, mask=roi_mask)
		landmark_obj = {
			'left_landmarks': left_landmarks,
			'right_landmarks': right_landmarks,
			'center_h': center_h,
			'top_landmarks': top_landmarks,
			'btm_landmarks': btm_landmarks,
			'center_w':	center_w
		}
		res['landmarks'] = landmark_obj
		
		# color_analysis = pcv.analyze.color(rgb_img=img, labeled_mask=roi_mask, n_labels=1, colorspaces='all')
		# color_analysis.save(f'color.png')

		hist_figure, _ = pcv.visualize.histogram(img=img, hist_data=True)
		res['hist_after'] = hist_figure

		hist_figure, _ = pcv.visualize.histogram(img=masked2, hist_data=True)
		res['hist_before'] = hist_figure

		return res


	def run(self):
		path = Path(self.params['src'])

		res = {}


		if path.is_dir():
			# this is already doing recursive deep search
			for root, dirs, files in os.walk(path):
				rel_path = root.replace(path.as_posix(), "")
				res[rel_path] = []
				print(f"{root}")
				thread_results = [None] * 4
				split_files = [files[i::4] for i in range(4)]
				
				def process_files_in_thread(thread_index, files_subset):
					results = []
					for file in files_subset:
						file_path = os.path.join(root, file)
						print(f"{file_path}")
						processed = self.image_process(file_path)
						results.append(processed)
					thread_results[thread_index] = results

				threads = []
				for i in range(4):
					t = threading.Thread(target=process_files_in_thread, args=(i, split_files[i]))
					threads.append(t)
					t.start()

				for t in threads:
					t.join()

				# join all results
				for i in thread_results:
					for j in i:
						res[rel_path].append(j)
				# for file in files:
				# 	file_path = os.path.join(root, file)
				# 	print(f"{file_path}")
				# 	processed = self.image_process(file_path)
				# 	# res[rel_path].append(processed)
				# 	# time.sleep(1)
			return res	
		elif path.is_file():
			res = {'': []}
			processed = self.image_process(path)
			res[''].append(processed)
			return res
		else:
			raise ValueError(f"{path} does not exist")
