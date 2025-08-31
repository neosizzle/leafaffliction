from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from pathlib import Path
from .transformer import Transformer

PIXEL_MAX = 255

def build_cnn():
	inputs = Input(shape=(256, 256, 3))  # Define the input layer explicitly
	
	# Normalize the inputs by scaling pixel values from [0, 255] to [0, 1]
	x = layers.Rescaling(1./255)(inputs)
	
	# First convolutional block
	x = layers.Conv2D(32, (3, 3), activation='relu')(x)
	x = layers.MaxPooling2D((2, 2))(x)
	x = layers.Dropout(0.5)(x)

	# Second convolutional block
	x = layers.Conv2D(64, (3, 3), activation='relu')(x)
	x = layers.MaxPooling2D((2, 2))(x)

	# Third convolutional block
	x = layers.Conv2D(128, (3, 3), activation='relu')(x)
	x = layers.MaxPooling2D((2, 2))(x)

	# Flatten the output from convolutional layers
	x = layers.Flatten()(x)

	# Fully connected dense layer
	x = layers.Dense(64, activation='relu')(x)

	# Output layer with 8 classes and softmax activation
	outputs = layers.Dense(9, activation='softmax')(x)

	model = models.Model(inputs=inputs, outputs=outputs)
	return model

class Classifier:
	def __init__(self, params, data, targets, classmap, pred_model=None):
		self.params = params
		self.data = np.array(data)
		self.targets = np.array(targets)	
		self.classmap = classmap
		self.pred_model = pred_model

	# return the model
	def run(self):
		# min max normalization with known max value
		input_data = self.data
		target = self.targets
		
		early_stopping = EarlyStopping(
			monitor='val_loss',
			patience=5,
			min_delta=0.001,
			mode='min',
			restore_best_weights=True,
			verbose=1
		)
		model = build_cnn()
		model.summary()
		model.compile(
			optimizer='adam',
			loss='sparse_categorical_crossentropy',
			metrics=['accuracy']
		)
		model.fit(
			input_data,
			target,
			epochs=5,
			batch_size=32,
			validation_split=0.2,
			callbacks=[early_stopping]
			)
		

		# print(input_data.shape)
		# print(self.classmap)

		# convert into tensors
		pass