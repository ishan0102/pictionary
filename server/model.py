# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Binary for trianing a RNN-based classifier for the Quick, Draw! data.

python train_model.py \
  --training_data train_data \
  --eval_data eval_data \
  --model_dir /tmp/quickdraw_model/ \
  --cell_type cudnn_lstm

When running on GPUs using --cell_type cudnn_lstm is much faster.

The expected performance is ~75% in 1.5M steps with the default self, configuration.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import ast
import functools
import sys

from datetime import datetime
import json
import numpy as np


import tensorflow as tf
import pickle


class SketchClassifier:
	def __init__(self):
		self.FLAGS = {
			'batch_norm': False, 
			'batch_size': 8, 
			'cell_type': 'lstm', 
			'classes_file': 'data/training.tfrecord.classes', 
			'conv_len': '[5, 5, 3]', 
			'dropout': 0.3, 
			'eval_data': './data/eval.tfrecord-?????-of-?????', 
			'gradient_clipping_norm': 9.0, 
			'learning_rate': 0.0001, 
			'model_dir': '../models/', 
			'num_conv': '[48, 64, 96]', 
			'num_layers': 3, 
			'num_nodes': 128, 
			'predict_for_data': '[[[73,66,46,23,12,11,22,48,58,67,70,65],[11,6,2,10,23,33,48,56,54,41,22,10]],[[66,85,71],[9,3,26]],[[24,1,2,8],[6,1,10,19]],[[64,88,134,176,180,184,184,174,111,63,47],[34,29,28,35,39,58,91,94,86,71,62]],[[64,61,62],[74,83,102]],[[83,84,87],[78,102,107]],[[157,159,164],[96,108,116]],[[175,182],[91,115]],[[182,186,198,209,223,234,251,255],[51,36,29,30,38,39,20,8]],[[157,136,128,133,139],[35,47,57,35,29]],[[104,94,84,84,89],[40,52,70,30,26]],[[111,105,105,109,121],[30,59,68,72,34]],[[159,153,153],[41,54,65]]]', 
			'predict_temp_file': './predict_temp.tfrecord', 
			'test': True, 
			'steps': 120000, 
			'training_data': './data/training.tfrecord-?????-of-?????'
		}

	def get_num_classes(self):
		classes = []
		with tf.gfile.GFile(self.FLAGS['classes_file'], "r") as f:
			classes = [x for x in f]
		num_classes = len(classes)
		return num_classes


	def get_input_fn(self, mode, tfrecord_pattern, batch_size):
		"""Creates an input_fn that stores all the data in memory.

		Args:
		mode: one of tf.contrib.learn.ModeKeys.{TRAIN, INFER, EVAL}
		tfrecord_pattern: path to a TF record file created using create_dataset.py.
		batch_size: the batch size to output.

		Returns:
			A valid input_fn for the model estimator.
		"""

		def _parse_tfexample_fn(example_proto, mode):
			"""Parse a single record which is expected to be a tensorflow.Example."""
			feature_to_type = {
					"ink": tf.VarLenFeature(dtype=tf.float32),
					"shape": tf.FixedLenFeature([2], dtype=tf.int64)
			}
			if mode != tf.estimator.ModeKeys.PREDICT:
				# The labels won't be available at inference time, so don't add them
				# to the list of feature_columns to be read.
				feature_to_type["class_index"] = tf.FixedLenFeature([1], dtype=tf.int64)

			parsed_features = tf.parse_single_example(example_proto, feature_to_type)
			parsed_features["ink"] = tf.sparse_tensor_to_dense(parsed_features["ink"])

			if mode != tf.estimator.ModeKeys.PREDICT:
				labels = parsed_features["class_index"]
				return parsed_features, labels
			else:
				return parsed_features  # In prediction, we have no labels

		def _input_fn():
			"""Estimator `input_fn`.

			Returns:
				A tuple of:
				- Dictionary of string feature name to `Tensor`.
				- `Tensor` of target labels.
			"""
			dataset = tf.data.TFRecordDataset.list_files(tfrecord_pattern)
			if mode == tf.estimator.ModeKeys.TRAIN:
				dataset = dataset.shuffle(buffer_size=10)
			dataset = dataset.repeat()
			# Preprocesses 10 files concurrently and interleaves records from each file.
			dataset = dataset.interleave(
					tf.data.TFRecordDataset,
					cycle_length=10,
					block_length=1)
			dataset = dataset.map(
					functools.partial(_parse_tfexample_fn, mode=mode),
					num_parallel_calls=10)
			dataset = dataset.prefetch(10000)
			if mode == tf.estimator.ModeKeys.TRAIN:
				dataset = dataset.shuffle(buffer_size=1000000)
			# Our inputs are variable length, so pad them.
			dataset = dataset.padded_batch(
					batch_size, padded_shapes=dataset.output_shapes)

			iter = dataset.make_one_shot_iterator()
			if mode != tf.estimator.ModeKeys.PREDICT:
					features, labels = iter.get_next()
					return features, labels
			else:
					features = iter.get_next()
					return features, None  # In prediction, we have no labels

		return _input_fn


	def model_fn(self, features, labels, mode, params):
		"""Model function for RNN classifier.

		This function sets up a neural network which applies convolutional layers (as
		configured with params.num_conv and params.conv_len) to the input.
		The output of the convolutional layers is given to LSTM layers (as configured
		with params.num_layers and params.num_nodes).
		The final state of the all LSTM layers are concatenated and fed to a fully
		connected layer to obtain the final classification scores.

		Args:
			features: dictionary with keys: inks, lengths.
			labels: one hot encoded classes
			mode: one of tf.estimator.ModeKeys.{TRAIN, INFER, EVAL}
			params: a parameter dictionary with the following keys: num_layers,
				num_nodes, batch_size, num_conv, conv_len, num_classes, learning_rate.

		Returns:
			ModelFnOps for Estimator API.
		"""

		def _get_input_tensors(features, labels):
			"""Converts the input dict into inks, lengths, and labels tensors."""
			# features[ink] is a sparse tensor that is [8, batch_maxlen, 3]
			# inks will be a dense tensor of [8, maxlen, 3]
			# shapes is [batchsize, 2]
			shapes = features["shape"]
			# lengths will be [batch_size]
			lengths = tf.squeeze(
					tf.slice(shapes, begin=[0, 0], size=[params.batch_size, 1]))
			inks = tf.reshape(features["ink"], [params.batch_size, -1, 3])
			if labels is not None:
				labels = tf.squeeze(labels)
			return inks, lengths, labels

		def _add_conv_layers(inks, lengths):
			"""Adds convolution layers."""
			convolved = inks
			for i in range(len(params.num_conv)):
				convolved_input = convolved
				if params.batch_norm:
					convolved_input = tf.layers.batch_normalization(
							convolved_input,
							training=(mode == tf.estimator.ModeKeys.TRAIN))
				# Add dropout layer if enabled and not first convolution layer.
				if i > 0 and params.dropout:
					convolved_input = tf.layers.dropout(
							convolved_input,
							rate=params.dropout,
							training=(mode == tf.estimator.ModeKeys.TRAIN))
				convolved = tf.layers.conv1d(
						convolved_input,
						filters=params.num_conv[i],
						kernel_size=params.conv_len[i],
						activation=None,
						strides=1,
						padding="same",
						name="conv1d_%d" % i)
			return convolved, lengths

		def _add_regular_rnn_layers(convolved, lengths):
			"""Adds RNN layers."""
			if params.cell_type == "lstm":
				cell = tf.nn.rnn_cell.BasicLSTMCell
			elif params.cell_type == "block_lstm":
				cell = tf.contrib.rnn.LSTMBlockCell
			cells_fw = [cell(params.num_nodes) for _ in range(params.num_layers)]
			cells_bw = [cell(params.num_nodes) for _ in range(params.num_layers)]
			if params.dropout > 0.0:
				cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
				cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
			outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
					cells_fw=cells_fw,
					cells_bw=cells_bw,
					inputs=convolved,
					sequence_length=lengths,
					dtype=tf.float32,
					scope="rnn_classification")
			return outputs

		def _add_cudnn_rnn_layers(convolved):
			"""Adds CUDNN LSTM layers."""
			# Convolutions output [B, L, Ch], while CudnnLSTM is time-major.
			convolved = tf.transpose(convolved, [1, 0, 2])
			lstm = tf.contrib.cudnn_rnn.CudnnLSTM(
					num_layers=params.num_layers,
					num_units=params.num_nodes,
					dropout=params.dropout if mode == tf.estimator.ModeKeys.TRAIN else 0.0,
					direction="bidirectional")
			outputs, _ = lstm(convolved)
			# Convert back from time-major outputs to batch-major outputs.
			outputs = tf.transpose(outputs, [1, 0, 2])
			return outputs

		def _add_rnn_layers(convolved, lengths):
			"""Adds recurrent neural network layers depending on the cell type."""
			if params.cell_type != "cudnn_lstm":
				outputs = _add_regular_rnn_layers(convolved, lengths)
			else:
				outputs = _add_cudnn_rnn_layers(convolved)
			# outputs is [batch_size, L, N] where L is the maximal sequence length and N
			# the number of nodes in the last layer.
			mask = tf.tile(
					tf.expand_dims(tf.sequence_mask(lengths, tf.shape(outputs)[1]), 2),
					[1, 1, tf.shape(outputs)[2]])
			zero_outside = tf.where(mask, outputs, tf.zeros_like(outputs))
			outputs = tf.reduce_sum(zero_outside, axis=1)
			return outputs

		def _add_fc_layers(final_state):
			"""Adds a fully connected layer."""
			return tf.layers.dense(final_state, params.num_classes)

		# Build the model.
		inks, lengths, labels = _get_input_tensors(features, labels)
		convolved, lengths = _add_conv_layers(inks, lengths)
		final_state = _add_rnn_layers(convolved, lengths)
		logits = _add_fc_layers(final_state)

		# Compute current predictions.
		predictions = tf.argmax(logits, axis=1)

		if mode == tf.estimator.ModeKeys.PREDICT:
				preds = {
						"class_index": predictions,
						#"class_index": predictions[:, tf.newaxis],
						"probabilities": tf.nn.softmax(logits),
						"logits": logits
				}
				#preds = {"logits": logits, "predictions": predictions}

				return tf.estimator.EstimatorSpec(mode, predictions=preds)
				# Add the loss.
		cross_entropy = tf.reduce_mean(
				tf.nn.sparse_softmax_cross_entropy_with_logits(
						labels=labels, logits=logits))

		# Add the optimizer.
		train_op = tf.contrib.layers.optimize_loss(
				loss=cross_entropy,
				global_step=tf.train.get_global_step(),
				learning_rate=params.learning_rate,
				optimizer="Adam",
				# some gradient clipping stabilizes training in the beginning.
				clip_gradients=params.gradient_clipping_norm,
				summaries=["learning_rate", "loss", "gradients", "gradient_norm"])

		return tf.estimator.EstimatorSpec(
				mode=mode,
				predictions={"logits": logits, "predictions": predictions},
				loss=cross_entropy,
				train_op=train_op,
				eval_metric_ops={"accuracy": tf.metrics.accuracy(labels, predictions)})


	def create_estimator_and_specs(self, run_config):
		"""Creates an Experiment configuration based on the estimator and input fn."""
		model_params = tf.contrib.training.HParams(
				num_layers=self.FLAGS['num_layers'],
				num_nodes=self.FLAGS['num_nodes'],
				batch_size=self.FLAGS['batch_size'],
				num_conv=ast.literal_eval(self.FLAGS['num_conv']),
				conv_len=ast.literal_eval(self.FLAGS['conv_len']),
				num_classes=self.get_num_classes(),
				learning_rate=self.FLAGS['learning_rate'],
				gradient_clipping_norm=self.FLAGS['gradient_clipping_norm'],
				cell_type=self.FLAGS['cell_type'],
				batch_norm=self.FLAGS['batch_norm'],
				dropout=self.FLAGS['dropout'])

		estimator = tf.estimator.Estimator(
				model_fn=self.model_fn,
				config=run_config,
				params=model_params)

		train_spec = tf.estimator.TrainSpec(input_fn=self.get_input_fn(
				mode=tf.estimator.ModeKeys.TRAIN,
				tfrecord_pattern=self.FLAGS['training_data'],
				batch_size=self.FLAGS['batch_size']), max_steps=self.FLAGS['steps'])

		eval_spec = tf.estimator.EvalSpec(input_fn=self.get_input_fn(
				mode=tf.estimator.ModeKeys.EVAL,
				tfrecord_pattern=self.FLAGS['eval_data'],
				batch_size=self.FLAGS['batch_size']))

		return estimator, train_spec, eval_spec


	# def main(self, unused_args):
	#   estimator, train_spec, eval_spec = create_estimator_and_specs(
	#       run_config=tf.estimator.RunConfig(
	#           model_dir=self.FLAGS['model_dir'],
	#           save_checkpoints_secs=300,
	#           save_summary_steps=100))
	#   tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
	def create_tfrecord_for_prediction(self, batch_size, stroke_data, tfrecord_file):
			def parse_line(stroke_data):
					"""Parse provided stroke data and ink (as np array) and classname."""
					inkarray = json.loads(stroke_data)
					stroke_lengths = [len(stroke[0]) for stroke in inkarray]
					total_points = sum(stroke_lengths)
					np_ink = np.zeros((total_points, 3), dtype=np.float32)
					current_t = 0
					for stroke in inkarray:
							if len(stroke[0]) != len(stroke[1]):
									print("Inconsistent number of x and y coordinates.")
									return None
							for i in [0, 1]:
									np_ink[current_t:(current_t + len(stroke[0])), i] = stroke[i]
							current_t += len(stroke[0])
							np_ink[current_t - 1, 2] = 1  # stroke_end
					# Preprocessing.
					# 1. Size normalization.
					lower = np.min(np_ink[:, 0:2], axis=0)
					upper = np.max(np_ink[:, 0:2], axis=0)
					scale = upper - lower
					scale[scale == 0] = 1
					np_ink[:, 0:2] = (np_ink[:, 0:2] - lower) / scale
					# 2. Compute deltas.
					#np_ink = np_ink[1:, 0:2] - np_ink[0:-1, 0:2]
					#np_ink = np_ink[1:, :]
					np_ink[1:, 0:2] -= np_ink[0:-1, 0:2]
					np_ink = np_ink[1:, :]

					features = {}
					features["ink"] = tf.train.Feature(float_list=tf.train.FloatList(value=np_ink.flatten()))
					features["shape"] = tf.train.Feature(int64_list=tf.train.Int64List(value=np_ink.shape))
					f = tf.train.Features(feature=features)
					ex = tf.train.Example(features=f)
					return ex

			if stroke_data is None:
					print("Error: Stroke data cannot be none")
					return

			example = parse_line(stroke_data)

			#Remove the file if it already exists
			if tf.gfile.Exists(tfrecord_file):
					tf.gfile.Remove(tfrecord_file)

			writer = tf.python_io.TFRecordWriter(tfrecord_file)
			for i in range(batch_size):
					writer.write(example.SerializeToString())
			writer.flush()
			writer.close()
			print ('wrote',tfrecord_file)

	def get_classes(self, ):
		classes = []
		with tf.gfile.GFile(self.FLAGS['classes_file'], "r") as f:
			classes = [x.rstrip() for x in f]
		return classes

	def main(self, unused_args):
		print("%s: I Starting application" % (datetime.now()))
		print("self.FLAGS", self.FLAGS)
		estimator, train_spec, eval_spec = self.create_estimator_and_specs(
				run_config=tf.estimator.RunConfig(
						model_dir=self.FLAGS['model_dir'],
						save_checkpoints_secs=300,
						save_summary_steps=100))
		print("estimator",estimator,"train_spec",train_spec,"eval_spec",eval_spec) 
		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

		if self.FLAGS['predict_for_data'] != None:
				print("%s: I Starting prediction" % (datetime.now()))
				class_names = self.get_classes()
				self.create_tfrecord_for_prediction(self.FLAGS['batch_size'], self.FLAGS['predict_for_data'], self.FLAGS['predict_temp_file'])
				predict_results = estimator.predict(input_fn=self.get_input_fn(
						mode=tf.estimator.ModeKeys.PREDICT,
						tfrecord_pattern=self.FLAGS['predict_temp_file'],
						batch_size=self.FLAGS['batch_size']))

				#predict_results = estimator.predict(input_fn=predict_input_fn)
				for idx, prediction in enumerate(predict_results):
						index = prediction["class_index"]  # Get the predicted class (index)
						probability = prediction["probabilities"][index]
						class_name = class_names[index]
						print("%s: Predicted Class is: %s with a probability of %f" % (datetime.now(), class_name, probability))
						break #We care for only the first prediction, rest are all duplicates just to meet the batch size



	def predict(self, stroke_data=None):
		if stroke_data is not None:
			self.FLAGS['predict_for_data'] = stroke_data
		print("%s: I Starting application" % (datetime.now()))
		print("self.FLAGS", self.FLAGS)
		estimator, train_spec, eval_spec = self.create_estimator_and_specs(
			run_config=tf.estimator.RunConfig(
				model_dir=self.FLAGS['model_dir'],
				save_checkpoints_secs=300,
				save_summary_steps=100))
		print("estimator",estimator,"train_spec",train_spec,"eval_spec",eval_spec) 
		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

		if self.FLAGS['predict_for_data'] != None:
			print("%s: I Starting prediction" % (datetime.now()))
			class_names = self.get_classes()
			self.create_tfrecord_for_prediction(self.FLAGS['batch_size'], self.FLAGS['predict_for_data'], self.FLAGS['predict_temp_file'])
			predict_results = estimator.predict(input_fn=self.get_input_fn(
				mode=tf.estimator.ModeKeys.PREDICT,
				tfrecord_pattern=self.FLAGS['predict_temp_file'],
				batch_size=self.FLAGS['batch_size']))

			#predict_results = estimator.predict(input_fn=predict_input_fn)
			for idx, prediction in enumerate(predict_results):
				index = prediction["class_index"]  # Get the predicted class (index)
				probability = prediction["probabilities"][index]
				class_name = class_names[index]
				print("%s: Predicted Class is: %s with a probability of %f" % (datetime.now(), class_name, probability))
				return class_name ,probability #We care for only the first prediction, rest are all duplicates just to meet the batch size

# pickle model
model = SketchClassifier()
pickle.dump(model, open('model.pkl', 'wb'))
# model.predict()
# model.predict("[[[186, 186, 186, 185, 183, 179, 172, 164, 159, 153, 145, 139, 136, 136, 136, 138, 141, 145, 151, 158, 167, 176, 187, 199, 208, 216, 220, 223, 224, 225, 222, 215, 205, 194, 184, 175, 168, 164, 164, 164, 164], [92, 92, 92, 92, 91, 90, 90, 91, 94, 99, 110, 120, 127, 134, 140, 145, 152, 158, 164, 166, 168, 168, 168, 167, 166, 164, 160, 156, 149, 142, 133, 121, 111, 101, 95, 90, 87, 86, 86, 86, 86]], [[398, 398, 394, 390, 388, 386, 383, 378, 373, 367, 359, 349, 340, 333, 326, 323, 320, 320, 322, 327, 335, 346, 359, 373, 388, 402, 415, 423, 426, 428, 428, 428, 424, 418, 408, 399, 391, 384, 380, 375, 369, 363, 358, 356, 356], [101, 101, 100, 99, 98, 97, 96, 96, 96, 98, 101, 104, 110, 116, 124, 133, 142, 148, 153, 158, 162, 165, 168, 170, 171, 170, 167, 164, 161, 158, 153, 147, 140, 131, 122, 114, 108, 102, 99, 96, 94, 93, 93, 94, 94]], [[303, 303, 302, 301, 300, 297, 289, 278, 263, 248, 231, 216, 201, 188, 175, 164, 152, 143, 137, 132, 127, 123, 119, 116, 115, 116, 119, 123, 128, 136, 145, 156, 169, 186, 211, 241, 272, 304, 332, 357, 375, 388, 397, 402, 404, 404, 404, 402, 399, 394, 385, 371, 356, 342, 330, 320, 310, 303, 298, 295, 293, 292, 292, 291, 291, 291, 289, 289, 289], [156, 156, 156, 156, 155, 155, 153, 150, 148, 147, 145, 146, 148, 153, 160, 168, 177, 186, 194, 207, 219, 231, 248, 265, 278, 292, 305, 317, 329, 341, 351, 359, 365, 370, 371, 370, 368, 363, 358, 350, 340, 329, 318, 308, 296, 286, 277, 267, 258, 247, 234, 218, 202, 186, 175, 166, 159, 153, 149, 147, 145, 145, 144, 144, 144, 144, 145, 145, 145]], [[182, 182, 182, 182, 181, 179, 177, 174, 169, 164, 158, 153, 149, 147, 147, 148, 151, 154, 157, 160, 164, 168, 173, 179, 184, 192, 203, 212, 217, 221, 223, 225, 226, 226, 226, 226, 224, 223, 219, 215, 209, 202, 194, 187, 181, 177, 174, 171, 169, 168, 168, 167, 167, 167, 167, 168, 169, 169], [211, 211, 210, 210, 210, 210, 210, 210, 212, 214, 216, 220, 224, 227, 231, 235, 240, 243, 246, 248, 250, 253, 255, 256, 256, 255, 253, 252, 250, 247, 244, 240, 237, 231, 226, 223, 220, 218, 216, 213, 210, 206, 203, 200, 199, 199, 200, 201, 202, 203, 203, 203, 204, 204, 204, 206, 207, 207]], [[289, 289, 287, 284, 280, 276, 272, 268, 264, 259, 255, 251, 246, 244, 244, 243, 243, 244, 246, 250, 256, 264, 273, 281, 288, 295, 301, 305, 307, 309, 311, 312, 314, 314, 314, 313, 309, 302, 294, 287, 282, 279, 276, 274, 274], [216, 216, 216, 216, 216, 216, 216, 217, 219, 222, 225, 229, 233, 236, 238, 239, 241, 242, 244, 245, 246, 246, 248, 249, 250, 251, 250, 248, 245, 242, 239, 236, 234, 231, 228, 225, 221, 217, 213, 211, 211, 211, 213, 214, 214]], [[229, 229, 229, 228, 226, 224, 223, 222, 222, 222, 222, 224, 228, 232, 239, 245, 252, 257, 260, 261, 262, 262, 261, 258, 253, 249, 244, 241, 239, 239, 239, 237, 235, 233, 230, 229, 228, 228, 227, 227, 227], [294, 294, 294, 295, 296, 298, 300, 303, 305, 308, 311, 314, 317, 319, 321, 321, 320, 318, 315, 311, 308, 305, 303, 300, 298, 297, 295, 294, 293, 293, 293, 294, 295, 297, 298, 299, 299, 299, 299, 299, 299]], [[195, 195, 194, 192, 190, 187, 186, 184, 184, 184, 185, 187, 189, 192, 194, 197, 199, 200, 200, 200, 199, 199, 198, 196, 195, 195, 195, 195, 194, 194, 193, 193], [230, 230, 230, 230, 231, 232, 235, 238, 240, 242, 243, 244, 244, 244, 244, 243, 242, 241, 239, 238, 237, 235, 233, 231, 230, 229, 229, 229, 229, 230, 230, 230]], [[281, 281, 281, 279, 278, 277, 277, 277, 277, 277, 278, 278, 279, 280, 282, 283, 285, 286, 287, 288, 288, 288, 288, 287, 286, 285, 284, 284, 284, 283, 282, 280, 279, 279, 279], [234, 234, 234, 234, 235, 236, 238, 239, 241, 241, 241, 242, 242, 242, 242, 242, 242, 241, 240, 239, 239, 238, 238, 237, 236, 234, 233, 233, 233, 233, 233, 233, 233, 233, 233]]]")