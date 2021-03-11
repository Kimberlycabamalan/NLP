import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True, 
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else: 
      return x

class CustomTraining(MyModel):
	@tf.function
	def train_step(self, inputs):
		inputs, labels = inputs
		with tf.GradientTape() as tape:
			predictions = self(inputs, training=True)
			loss = self.loss(labels, predictions)
		grads = tape.gradient(loss, model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
		return {'loss': loss}

def text_from_ids(ids):
  return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

def write_pred(preds, fname):
	with open(fname, 'wt') as f:
		for p in preds:
			p = p.replace("\n", "")
			print(p)
			f.write('{}\n'.format(p))

def load_test_data(fname):
	data = []
	with open(fname) as f:
		for line in f:
			inp = line[:-1]  # the last character is a newline
			data.append(inp)
	return data

if __name__ == '__main__':
	# parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
 #    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
 #    parser.add_argument('--work_dir', help='where to save', default='work')
 #    # parser.add_argument('--train_data', help='path to train data', default='example/input.txt')
 #    parser.add_argument('--train_data', help='path to train data', default='wikitext-103/wiki.train.tokens')
 #    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
 #    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
	# args = parser.parse_args()
	path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

	# Read, then decode for py2 compat.
	text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
	# length of text is the number of characters in it
	print('Length of text: {} characters'.format(len(text)))

	# The unique characters in the file
	vocab = sorted(set(text))
	print('{} unique characters'.format(len(vocab)))

	# Length of the vocabulary in chars
	vocab_size = len(vocab)

	# The embedding dimension
	embedding_dim = 256

	# Number of RNN units
	rnn_units = 1024

	example_texts = ['abcdefg', 'xyz']
	chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')

	ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))

	ids = ids_from_chars(chars)

	chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
	vocabulary=ids_from_chars.get_vocabulary(), invert=True)
	chars = chars_from_ids(ids)
	tf.strings.reduce_join(chars, axis=-1).numpy()

	all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
	ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
	# for ids in ids_dataset.take(10):
	# 	print(chars_from_ids(ids).numpy().decode('utf-8'))

	seq_length = 100
	examples_per_epoch = len(text)//(seq_length+1)

	sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

	dataset = sequences.map(split_input_target)

	# Batch size
	BATCH_SIZE = 64

	# Buffer size to shuffle the dataset
	# (TF data is designed to work with possibly infinite sequences,
	# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
	# it maintains a buffer in which it shuffles elements).
	BUFFER_SIZE = 10000

	dataset = (
		dataset
		.shuffle(BUFFER_SIZE)
		.batch(BATCH_SIZE, drop_remainder=True)
		.prefetch(tf.data.experimental.AUTOTUNE))

	# model = MyModel(
	# # Be sure the vocabulary size matches the `StringLookup` layers.
	# vocab_size=len(ids_from_chars.get_vocabulary()),
	# embedding_dim=embedding_dim,
	# rnn_units=rnn_units)
	# model.compile(optimizer='adam', loss=loss)

	# Directory where the checkpoints will be saved
	checkpoint_dir = './training_checkpoints'
	# Name of the checkpoint files
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_prefix,
		save_weights_only=True)

	model = CustomTraining(
	vocab_size=len(ids_from_chars.get_vocabulary()),
	embedding_dim=embedding_dim,
	rnn_units=rnn_units)
	
	model.compile(optimizer = tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

	model.fit(dataset, epochs=10)

	#predictions
	# for input_example_batch, target_example_batch in dataset.take(1):
	# 	example_batch_predictions = model(input_example_batch)
	# 	print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
	# sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
	# sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

	# print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
	# print()
	# next_char_pred = text_from_ids(sampled_indices).numpy()

	# next_char_pred_str = tf.compat.as_str_any(next_char_pred)
	# next_char_pred_list = list(''.join(next_char_pred_str.split('[UNK]')))
	# next_char_3 = next_char_pred_list[0:3]

	test_data = load_test_data('input.txt')
	pred = []
	for line in test_data:
		print(line)
		all_ids = ids_from_chars(tf.strings.unicode_split(line, 'UTF-8'))
		ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
		sequences = ids_dataset.batch(len(line))
		dataset = sequences.map(split_input_target)
		
		dataset = (
		dataset
		.batch(1)
		.prefetch(tf.data.experimental.AUTOTUNE))
		for input_example_batch, target_example_batch in dataset.take(1):
			example_batch_predictions = model(input_example_batch)
		sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
		sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
		next_char_pred = text_from_ids(sampled_indices).numpy()
		next_char_pred_str = tf.compat.as_str_any(next_char_pred)
		next_char_pred_list = list(''.join(next_char_pred_str.split('[UNK]')))
		next_char_3 = next_char_pred_list[0:3]
		print(next_char_3)
		pred.append(''.join(next_char_3))



	write_pred(pred, 'output.txt')

	# #Report on average loss
	# loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
	# example_batch_loss = loss(target_example_batch, example_batch_predictions)
	# mean_loss = example_batch_loss.numpy().mean()
	# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
	# print("Mean loss:        ", mean_loss)








