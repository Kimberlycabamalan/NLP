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

class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature=temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Create a mask to prevent "" or "[UNK]" from being generated.
    skip_ids = self.ids_from_chars(['','[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices = skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(ids_from_chars.get_vocabulary())]) 
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits] 
    predicted_logits, states =  self.model(inputs=input_ids, states=states, 
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert from token ids to characters
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Return the characters and model state.
    return predicted_chars, states

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
	vocab.remove('\n')
	vocab.remove(' ')
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

	model = MyModel(
	# Be sure the vocabulary size matches the `StringLookup` layers.
	vocab_size=len(ids_from_chars.get_vocabulary()),
	embedding_dim=embedding_dim,
	rnn_units=rnn_units)

	loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
	model.compile(optimizer='adam', loss=loss)

	# Directory where the checkpoints will be saved
	checkpoint_dir = './training_checkpoints'
	# Name of the checkpoint files
	checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

	checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		filepath=checkpoint_prefix,
		save_weights_only=True)
	
	EPOCHS = 50
	history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

	one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

	start = time.time()
	states = None
	test_data = load_test_data('example/input.txt')
	pred = []
	next_char = tf.constant(test_data)
	#next_char = tf.constant(['on'])
	result = []
	second_result = []
	third_result = []

	first_char, states = one_step_model.generate_one_step(next_char, states=states)
	second_choice, states = one_step_model.generate_one_step(next_char, states=states)
	third_choice, states = one_step_model.generate_one_step(next_char, states=states)
	print('1', first_char)
	print('2', second_choice)
	print('3', third_choice)
	print('\n')
	print(len(test_data))
	for i in range(0, len(test_data)):
		next_char_3 = first_char[i].numpy().decode('utf-8') + second_choice[i].numpy().decode('utf-8') + third_choice[i].numpy().decode('utf-8')
		pred.append(next_char_3)

	end = time.time()

	# print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
	# print(second_result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
	# print(third_result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)

	print(f"\nRun time: {end - start}")

	
	# for line in test_data:
	# 	print(line)
	# 	all_ids = ids_from_chars(tf.strings.unicode_split(line, 'UTF-8'))
	# 	ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
	# 	sequences = ids_dataset.batch(len(line))
	# 	dataset = sequences.map(split_input_target)
		
	# 	dataset = (
	# 	dataset
	# 	.batch(1)
	# 	.prefetch(tf.data.experimental.AUTOTUNE))
	# 	for input_example_batch, target_example_batch in dataset.take(1):
	# 		example_batch_predictions = model(input_example_batch)
	# 	sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
	# 	sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
	# 	next_char_pred = text_from_ids(sampled_indices).numpy()
	# 	next_char_pred_str = tf.compat.as_str_any(next_char_pred)
	# 	next_char_pred_list = list(''.join(next_char_pred_str.split('[UNK]')))
	# 	next_char_3 = next_char_pred_list[0:3]
	# 	print(next_char_3)
	# 	pred.append(''.join(next_char_3))

	##Previous code
	# model = CustomTraining(
	# vocab_size=len(ids_from_chars.get_vocabulary()),
	# embedding_dim=embedding_dim,
	# rnn_units=rnn_units)
	
	# model.compile(optimizer = tf.keras.optimizers.Adam(),loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

	# model.fit(dataset, epochs=10)

	# test_data = load_test_data('input.txt')
	# pred = []
	# for line in test_data:
	# 	print(line)
	# 	all_ids = ids_from_chars(tf.strings.unicode_split(line, 'UTF-8'))
	# 	ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
	# 	sequences = ids_dataset.batch(len(line))
	# 	dataset = sequences.map(split_input_target)
		
	# 	dataset = (
	# 	dataset
	# 	.batch(1)
	# 	.prefetch(tf.data.experimental.AUTOTUNE))
	# 	for input_example_batch, target_example_batch in dataset.take(1):
	# 		example_batch_predictions = model(input_example_batch)
	# 	sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
	# 	sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
	# 	next_char_pred = text_from_ids(sampled_indices).numpy()
	# 	next_char_pred_str = tf.compat.as_str_any(next_char_pred)
	# 	next_char_pred_list = list(''.join(next_char_pred_str.split('[UNK]')))
	# 	next_char_3 = next_char_pred_list[0:3]
	# 	print(next_char_3)
	# 	pred.append(''.join(next_char_3))

	write_pred(pred, 'output.txt')

	# #Report on average loss
	# loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
	# example_batch_loss = loss(target_example_batch, example_batch_predictions)
	# mean_loss = example_batch_loss.numpy().mean()
	# print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
	# print("Mean loss:        ", mean_loss)








