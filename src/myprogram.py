#!/usr/bin/env python
# coding: utf8
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import random
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

def save(vocab, chars_from_ids, work_dir):
    with open(os.path.join(work_dir, 'model.checkpoint.vocab'), 'wt') as f:
        for val in vocab:
            f.write(val + "\n")
        f.write(len(vocab))

def load(work_dir):
    data = []
    with open(os.path.join(work_dir, 'model.checkpoint.vocab')) as f:
        f = list(line)
        for line in f[:-1]:
            line = line.split()  # the last character is a newline
            data.append(line[0])
        vocab_size = f[-1]
    return data, vocab_size

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--train_data', help='path to train data', default='data/training.txt')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        start = time.time()
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)

        path_to_file = args.train_data

        # Read, then decode for py2 compat.
        text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
        # length of text is the number of characters in it
        print('Length of text: {} characters'.format(len(text)))

        # The unique characters in the file
        vocab = sorted(set(text))
        vocab.remove("\n")
        vocab.remove(" ")
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
                .batch(BATCH_SIZE, drop_remainder=False)
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

        EPOCHS = 1
        history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

        save(vocab, chars_from_ids, args.work_dir)
        '''
        vocab:
        vocab
        vocab size
        
        dataset
        epochs
        chars_from_id
        ids_from_chars
        '''
        end = time.time()
        print(f"\nRun time: {end - start}")
    elif args.mode == 'test':
        start = time.time()
        # Create a basic model instance
        vocab, vocab_size = load(args.work_dir)

        ids_from_chars = preprocessing.StringLookup(vocabulary=list(vocab))
        chars_from_ids = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=ids_from_chars.get_vocabulary(), invert=True)

        # The embedding dimension
        embedding_dim = 256

        # Number of RNN units
        rnn_units = 1024

        model = MyModel(
            # Be sure the vocabulary size matches the `StringLookup` layers.
            vocab_size=len(ids_from_chars.get_vocabulary()),
            embedding_dim=embedding_dim,
            rnn_units=rnn_units)

        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        model.compile(optimizer='adam', loss=loss)

        checkpoint_path = "training_checkpoints"
        model.load_weights(checkpoint_path)
        one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

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
        write_pred(pred, 'output.txt')
        end = time.time()
        print(f"\nRun time: {end - start}")
        '''
        print('Loading model')
        model = MyModel()
        hprev, char_to_ix, ix_to_char, vocab_size, Wxh, Whh, Why, bh, by = MyModel.load(args.work_dir)

        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)

        print('Making predictions')
        pred = model.run_pred(test_data, hprev, char_to_ix, ix_to_char, vocab_size, Wxh, Whh, Why, bh, by)

        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        print(pred)
        model.write_pred(pred, args.test_output)
        '''


    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))


