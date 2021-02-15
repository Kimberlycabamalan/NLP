#!/usr/bin/env python
import os
import string
import random
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def load_training_data(train_data):
        data = open(args.train_data, 'r').read()
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        char_to_ix = { ch:i for i,ch in enumerate(chars) }
        ix_to_char = { i:ch for i,ch in enumerate(chars) }

        return data, chars, data_size, vocab_size, char_to_ix, ix_to_char

    def load_test_data(fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    def run_train(data, chars, data_size, vocab_size, char_to_ix, ix_to_char, Wxh, Whh, Why, bh, by):
        n, p = 0, 0
        mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
        smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
        while n<=1000: # 100000
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            if p+seq_length+1 >= len(data) or n == 0:
                hprev = np.zeros((hidden_size,1)) # reset RNN memory
                p = 0 # go from start of data
            inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
            targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

            # forward seq_length characters through the net and fetch gradient
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = MyModel.lossFun(inputs, targets, vocab_size, Wxh, Whh, Why, bh, by, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby], [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

            p += seq_length # move data pointer
            n += 1 # iteration counter

        return hprev

    def lossFun(inputs, targets, vocab_size, Wxh, Whh, Why, bh, by, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)
        loss = 0
        # forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
            ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        dbh, dby = np.zeros_like(bh), np.zeros_like(by)
        dhnext = np.zeros_like(hs[0])
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(Whh.T, dhraw)
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def sample(h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):
            h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
            y = np.dot(Why, h) + by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(vocab_size), p=p.ravel())
            x = np.zeros((vocab_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

    def sample_top3(h, seed_ix, vocab_size, Wxh, Whh, Why, bh, by):
        """
        output the top3 probable next letters
        h is memory state, seed_ix is seed letter
        """
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        y = np.dot(Why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))
        ixes = np.random.choice(range(vocab_size),3 , p=p.ravel())
        return ixes

    def save(work_dir, hprev, char_to_ix, ix_to_char, vocab_size, Wxh, Whh, Why, bh, by):

        with open(os.path.join(work_dir, 'model.checkpoint.hprev'), 'wt') as f:
            for val in hprev:
                f.write(str(val[0]) + "\n")

        with open(os.path.join(work_dir, 'model.checkpoint.vocab'), 'wt') as f:
            for character in char_to_ix:
                if character == "\n":
                    f.write("<NEWLINE>\n")
                else:
                    f.write(character + "\n")
            f.write(str(vocab_size) + "\n")

        with open(os.path.join(work_dir, 'model.checkpoint.Wxh'), 'wt') as f:
            for line in Wxh:
                for val in line:
                    f.write(str(val) + "#")
                f.write("\n")

        with open(os.path.join(work_dir, 'model.checkpoint.Whh'), 'wt') as f:
            for line in Whh:
                for val in line:
                    f.write(str(val) + "#")
                f.write("\n")
        with open(os.path.join(work_dir, 'model.checkpoint.Why'), 'wt') as f:
            for line in Why:
                for val in line:
                    f.write(str(val) + "#")
                f.write("\n")
        with open(os.path.join(work_dir, 'model.checkpoint.bh'), 'wt') as f:
            for val in bh:
                f.write(str(val[0]) + "\n")
        with open(os.path.join(work_dir, 'model.checkpoint.by'), 'wt') as f:
            for val in by:
                f.write(str(val[0]) + "\n")

    def write_pred(preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))


    def run_pred(data, hprev, char_to_ix, ix_to_char, vocab_size, Wxh, Whh, Why, bh, by):
        # your code here
        preds = []
        for line in data:
            line = line.split()
            char = list(line[len(line)-1])
            i = char[len(char)-1] #get last character of input line
            sample_ix = MyModel.sample_top3(hprev, char_to_ix[i], vocab_size, Wxh, Whh, Why, bh, by)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)

            preds.append(txt)
        return preds


    def load(work_dir):
        with open(os.path.join(work_dir, 'model.checkpoint.hprev')) as f:
            print("loading hprev")
            values = []
            for line in f:
                values.append([line.split("\n")[0]])
            hprev = np.array(values).astype(np.float)

        with open(os.path.join(work_dir, 'model.checkpoint.vocab')) as f:
            print("loading char_to_ix")
            char_to_ix = {}
            ix_to_char = {}
            i = 0
            f = list(f)
            for line in f[:-1]:
                if line == "<NEWLINE>\n":
                    character = "\n"
                else:
                    character = line.split("\n")[0]
                char_to_ix[character] = i
                ix_to_char[i] = character
                i += 1

            vocab_size = int(f[-1].split("\n")[0])

            print(vocab_size)

        with open(os.path.join(work_dir, 'model.checkpoint.Wxh')) as f:
            print("loading Wxh")
            values = []
            for line in f:
                line = line.split("#")[:-1]
                values.append(line)
            Wxh = np.array(values).astype(np.float)

        with open(os.path.join(work_dir, 'model.checkpoint.Whh')) as f:
            print("loading Whh")
            values = []
            for line in f:
                line = line.split("#")[:-1]
                values.append(line)
            Whh = np.array(values).astype(np.float)

        with open(os.path.join(work_dir, 'model.checkpoint.Why')) as f:
            print("loading Why")
            values = []
            for line in f:
                line = line.split("#")[:-1]
                values.append(line)
            Why = np.array(values).astype(np.float)

        with open(os.path.join(work_dir, 'model.checkpoint.bh')) as f:
            print("loading bh")
            values = []
            for line in f:
                values.append([line.split("\n")[0]])
            bh = np.array(values).astype(np.float)

        with open(os.path.join(work_dir, 'model.checkpoint.by')) as f:
            print("loading by")
            values = []
            for line in f:
                values.append([line.split("\n")[0]])
            by = np.array(values).astype(np.float)

        return hprev, char_to_ix, ix_to_char, vocab_size, Wxh, Whh, Why, bh, by

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--train_data', help='path to train data', default='example/input.txt')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)

        print('Instatiating model')
        model = MyModel()

        print('Loading training data')
        data, chars, data_size, vocab_size, char_to_ix, ix_to_char = MyModel.load_training_data(args.train_data)

        # hyperparameters
        hidden_size = 128 # size of hidden layer of neurons
        seq_length = 32 # number of steps to unroll the RNN for
        learning_rate = 1e-1

        # model parameters
        Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
        Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
        Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
        bh = np.zeros((hidden_size, 1)) # hidden bias
        by = np.zeros((vocab_size, 1)) # output bias

        print('Training')
        hprev = MyModel.run_train(data, chars, data_size, vocab_size, char_to_ix, ix_to_char, Wxh, Whh, Why, bh, by)

        print('Saving model')
        MyModel.save(args.work_dir, hprev, char_to_ix, ix_to_char, vocab_size, Wxh, Whh, Why, bh, by)

    elif args.mode == 'test':
        print('Loading model')
        hprev, char_to_ix, ix_to_char, vocab_size, Wxh, Whh, Why, bh, by = MyModel.load(args.work_dir)

        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)

        print('Making predictions')
        pred = MyModel.run_pred(test_data, hprev, char_to_ix, ix_to_char, vocab_size, Wxh, Whh, Why, bh, by)

        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))

        print(pred)
        MyModel.write_pred(pred, args.test_output)


    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))


