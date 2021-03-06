"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

class Model:

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

'''
class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    @classmethod
    def load_training_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(data, chars, data_size, vocab_size, char_to_ix, ix_to_char):
        # hyperparameters
        hidden_size = 100 # size of hidden layer of neurons
        seq_length = 25 # number of steps to unroll the RNN for
        learning_rate = 1e-1

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
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = Model.lossFun(inputs, targets, vocab_size, Wxh, Whh, Why, bh, by, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # perform parameter update with Adagrad
            for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                                                            [dWxh, dWhh, dWhy, dbh, dby],
                                                                            [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

            p += seq_length # move data pointer
            n += 1 # iteration counter

        return hprev, Wxh, Whh, Why, bh, by

    def run_pred(self, data, hprev, vocab_size, Wxh, Whh, Why, bh, by):
        # your code here
        preds = []
        for line in data:
            line = line.split()
            char = list(line[len(line)-1])
            i = char[len(char)-1] #get last character of input line
            sample_ix = Model.sample_top3(hprev, char_to_ix[i], vocab_size, Wxh, Whh, Why, bh, by)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)

            preds.append(txt)
        return preds


    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            print(f.read())
        return MyModel()
'''