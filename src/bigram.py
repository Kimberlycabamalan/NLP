#!/usr/bin/env python
# coding: utf8

import os
import string
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import math
import operator

def load_training():
    data = pd.read_csv('data/xnli.dev.tsv', sep='\t', header=0)
    sentences = set(list(data["sentence1"]))
    sentences2 = set(list(data["sentence2"]))
    sentences |= sentences2

    with open("data/training.txt", 'wt') as f:
        for sentence in sentences:
            f.write('{}\n'.format(sentence))

def create_test():
    data = pd.read_csv('data/xnli.test.tsv', sep='\t', header=0)
    sentences = set(list(data["sentence1"]))

    with open("data/test.txt", 'wt') as f:
        for sentence in sentences:
            f.write('{}\n'.format(sentence[:-2]))

    with open("data/answer.txt", 'wt') as f:
        for sentence in sentences:
            f.write('{}\n'.format(sentence[-2]))

def get_tokens(sentences):
    result = {}
    result["<STOP>"] = 0
    token_count = 0
    for line in sentences:
        tokens = list(line.replace("\n", "").replace('\xa0', ' ')) # split by character

        for token in tokens:
            if token not in result:
                result[token] = 1
            else:
                result[token] += 1

        token_count += len(tokens) + 1
        result["<STOP>"] += 1

    # replace word with UNK if frequency is less than 3
    keys = result.keys()
    unk_tokens = []
    result["UNK"] = 0
    for key in keys:
        if result[key] < 3:
            result["UNK"] += result[key]
            unk_tokens.append(key)

    for unk_token in unk_tokens:
        del result[unk_token]

    return result, token_count

def get_unigram_probabilities(tokens, token_count):
    result = {}
    keys = tokens.keys()
    for key in keys:
        result[key] = tokens[key] / token_count
    return result

def get_unigram_perplexity(unigram_probabilities, file):
    total = 0
    token_count = 0
    new_lines = 0
    with open(file, 'r') as f:
        for line in f:
            tokens = list(line.replace("\n", ""))
            for token in tokens:
                if token in unigram_probabilities:
                    total += math.log(unigram_probabilities[token], 2)
                else:
                    total += math.log(unigram_probabilities["UNK"], 2)
            total += math.log(unigram_probabilities["<STOP>"], 2)
            token_count += len(tokens) + 1

    cross_entropy = -total / token_count
    result = math.pow(2, cross_entropy)
    return result

def get_bigram_counts(training_tokens, file):
    result = {}
    with open(file, 'r') as f:
        for line in f:
            first = "<START>"
            tokens = list(line.replace("\n", ""))
            tokens.append("<STOP>")
            for token in tokens:
                if token not in training_tokens:
                    second = "UNK"
                else:
                    second = token

                if second not in result:
                    result[second] = {}

                bigrams = result[second]
                if first not in bigrams:
                    bigrams[first] = 1

                else:
                    bigrams[first] += 1
                first = second

    return result

def get_bigram_probabilities(bigrams, training_tokens):
    result = bigrams
    keys = bigrams.keys()
    for key in keys:
        first_tokens = bigrams[key]
        for first in first_tokens:
            if first == "<START>":
                count1 = training_tokens["<STOP>"]
            else:
                count1 = training_tokens[first]

            p = first_tokens[first] / count1
            result[key][first] = p

    return result

def get_bigram_perplexity(bigram_probability, training_tokens, file):
    total = 0
    token_count = 0
    new_lines = 0
    with open(file, 'r') as f:
        for line in f:
            sentence_probability = 0
            first = "<START>"
            tokens = list(line.replace("\n", ""))
            tokens.append("<STOP>")
            for token in tokens:
                if token not in training_tokens:
                    second = "UNK"
                else:
                    second = token

                if second in bigram_probability:
                    first_tokens = bigram_probability[second]

                    # if word1 and word2 exist in V, p = count(bigram) / count(word1)
                    if first in first_tokens:
                        p = first_tokens[first]
                        sentence_probability += math.log(p, 2)

                    # bigram does not exist in training, p = 0
                    else:
                        sentence_probability -= math.inf

                # bigram does not exist in training, p = 0
                else:
                    sentence_probability -= math.inf
                first = second

            token_count += len(tokens)
            total += sentence_probability

    result = math.pow(2, (-total / token_count))
    return result

def get_trigram_counts(training_tokens, file):
    result = {}
    token_count = 0
    with open(file, 'r') as f:
        for line in f:
            first = "<START>"
            second = "<START>"

            tokens = list(line.replace("\n", ""))
            tokens.append("<STOP>")
            for token in tokens:
                if token not in training_tokens:
                    third = "UNK"
                else:
                    third = token

                if third not in result:
                    result[third] = {}
                trigrams = result[third]
                if second not in trigrams:
                    result[third][second] = {}
                bigrams = result[third][second]
                if first not in bigrams:
                    result[third][second][first] = 1
                else:
                    result[third][second][first] += 1

                first = second
                second = third
    return result

def get_trigram_probabilities(trigrams, bigrams, training_tokens):
    result = trigrams
    keys = trigrams.keys()

    for key in keys:
        second_tokens = trigrams[key]
        for second in second_tokens:
            first_tokens = second_tokens[second]
            for first in first_tokens:
                trigram_count = first_tokens[first]
                if first == "<START>" and second == "<START>":
                    bigram_count = training_tokens["<STOP>"]
                else:
                    bigram_count = bigrams[second][first]

                result[key][second][first] = trigram_count / bigram_count
    return result

def get_trigram_perplexity(trigram_probabilities, training_tokens, file):
    total = 0
    token_count = 0

    with open(file, 'r') as f:
        for line in list(f):
            sentence_probability = 0
            first = "<START>"
            second = "<START>"
            tokens = list(line.replace("\n", ""))
            tokens.append("<STOP>")

            for token in tokens:
                if token not in training_tokens:
                    third = "UNK"
                else:
                    third = token

                # if word1, word2 and word3 exist in V, p = count(trigram) / count(bigram)
                if third in trigram_probabilities:
                    second_tokens = trigram_probabilities[third]
                    if second in second_tokens:
                        first_tokens = second_tokens[second]
                        if first in first_tokens:
                            p = first_tokens[first]
                            sentence_probability += math.log(p, 2)
                        else:
                            sentence_probability -= math.inf

                    # trigram does not exist in training, p = 0
                    else:
                        sentence_probability -= math.inf

                # trigram does not exist in training, p = 0
                else:
                    sentence_probability -= math.inf
                first = second
                second = third
            token_count += len(tokens)
            total += sentence_probability

    result = math.pow(2, (-total / token_count))
    return result

def linear_interpolation(file, unigram, bigram, trigram, w1, w2, w3):
    total = 0
    token_count = 0
    with open(file, 'r') as f:
        for line in f:
            first = "<START>"
            second = "<START>"
            tokens = list(line.replace("\n", ""))
            tokens.append("<STOP>")

            for token in tokens:
                if token not in unigram:
                    third = "UNK"
                else:
                    third = token

                if third in trigram:
                    p = w1 * unigram[third]
                    second_tokens = trigram[third]

                    if second in second_tokens:
                        p += w2 * bigram[third][second]
                        first_tokens = second_tokens[second]

                        if first in first_tokens:
                            p += w3 * trigram[third][second][first]

                    total += math.log(p, 2)

                first = second
                second = third

            token_count += len(tokens)

    result = math.pow(2, (-total / token_count))
    return result

# abc d
def predict(bigram_probabilities, unigram_probabilities, file, test = False):
    with open(file, 'r') as f:
        guesses = []
        for line in f:
            tokens = list(line.replace("\n", ""))

            if test:
                if len(tokens) < 2:
                    guesses.append([])
                else:
                    prev = tokens[-2]
                    top_3 =  get_top_3(bigram_probabilities, prev)
                    guesses.append(top_3)
            else:
                if len(tokens) < 1:
                    guesses.append([])
                else:
                    prev = tokens[-1]
                    top_3 =  get_top_3(bigram_probabilities, unigram_probabilities, prev)
                    guesses.append(top_3)
    return guesses

def get_top_3(bigram_probabilities, unigram_probabilities, prev):
    # maps guesses to their probabilities
    guesses = {}

    keys = list(bigram_probabilities.keys())
    keys.remove("<STOP>")
    keys.remove("UNK")
    #keys.remove(" ")
    for key in keys:
        prev_tokens = bigram_probabilities[key]
        if prev in prev_tokens:
            guesses[key] = prev_tokens[prev]


    sorted_guesses = dict( sorted(guesses.items(), key=operator.itemgetter(1),reverse=True))
    keys = list(sorted_guesses.keys())

    # get top 3 guesses with highest freq
    num_guesses = len(keys)
    if num_guesses < 3:
        sorted_guesses = dict( sorted(unigram_probabilities.items(), key=operator.itemgetter(1),reverse=True))
        unigrams = list(sorted_guesses.keys())
        unigrams.remove("<STOP>")
        unigrams.remove("UNK")
        # unigrams.remove(" ")

    while num_guesses < 3:
        keys = set(keys)
        next_guess = unigrams.pop(0)
        keys.add(next_guess)
        num_guesses = len(keys)

    return list(keys)[:3]


def write_pred(preds, fname):
    with open(fname, 'wt') as f:
        for top_3 in preds:
            p = ''.join(top_3)
            p = p.replace("\n", "")
            p = p.replace("<s>", " ")
            f.write('{}\n'.format(p))

def load(work_dir):
    with open(os.path.join(work_dir, 'model.checkpoint.unigrams'), encoding='utf-8') as f:
        unigrams = {}
        # f = list(f)[:-1]
        for line in f:
                line = line.split()
                unigrams[line[0]] = float(line[1])

    with open(os.path.join(work_dir, 'model.checkpoint.bigrams'), encoding='utf-8') as f:
        bigrams = {}
        # f = list(f)[:-1]
        for line in f:
            line = line.split()
            second = line[0]
            first = line[1]
            p = float(line[2])
            if second not in bigrams:
                bigrams[second] = {}
            bigrams[second][first] = p

    return unigrams, bigrams

def save(bigram_probabilities, unigram_probabilities, work_dir):
    with open(os.path.join(work_dir, 'model.checkpoint.unigrams'), 'wt', encoding='utf-8') as f:
        unigrams = list(unigram_probabilities)
        for unigram in unigrams:
            if unigram == " ":
                f.write('{}\n'.format("<s> " + str(unigram_probabilities[unigram])))
            else:
                f.write('{}\n'.format(unigram + " " + str(unigram_probabilities[unigram])))

    with open(os.path.join(work_dir, 'model.checkpoint.bigrams'), 'wt', encoding='utf-8') as f:
        bigrams = list(bigram_probabilities)
        for bigram in bigrams:
            first_tokens = list(bigram_probabilities[bigram])
            for token in first_tokens:
                if bigram == " " and token == " ":
                    f.write('{}\n'.format("<s> <s> " + str(bigram_probabilities[bigram][token])))
                elif bigram == " ":
                    f.write('{}\n'.format("<s> " + token + " " + str(bigram_probabilities[bigram][token])))
                elif token == " ":
                    f.write('{}\n'.format(bigram + " <s> " + str(bigram_probabilities[bigram][token])))
                else:
                    f.write('{}\n'.format(bigram + " " + token + " " + str(bigram_probabilities[bigram][token])))


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test', 'load_data'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--train_data', help='path to train data', default='data/training.txt')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        print('Instatiating model')
        with open(args.train_data, 'r', encoding='utf-8') as f:
            sentences = list(f)

        tokens, token_count = get_tokens(sentences)
        unigram_probabilities = get_unigram_probabilities(tokens, token_count)
        bigram_counts = get_bigram_counts(tokens, args.train_data)
        bigram_probabilities = get_bigram_probabilities(bigram_counts, tokens)

        #bpreds = predict(bigram_probabilities, unigram_probabilities, args.test_data)
        # write_pred(preds, args.test_output)
        print('Saving model')
        save(bigram_probabilities, unigram_probabilities, args.work_dir)

    elif args.mode == "load_data":
        load_training()
        create_test()

    elif args.mode == 'test':
        print('Loading model')
        unigrams, bigrams = load(args.work_dir)

        print('Loading test data from {}'.format(args.test_data))
        print('Making predictions')
        preds = predict(bigrams, unigrams, args.test_data)

        print('Writing predictions to {}'.format(args.test_output))
        with open(args.test_data, 'r') as f:
            size = len(list(f))
        assert len(preds) == size, 'Expected {} predictions but got {}'.format(size, len(preds))
        write_pred(preds, args.test_output)

    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))


