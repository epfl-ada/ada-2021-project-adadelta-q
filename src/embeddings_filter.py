import os
import torch
import bz2
import json
import pandas as pd
import numpy as np
import fasttext
from tqdm import tqdm

PATH_TO_FILES = ['data/quotes-2015.json.bz2?download=1', 'data/quotes-2016.json.bz2?download=1', 'data/quotes-2017.json.bz2?download=1', 'data/quotes-2018.json.bz2?download=1']

FASTTEXT_FILE= 'data/fasttext.txt'
FASTTEXT_MODEL_FILE = 'data/fasttext.vec'

def data_gen(paths=None):
    """
    iterator over all given paths
    :param paths: list of file paths
    :return: generated instances
    """
    if paths is None:
        paths = PATH_TO_FILES
    paths = iter(paths)
    while next(paths):
        with bz2.open(PATH_TO_FILES[0], 'rb') as s_file:
            for instance in s_file:
                instance = json.loads(instance)
                yield instance


def setup_and_train_fasttext_data(data_generator, filepath=FASTTEXT_FILE, modelpath=FASTTEXT_MODEL_FILE):
    """
    Train cbow fastttext model
    :param data_generator: iterator over data
    :param filepath: filepath of fasttext data
    :param modelpath:
    :return:
    """
    if not os.path.exists(filepath):
        print("Writing Fasttext File")

        with open(filepath, 'w') as fastfile:
            for d in tqdm(data_generator):
                fastfile.write(d['quotation'])

    model = fasttext.train_unsupervised(filepath, model='cbow')
    model.save_model(modelpath)


def load_embeddings(model_path=FASTTEXT_MODEL_FILE):
    model = fasttext.load_model(model_path)
    vocab = model.words
    word_embeddings = np.array([model[word] for word in vocab])
    return word_embeddings, vocab

def filter_data(fasttext_model):
    # Filter data given a model and the original data, saves as a new single file
    # returns: filepath to new file
    # TODO: Find criteria to filter: top percentiles? Similarities according to data?
    raise NotImplementedError