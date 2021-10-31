import torch
import bz2
import json
import pandas as pd
import numpy as np
PATH_TO_FILES = ['data/quotes-2015.json.bz2?download=1', 'data/quotes-2016.json.bz2?download=1', 'data/quotes-2017.json.bz2?download=1', 'data/quotes-2018.json.bz2?download=1']



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


def init_bert():
    model = torch.hub.load('huggingface/pytorch-transformers', 'model',
                           'bert-base-uncased')  # Download model and configuration from S3 and cache.

