import os
import bz2
import json
import numpy as np
import fasttext
from tqdm import tqdm
import string
import nltk

from scipy.spatial.distance import cosine

from dask import dataframe as ddf
nltk.download('stopwords')
nltk.download('punkt')

PATH_TO_FILES = ['data/quotes-2015.json.bz2', 'data/quotes-2016.json.bz2',
                 'data/quotes-2017.json.bz2', 'data/quotes-2018.json.bz2',
                 'data/quotes-2019.json.bz2', 'data/quotes-2020.json.bz2']

FASTTEXT_FILE = 'data/fasttext.txt'
FASTTEXT_MODEL_FILE = 'data/fasttext.vec'


# @dask_delayed
def data_gen(paths=None):
    """
    iterator over all given paths
    :param paths: list of file paths
    :return: generated instances
    """

    if paths is None:
        paths = PATH_TO_FILES
    paths = iter(paths)

    for path in paths:
        with bz2.open(path, 'rb') as s_file:
            while True:
                try:
                    instance = json.loads(next(s_file))
                    yield instance
                except StopIteration:
                    break

def get_tokenizer(stemmer=nltk.stem.PorterStemmer(), stopwords=nltk.corpus.stopwords.words('english'),
                  return_as_list=False):
    """
    pre-process quote for training of fasttext
    :param text: string (quote)
    :param stemmer: nltk stemmer to be used
    :param stopwords: nltk stowrods to be used
    :return:
    """

    def tokenizer(text):
        text = "".join([ch for ch in text if ch not in string.punctuation])
        tokens = nltk.word_tokenize(text)
        tokens = [stemmer.stem(word.lower()) for word in tokens if word not in stopwords]
        if return_as_list:
            return tokens
        return " ".join(tokens)

    return tokenizer




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
        tokenize = get_tokenizer()

        def read_sample(d):
            return d['quotation']

        def process_sample(quote):
            return tokenize(quote) + '\n'

        def save_sample(tks):
            fastfile.write(tks)


        with open(filepath, 'w') as fastfile:

            for data in tqdm(data_generator):
                save_sample(process_sample(read_sample(data)))

    model = fasttext.train_unsupervised(filepath, model='cbow')
    model.save_model(modelpath)


def load_embeddings(model_path=FASTTEXT_MODEL_FILE, get_model=False):
    """
    Load embeddings from fasttext model
    :param model_path: path to fasttext model
    :param get_model: return the whole model
    :return: whoole model if return_model=true, o.w. only embeedings and vocab are returned as tuple
    """
    model = fasttext.load_model(model_path)
    if get_model:
        return model
    vocab = model.words
    word_embeddings = np.array([model[word] for word in vocab])
    return word_embeddings, vocab


def get_similarity_measure(keyvector, model, tokenizer=get_tokenizer(return_as_list=True)):
    """
    Given a model and keywords return a function that returns the min distance between the keywords and the
    given quote
    :param keywords: the desired keywords to match
    :param model: the trained model
    :param tokenizer: the tokenizer to be used (should be the same as used at model training time to match vocabulary)
    :return: similarity_measure(quote) -> similarity score
    """

    def similarity_measure(quote):
        if tokenizer is not None:
            tokenized = tokenizer(quote)
        else:
            tokenized = quote
        embeddings = [model.get_word_vector(word) for word in tokenized]

        similarities = [cosine(word, keyvector)
                        for word in embeddings]
        if len(similarities) > 0:
            similarity = min(similarities)
        else:
            print('No tokens for: ', quote)
            return 0
        return similarity

    return similarity_measure


