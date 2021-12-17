import os.path

import numpy as np

import src.embeddings_filter as ef
import bz2
from tqdm import tqdm
import json
import os
import pyarrow as pa
from pyarrow import parquet as pq
import pandas as pd

OUT_PATH = 'data/final_filtered.json.bz2'  # location of the final .json output file

PREPROCESSED_PATH = 'data/preprocessed.json.bz2'  # location to store preprocessed data (with tokens + basic filtered)
PROBABILITY_THRESHOLD = 0.3  # Threshold to filter out 'too uncertain' speaker-quote match)
FASTTEXT_FILEPATH = './data/fasttextfile.txt'  # filepath to the fasttext input file (tokenized, line-by-line)
FASTTEXT_MODELPATH = './data/data.vec'  # where fasttext model should be saved/loaded
KEYWORDS = ['stockmarket', 'stock', 'bonds', 'shares', 'obligations', 'finance']  # Keywords to compare against
# more keywords to try: finance,
COSINE_THRESHOLD = 0.3  # similarity threshold
COSINE_FILE = 'data/cosine.json.bz2'
# Number of lines to process at one time (reduce if memory issues arise)
N_LINES = 100000

def basic_preprocess(tokenize=ef.get_tokenizer(), dataloader=ef.data_gen(), processed_filepath=PREPROCESSED_PATH,
                     fasttext_filepath=FASTTEXT_FILEPATH):
    """
    implements basic preprocessing: remove unused rows and values, prepare fasttext input
    :param tokenize:
    :param dataloader:
    :param processed_filepath:
    :param fasttext_filepath:
    :return: none, saves files directly to disk
    """
    gen = dataloader


    with open(fasttext_filepath, 'w') as fastfile:
        df = pd.DataFrame([x for _, x in zip(range(N_LINES), gen)])
        print(df.columns)

        def process(df):
            df = df[df['speaker'] == 'None']
            df = df[df['probas'].apply(lambda x: float(x[0][1]) < PROBABILITY_THRESHOLD)]
            print(df['quotation'])
            df['tokenized'] = df['quotation'].apply(lambda quote: tokenize(quote) + '\n')
            df.drop(columns=['probas', 'phase'], inplace=True)
            for token in df['tokenized']:
                fastfile.write(token)
            return pa.Table.from_pandas(df)

        table = process(df)
        print(table.schema)
        print(df)
        with pq.ParquetWriter(os.path.join('data', 'full_set.parquet.gzip'), table.schema) as writer:

            while len(df) > 0:
                table = process((df))
                print(len(table))
                writer.write_table(table)
                df = pd.DataFrame([x for _, x in zip(range(N_LINES), gen)])
            writer.write_table(process((df)))

    print('Finished initial Preprocessing')



def preprocess_and_filter():

    if not (os.path.exists(PREPROCESSED_PATH) and os.path.exists(FASTTEXT_FILEPATH)):
        basic_preprocess(ef.get_tokenizer(), ef.data_gen())

    # fasttext training
    if not os.path.exists(FASTTEXT_MODELPATH):
        ef.setup_and_train_fasttext_data(ef.data_gen(), filepath=FASTTEXT_FILEPATH, modelpath=FASTTEXT_MODELPATH)

    if not os.path.exists(COSINE_FILE):
        model = ef.load_embeddings(FASTTEXT_MODELPATH, get_model=True)
        # keyvector = sum([model.get_word_vector(keyword) for keyword in KEYWORDS]) / len(KEYWORDS)
        keyvector = np.median(np.asarray([model.get_word_vector(keyword) for keyword in KEYWORDS]), axis=0)
        similarity = ef.get_similarity_measure(keyvector, model, tokenizer=None)
        print("Computing cosine similarities")

        similarities = []
        with bz2.open(COSINE_FILE, 'wb') as d_file:
            for data in tqdm(ef.data_gen([PREPROCESSED_PATH])):
                data['cosine_similarity'] = similarity(data['tokenized'].split(' '))
                similarities.append(data['cosine_similarity'])

                d_file.write((json.dumps(data) + '\n').encode('utf-8'))

        similarities.sort()
        print("avg similarity: ", sum(similarities) / len(similarities))
        print("median similarity: ", similarities[int(len(similarities) / 2)])
        np.save(os.path.join('data', 'similarities_as_list.npy'), np.asarray(similarities))

    if not os.path.exists(OUT_PATH):
        final_count = 0
        with bz2.open(OUT_PATH, 'wb') as d_file:
            for data in tqdm(ef.data_gen([COSINE_FILE])):
                if data['cosine_similarity'] < COSINE_THRESHOLD:
                    final_count += 1
                    d_file.write((json.dumps(data) + '\n').encode('utf-8'))
        print("Finished Preprocessing, final number of quotes:", final_count)

