import os.path

import src.embeddings_filter as ef
import bz2
from tqdm import tqdm
import json
import os
import dask.dataframe as dd

from dask.diagnostics import ProgressBar

OUT_PATH = 'data/final_filtered.json.bz2'  # loaction of the final .json output file

PREPROCESSED_PATH = 'data/preprocessed.json.bz2'  # location to store preprocessed data (with tokens + filtered)
PROBABILITY_THRESHOLD = 0.3  # Threshould to filter out 'too uncertain' speaker-quote match)
FASTTEXT_FILEPATH = './data/fasttextfile.txt'  # filepath to the fasttext input file (tokenized, line-by-line)
FASTTEXT_MODELPATH = './data/data.vec'  # where fasttext model should be saved/loaded
KEYWORDS = ['market', 'stock', 'bonds', 'shares', 'obligations']  # Keywords to compare against
COSINE_THRESHOLD = 0.3  # similarity threshold


def basic_preprocess(tokenize=ef.get_tokenizer(), dataloader=ef.data_gen(), processed_filepath=OUT_PATH,
                     fasttext_filepath=FASTTEXT_FILEPATH):
    number_of_quotes = 0
    dropped_none = 0
    dropped_prob = 0
    dropped_no_token = 0

    with open(fasttext_filepath, 'w') as fastfile:
        with bz2.open(processed_filepath, 'wb') as d_file:
            for data in tqdm(dataloader):
                number_of_quotes += 1

                if data['speaker'] == 'None':
                    dropped_none += 1
                    continue
                if float(data['probas'][0][1]) < PROBABILITY_THRESHOLD:
                    dropped_prob += 1
                    continue

                token = tokenize(data['quotation']) + '\n'
                if len(token) == 0:
                    dropped_no_token += 1
                    continue
                del data['probas']
                del data['phase']
                fastfile.write(token)
                data['tokenized'] = token
                d_file.write((json.dumps(data) + '\n').encode('utf-8'))
    print('Finished initial Preprocessing')
    print("Total processed Quotees: ", number_of_quotes)
    print('Of which no speaker: ', dropped_none)
    print('Of which uncertain: ', dropped_prob)
    print('Of which not tokenizable: ', dropped_no_token)

    print("Total Quotes written: ", number_of_quotes - dropped_no_token - dropped_prob - dropped_none)


def dask_basic_preprocess():
    """
    Feasibilyt study, do not use currently
    :return: preprocessed data
    """
    assert False, 'Dask preprocessor not working yet, use lower perfromance basic_preporcess'
    tokenize = ef.get_tokenizer()

    def reader(paths):
        for path in paths:
            with bz2.open(path, 'rb') as s_file:
                def get_next():

                    while True:
                        try:
                            return json.loads(next(s_file))
                        except StopIteration:
                            next(paths)
        return get_next

    def strip_vals(data):
        data = data.copy()
        if data['speaker'] == 'None':
            return None
        if float(data['probas'][0][1]) < PROBABILITY_THRESHOLD:
            return None
        token = tokenize(data['quotation']) + '\n'
        if len(token) == 0:
            return None
        del data['probas']
        del data['phase']
        data['tokenized'] = token
        return data

    def save(instance):
        if instance is not None:
            json_line = (json.dumps(instance) + '\n').encode('utf-8')
            with open(FASTTEXT_FILEPATH, 'w') as fastfile:
                with bz2.open(OUT_PATH, 'wb') as d_file:
                    fastfile.write(instance['tokenized'])
                    d_file.write(json_line)

    def run_preprocess(instance):
        instance = strip_vals(instance)
        save(instance)

    print('Running Dask')
    df = dd.read_json(
        ['data/quotes-2015.json', 'data/quotes-2016.json', 'data/quotes-2017.json', 'data/quotes-2018.json',
         'data/quotes-2019.json', 'data/quotes-2020.json'], blocksize=2 ** 18)
    with ProgressBar():
        # df.apply(run_preprocess, axis=1, meta=None).compute()
        # b.map(run_preprocess).compute()
        # print(df[df['speaker']=='None'].count().compute())
        df.drop(df['speaker'] == 'None', axis=1)
        df.drop(float(df['probas'][0][1] < PROBABILITY_THRESHOLD), axis=1)
        df.drop(columns=['probas', 'phase'])
        df['token'] = tokenize(df['quotation']) + '\n'
        df.to_parquet('data/parquet').compute()


def main():
    if not (os.path.exists(PREPROCESSED_PATH) and os.path.exists(FASTTEXT_FILEPATH)):
        basic_preprocess(ef.get_tokenizer(), ef.data_gen())

    # fasttext training
    if not os.path.exists(FASTTEXT_MODELPATH):
        ef.setup_and_train_fasttext_data(ef.data_gen(), filepath=FASTTEXT_FILEPATH, modelpath=FASTTEXT_MODELPATH)

    COSINE_FILE = 'data/cosine.json.bz2'
    if not os.path.exists(COSINE_FILE):
        model = ef.load_embeddings(FASTTEXT_MODELPATH, get_model=True)
        similarity = ef.get_similarity_measure(KEYWORDS, model, tokenizer=None)
        print("Computing cosine similarities")

        similarities = []
        with bz2.open(COSINE_FILE, 'wb') as d_file:
            for data in tqdm(ef.data_gen([OUT_PATH])):
                data['cosine_similarity'] = similarity(data['tokenized'].split(' '))
                similarities.append(data['cosine_similarity'])

                d_file.write((json.dumps(data) + '\n').encode('utf-8'))

        similarities.sort()
        print("avg similarity: ", sum(similarities) / len(similarities))
        print("median similarity: ", similarities[int(len(similarities) / 2)])

    if not os.path.exists(OUT_PATH):
        final_count = 0
        with bz2.open(OUT_PATH, 'wb') as d_file:
            for data in tqdm(ef.data_gen([COSINE_FILE])):
                if data['cosine_similarity'] < COSINE_THRESHOLD:
                    final_count += 1
                    d_file.write((json.dumps(data) + '\n').encode('utf-8'))
        print("Finished Preprocessing, final number of quotes:", final_count)


if __name__ == "__main__":
    main()
