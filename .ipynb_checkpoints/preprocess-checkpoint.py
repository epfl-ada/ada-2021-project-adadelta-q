import os.path

import src.embeddings_filter as ef
import bz2
from tqdm import tqdm
import json
import os

OUT_PATH = 'data/final_filtered.json.bz2'

PREPROCESSED_PATH = 'data/preprocessed.json.bz2'
PROBABILITY_THRESHOLD = 0.3
FASTTEXT_FILEPATH = './data/fasttextfile.txt'
FASTTEXT_MODELPATH = './data/data.vec'
KEYWORDS = ['market', 'stock', 'bonds', 'shares', 'obligations']

COSINE_THRESHOLD = 0.3


def basic_preprocess(tokenize=ef.get_tokenizer(),dataloader=ef.data_gen(), processed_filepath=OUT_PATH,fasttext_filepath=FASTTEXT_FILEPATH):
    number_of_quotes = 0
    dropped_none = 0
    dropped_prob = 0
    dropped_no_token = 0

    with open(fasttext_filepath, 'w') as fastfile:
        with bz2.open(processed_filepath, 'wb') as d_file:
            for data in tqdm(dataloader):
                number_of_quotes += 1

                if data['speaker'] == 'None':
                    dropped_none +=1
                    continue
                if float(data['probas'][0][1]) < PROBABILITY_THRESHOLD:
                    dropped_prob += 1

                token = tokenize(data['quotation']) + '\n'
                if len(token) == 0:
                    dropped_no_token +=1
                    continue
                del data['probas']
                del data['phase']
                fastfile.write(token)
                data['tokenized'] = token
                d_file.write((json.dumps(data)+'\n').encode('utf-8'))
    print('Finished initial Preprocessing')
    print("Total processed Quotees: ", number_of_quotes)
    print('Of which no speaker: ', dropped_none)
    print('Of which uncertain: ', dropped_prob)
    print('Of which not tokenizable: ', dropped_no_token)

    print("Total Quotes written: ", number_of_quotes - dropped_no_token - dropped_prob - dropped_none)


def main():

    # basic preprocessing
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

                    d_file.write((json.dumps(data)+'\n').encode('utf-8'))

        similarities.sort()
        print("avg similarity: ", sum(similarities)/len(similarities))
        print("median similarity: ", similarities[int(len(similarities)/2)])

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
