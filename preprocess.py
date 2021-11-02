import os.path

import src.embeddings_filter as ef
import bz2
from tqdm import tqdm
import json
import os

OUT_PATH = 'data/preprocessed.json.bz2'
PROBABILITY_THRESHOLD = 0.3
FASTTEXT_FILEPATH = './data/fasttextfile.txt'


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
                d_file.write((json.dumps(data)+'\n').encode('utf-8'))
    print('Finished initial Preprocessing')
    print("Total processed Quotees: ", number_of_quotes)
    print('Of which no speaker: ', dropped_none)
    print('Of which uncertain: ', dropped_prob)
    print('Of which not tokenizable: ', dropped_no_token)

    print("Total Quotes written: ", number_of_quotes - dropped_no_token - dropped_prob - dropped_none)


def main():
    basic_preprocess(ef.get_tokenizer(), ef.data_gen())

    # basic preprocessing
    if not (os.path.exists(OUT_PATH) and os.path.exists(FASTTEXT_FILEPATH)):
        basic_preprocess(ef.get_tokenizer(), ef.data_gen())

    # fasttext training
    ef.setup_and_train_fasttext_data(ef.data_gen(), filepath=FASTTEXT_FILEPATH)







if __name__ == "__main__":
    main()
