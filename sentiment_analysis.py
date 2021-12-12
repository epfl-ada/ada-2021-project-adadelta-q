import pandas
from transformers  import BertTokenizer, TFBertForSequenceClassification, pipeline
from transformers.pipelines.base import KeyDataset
from tqdm import tqdm
import json
import bz2
import pyarrow
from datasets import Dataset
import os

def get_dataset_as_dict(path='data/final_filtered.json.bz2'):
    list_of_dict = []
    k = 0
    with bz2.open(path, 'rb') as s_file:
        while True:
            try:
                d = json.loads(next(s_file))
                list_of_dict.append(d)
                k += 1
            except StopIteration:
                break

    return list_of_dict

def get_labels(path='data/final_filtered.json.bz2'):
    # Initialize the zer-shot classifier (it will use the default model robert-large-mnli)
    classifier = pipeline("zero-shot-classification",model='facebook/bart-large-mnli', device=0)

    # Crete the hypothesis we want to use
    hypotheses = "The sentiment of this quote is {}."

    # Create the labels
    the_labels = ["positive", "negative", "neutral"]
    print("Reading data from "+path)
    df = pandas.DataFrame(get_dataset_as_dict(path))
    print("Constructing Dataset")
    dataset = Dataset(pyarrow.Table.from_pandas(df))
    keys = KeyDataset(dataset, 'quotation')
    print("Running Inference")
    classified = [quote['labels'][0] for quote in
                  tqdm(classifier(keys, the_labels, hypothesis_template=hypotheses, multi_label=True))]

    df['sentiment'] = classified

    return df

def main():
    df = get_labels()
    df.to_hdf(os.path.join('data', 'final_w_sentiment.h5'), key='df', mode='w')


if __name__ == "__main__":
    main()