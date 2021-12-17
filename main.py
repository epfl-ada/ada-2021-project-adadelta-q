import urllib.error

import wget
import src.embeddings_filter as ef
import os
from tqdm import tqdm
import pyarrow as pa
import argparse
from pyarrow import parquet as pq
import pandas as pd
import itertools

def download_files(filenames=ef.PATH_TO_FILES, links=ef.DOWNLOAD_LINKS):

    print('Checking and downloading missing files')
    for filename, link in zip(filenames, links):
        if not os.path.exists(filename):
            try:
                wget.download(link, filename)
            except urllib.error.HTTPError:
                print('File appears unavailable. Contact the repository host to obtain a copy manually')
                print('Missing file: ',filename)


download_links = ['https://drive.google.com/file/d/10W_QJFsvzfBJWyeYB938RcP69v-I5z68/view?usp=sharing']
download_names = [os.path.join('data','fasttext.vec')]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pretrained", type=bool)
    args = parser.parse_args()
    if args.use_pretrained:
        download_files(download_names,download_links)


    else:

        download_files()
        ## TODO:
        # - set seed
        # - Download
        # - Read/write as parquet
        # - fasttext training
        # - fasttext filter (+ Options?)
        # - sentiment analysis
        # - output parquet


        print("writing dataset as parquet for fast processing")




if __name__ == "__main__":
    main()