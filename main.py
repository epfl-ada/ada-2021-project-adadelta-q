import wget
import src.embeddings_filter as ef
import os
from tqdm import tqdm
import pyarrow as pa
import argparse
from pyarrow import parquet as pq
import pandas as pd


def download_files():

    print('Checking and downloading missing files')
    for filename, link in zip(ef.PATH_TO_FILES, ef.DOWNLOAD_LINKS):
        if not os.path.exists(filename):
            wget.download(link, filename)


    for line in ef.data_gen():


download_links = []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pretrained", type=bool)
    args = parser.parse_args()
    if args.use_pretrained():
        ## TODO:
        # - Download
        # - profit
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
        gen = ef.data_gen()
        table = pa.Table.from_pandas(pd.DataFrame(next(gen)))
        print("writing dataset as parquet for fast processing")
        with pq.ParquetWriter('full_set.parquet', table.schema) as writer:

            for item in next(gen):
                writer.write_table(table)
                table = pa.Table.from_pandas(pd.DataFrame(next(item)))
            writer.write_table(table)



if __name__ == "__main__":
    main()