import wget
import os
import argparse

import urllib.error

import src.embeddings_filter as ef
from src.sentiment_analysis import get_and_save_labels
from src.filtering import preprocess_and_filter

# pretrained files download
download_links = ['https://drive.google.com/file/d/10W_QJFsvzfBJWyeYB938RcP69v-I5z68/view?usp=sharing','https://drive.google.com/file/d/15bh9ID0ST5haSDQCumCZoJb6slde7mEy/view?usp=sharing', 'https://drive.google.com/file/d/1fAypETHa6ihn3p0EZG073dsLmGPPjqjX/view?usp=sharing', 'https://drive.google.com/file/d/1aCulyDh-yCdZQHw5BJDMPl1m3-Qjn0A3/view?usp=sharing']
download_names = [os.path.join('data','fasttext.vec'), os.path.join('data','SP500_adj1.csv'), os.path.join('data','final_w_sentiment.parquet.gzip'), os.path.join('data','top100_sentiment.parquet.gzip')]


def download_files(filenames=ef.PATH_TO_FILES, links=ef.DOWNLOAD_LINKS):
    """
    Download files
    :param filenames: file names (how they should be saved)
    :param links: URL to corresponding files
    :return: nothing
    """

    print('Checking and downloading missing files')
    for filename, link in zip(filenames, links):
        if not os.path.exists(filename):
            try:
                wget.download(link, filename)
            except urllib.error.HTTPError:
                print('File appears unavailable. Contact the repository host to obtain a copy manually')
                print('Missing file: ',filename)


def main():
    # parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_pretrained", type=bool)
    args = parser.parse_args()

    # simply download files
    if args.use_pretrained:
        print("Downlaoding required files into the data folder")
        download_files(download_names,download_links)

    # do download, preprocessing, clustering etc...
    else:
        print("Training from scratch, this may take a while....")
        download_files()
        preprocess_and_filter()
        get_and_save_labels()
        print("Done. Produced files are in the data folder")





if __name__ == "__main__":
    main()