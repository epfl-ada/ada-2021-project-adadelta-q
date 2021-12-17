# ADADELTA-Q Project Repository: 'Whose Sentiment towards the Market influences the Market Sentiment?'
This is the group project repository for the group ADADELTA-Q of the EPFL Applied Data Analysis (2021) course.
## Table of Contents
1. [Abstract](#abstract)
2. [Research Questions](#research-questions)
3. [Data used](#Data)
4. [Methodology](#Methodology)
5. [Proposed Timeline](#proposed-timeline)
6. [Organization within the Team](#organization-within-team)
7. [Appendix](#appendix)
## Abstract
The goal of this project is to construct a pipeline that allows to identify those individuals whose quotes have an impact on the stock market and its fluctuations. We use the [Quotebank](https://zenodo.org/record/4277311#.YX0LcpuxW0o) dataset as it includes a large amount of quotations from different sources since 2015. We filter these quotes to retain only quotations related to the stock market. Then, we classify the sentiment of each quote as either positive, neutral, or negative, and we study whether for each individual these sentiments are correlated with a sudden change in the US stock market.
![General Overview of Data Pipeline](./data/diagram_ada.png "Proposed Pipeline")
## Research Questions
According to the Efficient Market Hypothesis, asset prices reflect the value of the given assets given all the information available at a certain moment in time. Therefore, new information is expected to lead to changes in prices and more assets being traded. Even less informative news by influential people might change the expectation of investors and lead to change in prices. Who has this influence on the market? This is the main research goal of this project, which we want to study by analyzing the following research questions:
1. To what extent do published quotations sentiment (detected as positive, negative or neutral) impact the stock market? 
2. Which are the people that have affected the financial market the most? 

To answer these questions we detect the sentiment expressed in the quotations, as a sanity check that the sentiment detected is reasonable we answer the following question:
1. Does the sentiment detected correspond to financial events that happened between 2015 and 2020?

## Data

### Quotebank
From [Quotebank](https://zenodo.org/record/4277311#.YX0LcpuxW0o), we only consider the years from 2015 until 2020. 

### Stock Data

We will make use of the SPY(S&P500) dataset from 2015-01-01 until 2020-04-16 that can be retrieved from the yahoofinance library. We employ the S&P500 index since it is often used as a proxy for the general market trend. We will perform the analysis using both the daily opening value of the index ("Open") as well as the traded volume ("Volume"). The latter is employed to account for the fact that certain events (e.g. published quotations) might have opposite effects on different stocks included in the index leading to a zero-net change in the index value, instead we would detect a spike on the volume traded.

### Sentiment analysis
Since the dataset with sentiment analysis is too large for github, if you want to run the notebook you can download the dataset from the following link:
https://drive.google.com/drive/u/2/folders/1Cmvi3-6fsMT4bRjNaohUJmJNcqG0bR21

#### Preprocessing
- The [EDA](https://github.com/epfl-ada/ada-2021-project-adadelta-q/blob/icapado-patch-1/S%26P500_analysis.ipynb) revealed multiple missing values corresponding to festivities or weekend days, as expected. We will deal with this challenge in the following way: since the information accumulated over the weekend impacts the market on the first opening day, we consider the quotes that have been published during closing days as if they were published on the following first opening day.
- For the prices we considered the variation between the opening stock value of the current day and the opening value of the following day.
- Because we found that some speakers were mentioned with different names, we grouped all these quotes under the same individual.
- To simplify our analysis we decided to keep only the hundred most quoted people. 

## Methodology
The pipeline of the methods can be summarized as follows:
### Topic Filtering
The selection of the quotes that are relevant to the topic of financial markets is done using vector representations of words, in particular we employ [fastText](https://fasttext.cc/) , a state-of-the-art model for text representations. 
We tokenize the data and convert to lower case, removing any english stopwords. 
We then train the model using the standard unsupervised fastText. This way, each quote is transformed into a set of word embeddings. We define a set of keywords that we consider representative of the topic we are analyzing, i.e.`[stock market, stock, bonds, shares, obligations, finance]` and compute the median vector representation of the words. We then compare the embedding of each word in a quote to the computed key word vector and choose the closest distance as representative value for similarity of the whole quote. We filter quotes by some threshold of similarity to reduce data.

### Sentiment Classification
To perform this step we considered three possible options: unsupervised methods, pre-trained BERT, or transfer learning. 

#### Pre-trained BERT
Since Quotebank has been developed using BERT, this model was our first choice. The pre-trained BERT was not able to correctly detect the polarities of two simple artificial sentences, which shows the need of fine-tuning the model for our task. Since we do not have labeled data, we looked into unsurpassed methods. 

#### Unsupervised methods
The fastest and easiest to implement method was VADER from the NLTK library. An experiment done with 1000 quotations showed us that a large percentage of times the vader model simply classified a quotation as neutral even if the sentiment was clearly more radical. 

#### Transfer Learning 
Lately, "Zero-Shot NLP models" have been popularized, which are pre-trained models able to perform well on unseen and unlabeled data without needing fine-tuning. 
The models available in the Transformers' library are trained using the Natural Language Inference (NLI) approach. Using the same 1000 quotations [we used for testing](https://github.com/epfl-ada/ada-2021-project-adadelta-q/blob/Luca/sentiment_analysis.ipynb) VADER and using the default model roberta-large-mnli, we found a more intuitive and reasonable classification of the sentences' polarities. We will thus use this method to classify all the quotations. 
The running time is around 20 hours on a ryzen 5900 with an RTX 3060Ti graphics card.

#### Correlation analysis
Once the market-related quotes are selected and classified according to their sentiment, we combine this data with the stock market data described above. We plan on doing this into two different ways:
1. We compare the correlation between single speakers quotes sentiment and the daily variation of the stock market.
2. For each speaker we test whether the traded volume is larger on average in the days when they expressed one or more sentiment-carrying quotes. 

We used spearman rank correlation for the first task and one-sided hypothesis testing for the second one.
We then compare the two results.
## Proposed Timeline

We propose the following timeline (numbers refer to list [here](#organization-within-team)): 
- [x] P1 (Individual):
   - [x] Idea 1
   - [x] Idea 2
   - [x] Idea 3
- [x] P2:
  - [x] 1., 2. fully and 3. partially implemented
  - [x] Feasibility analysis of 4.-7.
  - [x] README.md
  - [x] Notebook
- [x] P3:
   - [x] Updated Readme
   - [x] Updated Notebook
   - [x] Data Story
Should be confirmed with [webiste](https://dlab.epfl.ch/teaching/fall2021/cs401/projects/) before each deadline.
## Organization within Team

1. [Filter Quotebank data](./preprocess.py) using a semantic filter to obtain a subset of topic relevant data.: (AdadeltaQ team)
   - [x] Identify most suitable (pretrained) filter (fasttext)
   - [x] Apply filter
   - [x] Fine-Tune Hyperparameters (e.g. cutoff values): to get reasonably sized data
2. Process data so that quotes can easily be matched to speakers: (AdadeltaQ team)
   - [x] Remove irrelevant data (could also be done in 1. -> has been done in 1)
   - [x] Group by speakers 
3. Apply a sentiment classifier with respect to a topic: (AdadeltaQ team)
   - [x] Identify suitable (pretrained on news data) sentiment filter (fasttext/BERT/Zero-shot)
   - [x] Apply
   - [x] Fine-tune
4. Merge topic speakers with time-series of topic data (S&P500): (AdadeltaQ team)
   - [x] Identify what stock/index we want
   - [x] Identify suitable Data source ([Yahoo finance](https://pypi.org/project/yfinance/) ) 
   - [x] Preprocess stock  (remove irrelevant fields, ensure easy format for merging)
   - [x] Merge data
5. Find a possible correlation between speakers quotes sentiment and market shares value: (AdadeltaQ team)
   - [x] Identify how (spearman rank correlation)
6. Find the speaker whose quotes had an impact on the traded volume: (AdadeltaQ team)
   - [x] Identify how (one-side hypothesis tests)
7. Compare the two results: (AdadeltaQ team)
   - [x] Scatter plot of the p values.
8. Review: (AdadeltaQ team)
   - [x] Critically identify biases and other issues with our pipeline
   - [x] Fix what can be fixed in the time given 
   - [x] [Report any issues related to Quotebank](https://docs.google.com/forms/d/e/1FAIpQLSfe14V9gKV3chVSC7_Y_mTIJz_YcvgbIaxGSESmH1kS9RbcZA/viewform)
   - [x] Final Notebook
   - [x] Datastory

To avoid confusion we would like to point out that our team met regularly during the course of the semester to work on the project together and each member contributed to the development in all stages both with code, text and ideas on how to improve the project.
The work was never soley done by a subset of the team.
## Appendix

### Repository Structure
```
ada-2021-project-adadelta-q
│   README.md
│   filtering.py    # filtering script to process entire dataset
│   filtering_tests.ipynb    # some basic tests on filtering
│   sentiment_analyisis.ipynb # Sentiment analiysis tetss
│   S\&P500_analysis.ipynb # Stock data trials
└───data  # store data here (will be ignored by git)
│   
└───src    # for supporting code
```
### Downloading Data
On any linux system with wget:
```
cd data
wget -i downloadables.txt 
```
will download all Quotebank data from 2015 to 2020.

### Environment Setup using [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html)

To setup, run:
```
conda create --name <env> --file requirements.yml
```
where `<env>` is the desired environment name.
### Running the preprocessing code
```shell
conda activate <env>
python main.py
```
This will download the entire dataset and run the entire preprocessing pipeline, including topic filtering and augmentation with sentiments.
As the entire pipeline can take 24+ hours to finish and is subject to certain requirements, such as a cuda-capable GPU to finish in a reasonable timeframe, we have added the option to download the preprocessed data directly.
Simply add the option `--use_pretrained` to download the filtered dataset directly.
