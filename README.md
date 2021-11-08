# ADADELTA-Q Project Repository: 'Trendsetters'
This is the group project repository for the group ADADELTA-Q of the EPFL Applied Data Analysis (2021) course.
## Table of Contents
1. [Abstract](#abstract)
2. [Research Questions](#research-questions)
3. [Proposed Additional Datasets](#proposed-additional-datasets)
4. [Methods](#methods)
5. [Proposed Timeline](#proposed-timeline)
6. [Organization within the Team](#organization-within-team)
7. [Appendix](#appendix)
## Abstract
The Idea of this project is to construct a general pipeline to identify trends within a topic/community subset of the [Quotebank](https://zenodo.org/record/4277311#.YX0LcpuxW0o) dataset and identify the individuals that have set a trend and those that are following it.
We further intend to apply the pipeline directly to stock data.
## Research Questions
1. Do the published quotations impact the market stock value? 
2. After how long can we notice the effect of the quotations?
3. Which are the people that have affected it the most? Are them always the same?
4. The words said by one person are sufficient for having an impact on it, or they became relevant when other people sustain them? 

## Proposed Additional Datasets
### Stock Data
We will need the SPY(S&P500) dataset from 2015-2020 that we can find from the yahoofinance library (we will consider just the "Close" column of the dataframe that represents the final S&P500 daily value).
## Methods
### Topic Filter
We obtain fasttext word vectors by training fasttext on the entire dataset. We define a set of keywords for our topic. 
We filter according to the cosine similarity between the keywords and the quotes.
#### Preprocessing
An initial training on fasttext with raw quotes shows that some preprocessing is necessary because the most similar terms where simply stems of each other.
We tokenize the data and convert to lower case, removing any english stopwords. We save the quotes in a new file so that they can be read by the fasttext library.
#### Fasttext Training
We train using the standard fasttext library in an unsupervised way.
#### Filtering
We filter by cosine similarity to the keywords.
## Proposed Timeline

We propose the following timeline (numbers refer to list [here](#organization-within-team)): 
- [x] P1 (Individual):
   - [x] Idea 1
   - [x] Idea 2
   - [x] Idea 3
- [ ] P2:
  - [ ] 1., 2. fully and 3. partially implemented
  - [ ] Feasibility analysis of 4.-7.
  - [ ] Readme
  - [ ] Notebook
- [ ] P3:
   - [ ] Updated Readme
   - [ ] Updated Notebook
   - [ ] Data Story
Should be confirmed with [webiste](https://dlab.epfl.ch/teaching/fall2021/cs401/projects/) before each deadline.
## Organization within Team

1. Filter Quotebank data using a semantic filter to obtain a subset of topic relevant data.:
   - [ ] Identify most suitable (pretrained) filter (fasttext)
   - [ ] Apply filter
   - [ ] Fine-Tune Hyperparameters (e.g. cutoff values) to get reasonably sized data
2. Process data so that quotes can easily be matched to speakers
   - [ ] Remove irrelevant data (could also be done in 1.)
   - [ ] Group by speakers
3. Apply a sentiment classifier with respect to a topic
   - [ ] Identify suitable (pretrained on news data) sentiment filter (fasttext/BERT)
   - [ ] Apply
   - [ ] Fine-tune
4. Merge topic speakers with time-series of topic data (e.g. S&P500)
   - [ ] Identify what stock/index we want
   - [ ] Specification of what data format we need (Granularity, Fields)
   - [ ] Identify suitable Data source (e.g [Yahoo finance](https://pypi.org/project/yfinance/) / [SimFin](https://github.com/SimFin/simfin) ) 
   - [ ] Obtain (API Keys?) Data
   - [ ] Preprocess stock  (remove irrelevant fields, ensure easy format for merging)
   - [ ] Merge data
5. Filter speakers that have low/now correlation with topic data
   - [ ] Identify how, e.g. [Granger Causailty](https://en.wikipedia.org/wiki/Granger_causality)
6. Compare speakers with each other
7. Filter speakers that are *'shadowing'* other speakers.
8. Review
   - [ ] Critically identify biases and other issues with our pipeline
   - [ ] Fix what can be fixed in the time given 
   - [ ] [Report any issues related to Quotebank](https://docs.google.com/forms/d/e/1FAIpQLSfe14V9gKV3chVSC7_Y_mTIJz_YcvgbIaxGSESmH1kS9RbcZA/viewform)
   - [ ] Final Notebook
   - [ ] Datastory


## Appendix

### Repository Structure
```
ada-2021-project-adadelta-q
│   README.md
│   notebook.ipynb    
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

### Environment
To setup, run:
```
conda create --name <env> --file requirements.txt
```

