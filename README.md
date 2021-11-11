# ADADELTA-Q Project Repository: 'Are random fluctuations of the stock market really random?'
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
The main idea of this project is to construct a pipeline that allows to identify those individuals whose quotes have an impact on the stock market and its fluctuations. We use the [Quotebank](https://zenodo.org/record/4277311#.YX0LcpuxW0o) dataset as it includes a large amount of quotations from different sources since 2015. We filter these quotes using fastTest wordembeddings and cosine similarity in order to retain only quotations related to the stock market. Then, we classify the sentiment of each quote as either positive, neutral, or negative, and based on these sentiments we study whether for each individual these quotes are correlated with a sudden change in the US stock market. To perform the sentiment analysis we explore both unsupervised and supervised methods, for the latter we use pre-trained models, and for the market data we use Yahoo finance which provides data for the value of the stocks, the traded volume and the volatility (double check). 

## Research Questions
According to the Efficient Market Hypothesis, the asset prices of the stock exchange reflect the value of the given asset given all the information available at a certain moment in time. Therefore, new information might lead to changes in the real value of the assets being traded and thus to more exchanges of assets. These information need not to be directly related to a certain company, even less informative news by influential people might change the expectation of investors and lead to change in prices. Who has this influence on the market? This is the main research goal of this project, but in order to be able to answer this question, we will need to answer sequentially the following, more precise, research questions:
1. Do the published quotations impact the market stock value? 
2. After how long can we notice the effect of the quotations?
3. Which are the people that have affected it the most? Are them always the same?
4. The words said by one person are sufficient for having an impact on it, or they became relevant when other people sustain them? 

## Data
In order to perfrom the aforementioned analysis we will employ the [Quotebank](https://zenodo.org/record/4277311#.YX0LcpuxW0o) dataset considering only the years from 2015 until 2020. On the other hand, we will make use of the yahoofinaince library in order to gather the required data regarding the stock market. More precisely, we will employ the S&P500 index since it is often empployed as a proxy for the evolution of the market, and in particular we will perfrom the analysis using both the value itself of the index as well as the traded volume to account for possible opposite effects on different stocks.

## Methodology
The pipeline of the methods used in order to perfrom the analysis can be summarized as follows:
### Topic Filtering
The selection of the quotes that are relevant to the topic of financial markets is done using vector representations of words, in particular we emply fastText (why???). An initial training on fasttext with raw quotes shows that some preprocessing is necessary because the most similar terms where simply stems of each other. We tokenize the data and convert to lower case, removing any english stopwords. We save the quotes in a new file so that they can be read by the fasttext library. We then train the model using the standard fastText library in an unsupervised manner. This way, each quote is transformed into a set of wordembeddings. Then, we (and the user can change this setting) define a set of keywords that we consider representative of the topic we are analyzing. Currenlty the words we are using as seeds are [market, stocks, trade, bonds]. Then the actual filtering takes place, that is for each word in each sentence we computeits the cosine similarity value with the average vector of the seed words and we keep the sentence only if there exist a word whose similarity is larger than a certain threshold (for now set to XXX).  

### Sentiment Classification
Once the irrelevant quotations are filtered out of the dataset, we perform the sentiment analysis to detect the sentiment expressed in each quotation. To perfrom this step we considered three possible options: unsupervised methods, Fine-tuning pre-trained supervised methods, or transfer learning. 

#### Unsupervised methods
Since the Quotebank dataset does not have labels for the sentiment, the frist approach we considered was unsupervised methods. The fastest and easiest to implement method was VADER from the NLTK library. An experiment done with 1000 quotations showed us that a large percentage of times the vader model simply classified a quoation as neutral even if the sentiment was clearly more radical. 

#### Pre-trained supervised methods
Given the poor discriminative performance of VADER, we decided to look into supervised models pretrained on other tasks. Since Quotebank has been developed using BERT, this model was our first choice. To get a first insight about its ability to directly classify the sentiment, we used two short artificial sentences that clearly has opposite sentiment. The pr-trained BERT was not able to distinguish the two opinions. This shows the need of fine-tuning the model for our task, but this goes back to the problem of not having labeled data. 

#### Transfer Learning 
In the past years, or even months, a large attention has been dedicated to the so called "Zero-Shot NLP models", which are models pre-trained on different tasks but that are able to perform on unseen and unlabled data without needing fine-tuning, which is what we need! We use the Transformers library which employs models from the Hugging Face hub. The models available in the Transformers library are trained using the Natural Language Inference (NLI) approach, and thus require for each sentence a premise and an hypothesis to be tested. In our case the hypothesis are the sentiment and thus we set the hypothesis to be "The sentiment of this quote is positive/neutral/negative". It has been shown that the selection of the working of the hypothesis has an influence on the performance so we might need to compare different formulations as well. Nevertheless, using the same 1000 quotations we used for testing VADER and using the default model roberta-large-mnli, we found a more intuitive and reasonable classification of the sentences' polarities. We will thus use this method to classify all the quotations. The running time will be significant (it took approximately 1 hour for 1000 sentences), but since we only need to do it once it is feasible.

### Correlation analysis


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
### Bibliography 
- Hutto, C., & Gilbert, E. (2014). Vader: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the International AAAI Conference on Web and Social Media* (Vol. 8, No. 1).
