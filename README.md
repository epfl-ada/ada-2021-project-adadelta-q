# ADADELTA-Q Project Repository: 'Trendsetters'
This is the group project repository for the group ADADELTA-Q of the EPFL Applied Data Analysis (2021) course.
## Project Idea
The Idea of this project is to construct a general pipeline to identify trends within a topic/community subset of the [Quotebank](https://zenodo.org/record/4277311#.YX0LcpuxW0o) dataset and identify the individuals that have set a trend and those that are following it.
We further intend to apply the pipeline directly to stock data.
## Proposed implementation steps/TODO


1. Filter Quotebank data using a semantic filter to obtain a subset of topic relevant data.:
   - [ ] Identify most suitable (pretrained) filter (BERT?)
   - [ ] Apply filter
   - [ ] Fine-Tune Hyperparameters (e.g. cutoff values) to get reasonably sized data
2. Process data so that quotes can easily be matched to speakers
   - [ ] Remove irrelevant data (could also be done in 1.)
   - [ ] Group by speakers
3. Apply a sentiment classifier with respect to a topic
   - [ ] Identify suitable (pretrained on news data) sentiment filter
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

## Milestones & Main Deliverables
Should be confirmed with [webiste](https://dlab.epfl.ch/teaching/fall2021/cs401/projects/) before each deadline.
- [x] P1 (Individual):
   - [x] Idea 1
   - [x] Idea 2
   - [x] Idea 3
- [ ] P2:
   - [ ] Readme
   - [ ] Notebook
- [ ] P3:
   - [ ] Updated Readme
   - [ ] Updated Notebook
   - [ ] Data Story

## Repository Structure
```
ada-2021-project-adadelta-q
│   README.md
│   notebook.ipynb    
└─── data  # store data here (will be ignored by git)
│   
└───src    # for supporting code
```
## Downloading Data
On any linux system with wget:
```
cd data
wget -i downloadables.txt 
```
will download all Quotebank data from 2015 to 2020.