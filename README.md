[Biohack](http://biohack.ru) is a local hackathon held at St.Petersburg, Russia at the first weekend of sunny spring.

This repo contains code and scripts our team started while working on one of [the Biohack-2019 projects](http://biohack.ru/projects2019) named "Comparison of gene expression data with article texts" by [Alexey Sergushichev](https://github.com/assaron), both with various attempts to analyze these data and to prepare improved gene expression data annotations.

The results are being published here and this is still in progress: we made a lot of different annotations, most of them were not formatted properly during the main event; we also feel that we need to make our work fully reproducible to make it useful in future research (probably with different gene expression datasets).

## Input

We were given the following data sources:

1. ARCHS4 - gene expression data (gene level) - as provided [here](https://amp.pharm.mssm.edu/archs4/download.html);

2. [GeoMetaDB](https://gbnci-abcc.ncifcrf.gov/geo/) as the main source of metadata for various gene expression experiments;

3. PMC PubMed papers Open Access Subset.

## Tasks

The tasks we were given included the following:

- [ ] 1) the preparation of various annotated datasets with different levels of difficulty;
- [ ] 2) the creation of binary classifiers for gender, tissue, experiment protocol, etc.;
- [ ] 3) the creation of multi-label classifiers for corresponding papers MeSH terms, author's keywords, keywords, extracted from paper text or abstract, etc.
- [ ] 4) the analysis of false-positive predictions of keywords (they might appear due to errors in data annotation);
- [ ] 5) the validation of our classifiers' predictions with the help of Gene Ontology analysis tools;
- [ ] 6) to make other things, i.e., to build joint embedding representations for keywords and gene expression profiles, to make the profiles searchable by combination of keywords of different kinds

## THE PROBLEM

The main reason why this project was proposed at the hackathon was GeoMetaDB. Information in this database is incomplete, especially when you compare it with [NCBI GEO data repository](https://www.ncbi.nlm.nih.gov/geo/). We used this repository as an additional place to find missing pieces of data (we refer to the process of missing pieces collection as `task 0` - it was not listed as one of the project tasks, but it seemed wise to do it).

## Repo structure

The repo has the following structure:

- `tasks` contains markdown files with detailed descriptions what we did to solve the tasks and where to find the results (if they are provided) or related scripts. It also contains separate file with description of `task 0`, which appeared here as our way to fix inconsistency in metadata annotations.
- `data` contains all the intermediate datasets used in this project;
- `scripts` contains scripts, `embeddings` contains embeddings, `models` contains various classifiers and related files (the contents is described at tasks files).

## Input Datasets collection

## Authors

- [@latticetower](https://github.com/latticetower)
- [@ALioznova](https://github.com/ALioznova)
- [@Mikhail-Sosnin](https://github.com/Mikhail-Sosnin)
- [@mmurashko](https://github.com/mmurashko)
- [@Nasuli](https://github.com/Nasuli)


