# Geographic Entity Recognition

This repository contains code for extracting geographic entities from Wikipedia descriptions of places within the United Kingdom. The project centres around an NER transformer model using the python package [AllenNLP](https://allennlp.org) with integrates with [Optuna](https://optuna.org/) to automate the hyperparameter search. [Poetry](https://python-poetry.org/) is used for the virtual environment and dependency control. Manual labelling was assisted through the text annotation tool [Doccano](https://doccano.herokuapp.com/). 

A live demo is available at https://share.streamlit.io/cjber/ger_wiki/ger_streamlit.py.

## Workflow

* Labelled SpaceEval corpus converted to CoNLL format keeping only **PATH** and **PLACE** entities
    - `data_processing/preprocess/spaceeval_conll.py`
* Roberta transformer model with Optuna optimisation fine-tuned using this data 
    - `configs/ger_transformer.jsonnet`
* DBPedia queried to obtain place descriptions within the UK 
    - `data_processing/preprocess/dbpedia_query.py`
    - Preliminary entity tags generated using transformer model predictions 
        - `ger_wiki/predictor.py`
    - Tags converted into format compatible with Doccano 
        - `data_processing/preprocess/doccano_functions.py`
    * Doccano used to manually verify and fix incorrect labels
* Transformer model fine-tuned using labelled Wikipedia data 
    - `configs/wiki_best.jsonnet`
* Fine-tuned transformer model used to extract all place names and place nominals from corpus of 35,000 place descriptions 
    - `ger_wiki/batch_predictor.py`

SpaceEval data is not included as part of this repository but the corpus is available [here](http://alt.qcri.org/semeval2015/task8/index.php?id=data-and-tools). The Wikipedia query may be ran to obtain place descriptions. The manually labelled Wikipedia data is included in this repository for the training of the `wiki` NER model.

This project hopes to capture mentions of place not normally found within gazetteers to inform future research.

## Reproduce Results

### Wikipedia NER Model

Labelled Wikipedia data is included in this repository and may be used to train the NER model presented to reproduce the results of this study.

NOTE: Running this model will download a ~500mb model archive.

* Clone this repository `git clone https://github.com/cjber/ger_wiki.git`
* Install Poetry: https://python-poetry.org/docs/#installation
* Install dependencies `poetry install` and activate shell `poetry shell`
* Train Wikipedia NER model using optimised hyperparameters `python main.py wiki`

If you want to replicate the dataset of geographic entities mentioned on Wikipedia place articles:

* Run `python ./data_processing/d1_base_process.py` NOTE: This will likely produce an error due to missing SpaceEval data but will still create the Wikipedia data.
* Run `python main.py wiki --predict`

## Unit Testing

Run `pytest` from the base directory of this project for tests.
