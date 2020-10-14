# Geographic Entity Recognition

This repository contains the workflow for extracting geographic entities from Wikipedia descriptions of places within the United Kingdom. The project centres around an NER transformer model using the python package [AllenNLP](https://allennlp.org) with integrates with [Optuna](https://optuna.org/) to automate the hyperparameter search. [Poetry](https://python-poetry.org/) is used for the virtual environment and dependency control. Manual labelling was assisted through the text annotation tool [Doccano](https://doccano.herokuapp.com/). The workflow is as follows:

* Labelled ISO-Space corpus converted to CoNLL format keeping only **PATH** and **PLACE** entities
    - `data_processing/preprocess/spaceeval_conll.py`
* Roberta transformer model with Optuna optimisation trained using this data 
    - `configs/ger_transformer.jsonnet`
* DBPedia queried to obtain place descriptions within the UK 
    - `data_processing/preprocess/dbpedia_query.py`
    - Preliminary entity tags generated using transformer model predictions 
        - `ger_wiki/predictor.py`
    - Tags converted into format compatible with Doccano 
        - `data_processing/preprocess/doccano_functions.py`
    * Doccano used to manually verify and fix incorrect labels
* Transformer model fine-tuned using labelled Wikipedia data 
    - `configs/ger_transformer_fromarchive.jsonnet`
* Fine-tuned transformer model used to extract all mentions of place from corpus of 52,000 place descriptions 
    - `ger_wiki/batch_predictor.py`

The data is not included as part of this repository but the ISO-Space corpus is available [here](http://alt.qcri.org/semeval2015/task8/index.php?id=data-and-tools) and the Wikipedia query may be ran to obtain place descriptions. Labelled place descriptions and extracted places will be made available in the future. This project hopes to capture mentions of place not normally found within gazetteers to inform future research.

Tests may be ran by running `pytest` from the base directory of this project.
