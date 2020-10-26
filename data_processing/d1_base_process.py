"""
This section does not need to be run more than once. It converts the ISO-Space
data to CoNLL format and queries Wikipedia creating an overall dataset and 200
article subset for pseudo labels.
"""
from data_processing.preprocess.spaceeval_conll import SpaceConll
from data_processing.preprocess.dbpedia_query import run_query, clean_abs
from pathlib import Path

DATA_PATH = Path('./data_processing/data/')

wiki_csv = run_query()
wiki_csv = clean_abs(wiki_csv)

# sample to create pseudo labels from base model
wiki_csv.sample(n=200).to_csv(DATA_PATH / "interim/wiki/predict.csv")
wiki_csv.to_csv(DATA_PATH / "raw/wiki/wiki_info.csv")

space_conll = SpaceConll()
# output conll train data
space_conll.output_conll(
    input_dir=DATA_PATH / 'raw/spaceeval_data/train/',
    output_file=DATA_PATH / 'processed/raw_conll/spaceeval_train.conll'
)
# output conll test data
space_conll.output_conll(
    input_dir=DATA_PATH / 'raw/spaceeval_data/test/',
    output_file=DATA_PATH / 'processed/raw_conll/spaceeval_test.conll'
)
