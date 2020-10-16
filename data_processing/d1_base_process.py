from data_processing.preprocess.spaceeval_conll import SpaceConll
from pathlib import Path

"""
This section does not need to be run more than once. It converts the ISO-Space
data to CoNLL format and creates data formatted to input into Doccano from
the Wikipedia corpus.
"""
data_path = Path('./data_processing/data/')

space_conll = SpaceConll()
# output conll train data
space_conll.output_conll(
    input_dir=data_path / 'raw/spaceeval_data/train/',
    output_file=data_path / 'processed/raw_conll/spaceeval_train.conll'
)
# output conll test data
space_conll.output_conll(
    input_dir=data_path / 'raw/spaceeval_data/test/',
    output_file=data_path / 'processed/raw_conll/spaceeval_test.conll'
)
