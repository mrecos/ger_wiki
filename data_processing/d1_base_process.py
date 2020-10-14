from data_processing.preprocess import doccano_functions as df
from data_processing.preprocess.spaceeval_conll import SpaceConll

"""
This section does not need to be run more than once. It converts the ISO-Space
data to CoNLL format and creates data formatted to input into Doccano from
the Wikipedia corpus.
"""

space_conll = SpaceConll()
# output conll train data
space_conll.output_conll(
    input_dir='./data_processing/data/raw/spaceeval_data/train/',
    output_file='./data_processing/data/processed/raw_conll/spaceeval_train.conll'
)
# output conll test data
space_conll.output_conll(
    input_dir='./data_processing/data/raw/spaceeval_data/test/',
    output_file='./data_processing/data/processed/raw_conll/spaceeval_test.conll'
)
