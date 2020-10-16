from data_processing.preprocess import doccano_functions as df
from pathlib import Path

"""
This section takes manually labelled entities from Doccano and converts into
training and testing CoNLL formatted data for the second model
"""
data_path = Path('./data_processing/data/')

# convert prediction outputs into doccano input
df.predictions_to_doccano(
    input_file=data_path / 'interim/wiki/predictions.jsonl',
    output_file=data_path / 'interim/wiki/doccano_input.jsonl'
)

# split labelled doccano data for training and evaluation
df.split_text(
    input_file=data_path / 'interim/wiki/doccano/doccano_output.json1',
    large_file=data_path / 'interim/wiki/wiki_train.jsonl',
    sample_file=data_path / 'interim/wiki/wiki_test.jsonl',
    sample=20)

df.doccano_to_conll(
    input_file=data_path / 'interim/wiki/wiki_train.jsonl',
    output_file=data_path / 'processed/wiki_train.conll'
)
df.doccano_to_conll(
    input_file=data_path / 'interim/wiki/wiki_test.jsonl',
    output_file=data_path / 'processed/wiki_test.conll'
)
