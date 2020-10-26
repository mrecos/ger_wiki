import os

import jsonlines
import pandas as pd
import spacy
from allennlp.data.dataset_readers.dataset_utils import bioul_tags_to_spans
from allennlp.models.archival import load_archive
from spacy.pipeline import Sentencizer
from tqdm import tqdm

from ger_wiki.predictor import TextPredictor


class RunBatchPredictions:
    def __init__(self,
                 archive_path: str,
                 predictor_name: str,
                 text_path: str,
                 cuda_device: int,
                 language: str = "en_core_web_sm"):
        archive = load_archive(archive_path, cuda_device=0)
        self.predictor = TextPredictor.from_archive(
            archive,
            predictor_name=predictor_name
        )

        self._nlp = spacy.load(language)
        sentencizer = Sentencizer()
        self._nlp.add_pipe(sentencizer)

        self.text = self.read_lines(text_path)

    def read_lines(self, text_path):
        csv = pd.read_csv(text_path, index_col=0)
        # very simple sentencizer (spacy far too slow)
        csv['abs'] = csv['abs']\
            .str.split('\. ')
        csv = csv.explode('abs')
        csv['abs'] = csv['abs'].astype(str)
        # remove very short sentences that are likely incomplete
        csv = csv[csv['abs'].apply(lambda x: len(x) > 10)]

        return [{'sentence': row['abs'],
                 'place': row['label'],
                 'wiki_point': row['point']} for _, row in csv.iterrows()]

    def run_batch_predictions(self, batch_size):
        chunks = (len(self.text) - 1) // batch_size + 1
        self.predictions = []
        for i in tqdm(range(chunks)):
            batches = self.text[i*batch_size: (i+1)*batch_size]
            batches_out = self.predictor.predict_batch_json(batches)
            for batch in batches_out:
                self.predictions.append(batch)

    def write_json(self, json_file):
        if os.path.exists(json_file):
            os.remove(json_file)

        with jsonlines.open(json_file, mode='w') as writer:
            for line in self.predictions:
                writer.write(line)

    def write_csv(self, csv_file):
        if os.path.exists(csv_file):
            os.remove(csv_file)

        for batch in self.predictions:
            words = batch['words']
            spans = bioul_tags_to_spans(batch['tags'])
            tags_list = []

            for span in spans:
                offsets = span[1]
                label = span[0]
                word = ' '.join(words[offsets[0]: offsets[1]+1])
                tags_list.append((word, label,
                                  batch['sentence'],
                                  batch['place'],
                                  batch['wiki_point']))

            if tags_list != []:
                tags_dataframe = pd.DataFrame(tags_list)
                tags_dataframe.columns = ['Place', 'Type',
                                          'Sentence', 'Place', 'wiki_point']
                tags_dataframe.to_csv(
                    csv_file,
                    mode='a',
                    header=False,
                    index=False
                )
