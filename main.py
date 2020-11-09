#!/usr/bin/env python
import logging

import typer
from allennlp.commands.train import train_model_from_file

from ger_wiki.batch_predictor import RunBatchPredictions
from ger_wiki.optimisation import Optimiser

# logging.getLogger('allennlp.common.params').disabled = True
# logging.getLogger('allennlp.common.util').disabled = True
# logging.getLogger('allennlp.nn.initializers').disabled = True
# logging.getLogger('allennlp.modules.token_embedders.embedding').disabled = True
# logging.getLogger('urllib3.connectionpool').disabled = True

app = typer.Typer()


def optuna_model(name: str):
    archive = name == 'archive'
    typer.echo(f"Running Optuna {name}")
    optimiser = Optimiser(study_name=f'optuna_{name}',
                          timeout=60*60*10,
                          archive=archive)
    optimiser.run_optimisation(
        config_file=f'./configs/{name}_optuna.jsonnet',
        best_output=f"./configs/{name}_best.jsonnet",
        n_trials=50
    )
    optimiser.save_metrics()
    optimiser.delete_archives()


def train_model(name: str):
    typer.echo(f"Running {name}")
    try:
        train_model_from_file(
            parameter_filename=f'./configs/{name}.jsonnet',
            serialization_dir=f"./models/{name}_model",
            include_package=['ger_wiki', 'allennlp_models'],
            force=True
        )
    except FileNotFoundError as e:
        print(e)


def get_predictions(name: str):
    # run predictions on Wikipedia corpus using second model
    batch_predictor = RunBatchPredictions(
        archive_path=f'./models/{name}_best_model/model.tar.gz',
        predictor_name='text_predictor',
        text_path='./data_processing/data/raw/wiki/wiki_info.csv',
        cuda_device=0
    )
    batch_predictor.run_batch_predictions(batch_size=8)
    batch_predictor.write_csv(
        csv_file='./data_processing/data/results/predictions.csv'
    )
    batch_predictor.write_json(
        json_file='./data_processing/data/results/predictions.json'
    )


def create_predictions():
    batch_predictor = RunBatchPredictions(
        archive_path='./models/space_model/model.tar.gz',
        predictor_name='text_predictor',
        text_path='./data_processing/data/interim/wiki/predict.csv',
        cuda_device=0
    )
    batch_predictor.run_batch_predictions(batch_size=8)
    batch_predictor.write_json(
        json_file='./data_processing/data/interim/wiki/predictions.jsonl'
    )


def main(name: str,
         baseline: bool = False,
         optimise: bool = False,
         predict: bool = False,
         create_pseudo: bool = False):
    """
    Choose an NER model to train, or use model predictor.

    By default trains the chosen model using best config, generated
    through --optimise

    :param name str: Name from space, wiki, archive\n
    :param baseline bool: Train baseline model\n
    :param optimise bool: Optimise using optuna\n
    :param predict bool: Label Wikipedia corpus using predictor\n
    :param create_pseudo bool: Create pseudo labels using predictor\n
    """
    if optimise:
        optuna_model(name)
    elif predict:
        get_predictions(name)
    elif create_pseudo:
        create_predictions()
    else:
        name += "_baseline" if baseline else "_best"
        train_model(name)


if __name__ == '__main__':
    typer.run(main)
