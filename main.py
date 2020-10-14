#!/usr/bin/env python
import sys

from allennlp.commands.train import train_model_from_file

from ger_wiki.batch_predictor import RunBatchPredictions
from ger_wiki.optimisation import Optimiser


def main(argv):
    if argv == ['1']:
        print("Optuna training base model.")
        optimiser = Optimiser(study_name='optuna_base', timeout=60*60*10)
        optimiser.run_optimisation(
            config_file='./configs/optuna_transformer.jsonnet',
            best_output="./configs/optuna_ger_best.jsonnet",
            n_trials=50
        )
        optimiser.save_metrics()

    elif argv == ['2']:
        print("Training Base Model with SemEval data.")
        # train base model using SemEval data
        train_model_from_file(
            parameter_filename='./configs/optuna_ger_best.jsonnet',
            serialization_dir='./models/model_base',
            include_package=['ger_wiki', 'allennlp_models'],
            force=True
        )

    elif argv == ['3']:
        print("Optuna training archive model.")
        optimiser = Optimiser(study_name='optuna_fromarchive',
                              archive=True, timeout=60*60*10)
        optimiser.run_optimisation(
            config_file='./configs/optuna_fromarchive.jsonnet',
            best_output="./configs/optuna_fromarchive_best.jsonnet",
            n_trials=50
        )
        optimiser.save_metrics()

    elif argv == ['4']:
        print("Training archive model.")
        train_model_from_file(
            parameter_filename='./configs/optuna_fromarchive_best.jsonnet',
            serialization_dir='./models/model_fromarchive',
            include_package=['ger_wiki', 'allennlp_models'],
            force=True
        )

    elif argv == ['5']:
        print("Running batch predictions.")
        # run predictions on Wikipedia corpus using second model
        batch_predictor = RunBatchPredictions(
            archive_path='./models/model_fromarchive/model.tar.gz',
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

    elif argv == ['1a']:
        print("Create Doccano pseudo labels.")
        # run predictions on Wikipedia corpus using second model
        RunBatchPredictions(
            archive_path='./models/model_base/model.tar.gz',
            predictor_name='text_predictor',
            text_path='./data_processing/data/interim/wiki/predict.csv',
            cuda_device=0
        )
        batch_predictor.run_batch_predictions(batch_size=8)
        batch_predictor.write_json(
            json_file='./data_processing/data/interim/wiki/predictions.jsonl',
        )

    elif argv == ['1b']:
        print("Training baseline model.")
        train_model_from_file(
            parameter_filename='./configs/model_baseline.jsonnet',
            serialization_dir='./models/model_baseline/',
            include_package=['ger_wiki', 'allennlp_models'],
            force=True
        )


if __name__ == '__main__':
    main(sys.argv[1:])
