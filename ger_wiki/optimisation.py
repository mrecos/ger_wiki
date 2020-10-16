import csv
from collections import OrderedDict

import optuna
import optuna.integration.allennlp
from optuna.importance import get_param_importances
from optuna.pruners import HyperbandPruner


class Optimiser:
    def __init__(self,
                 study_name: str,
                 archive: bool = False,
                 timeout: int = None):
        self.study_name = study_name
        self.timeout = timeout
        self.archive = archive

        self.MODEL_PATH = './models/optuna_models/' + self.study_name

        self.study = optuna.create_study(
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            direction='maximize',
            pruner=HyperbandPruner(),
            storage='sqlite:///optuna_database/' + self.study_name + '.db',
            load_if_exists=True
        )

    def objective(self, trial: optuna.Trial) -> float:
        trial.suggest_float('lr', 1e-6, 1e-4)
        trial.suggest_categorical('batch_size', [4, 8, 14])
        trial.suggest_float('weight_decay', 0.0, 0.1)

        if not self.archive:
            trial.suggest_float('dropout', 0.0, 0.8)

        executor = optuna.integration.allennlp.AllenNLPExecutor(
            trial=trial,
            config_file=self.config_file,
            serialization_dir=self.MODEL_PATH + f'/trial_{trial.number}',
            metrics='best_validation_f1-measure-overall'
        )
        return executor.run()

    def run_optimisation(self,
                         config_file: str,
                         best_output: str,
                         n_trials: int = None):
        self.config_file = config_file

        self.study.optimize(
            self.objective,
            n_jobs=1,
            n_trials=n_trials,
            timeout=self.timeout
        )

        optuna.integration.allennlp.dump_best_config(
            self.config_file,
            best_output,
            self.study
        )
        print('Number of finished trials: ', len(self.study.trials))
        print('Best trial:')
        trial = self.study.best_trial

        print('  Value: ', trial.value)
        print('  Params: ')
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

    def save_metrics(self):
        trials_importance: OrderedDict = get_param_importances(self.study)

        with open("./data_processing/data/results/" +
                  self.study_name + "_importance.csv", "w") as csv_file:
            w = csv.writer(csv_file)
            for key, val in trials_importance.items():
                w.writerow([key, val])

        trails_df = self.study.trials_dataframe()
        trails_df.to_csv("./data_processing/data/results/"
                         + self.study_name + "_history.csv",
                         mode='w',
                         index=False)
