{
    "data_loader": {
        "batch_sampler": {
            "batch_size": 4,
            "type": "bucket"
        }
    },
    "dataset_reader": {
        "token_indexers": {
            "tokens": {
                "max_length": 512,
                "model_name": "roberta-base",
                "type": "pretrained_transformer_mismatched"
            }
        },
        "type": "ger_wiki.reader.GerReader"
    },
    "model": {
        "archive_file": "./models/model_base/model.tar.gz",
        "type": "from_archive"
    },
    "numpy_seed": 42,
    "pytorch_seed": 42,
    "random_seed": 42,
    "train_data_path": "./data_processing/data/processed/wiki_train.conll",
    "trainer": {
        "cuda_device": 0,
        "grad_norm": 1,
        "learning_rate_scheduler": {
            "mode": "max",
            "patience": 2,
            "type": "reduce_on_plateau"
        },
        "num_epochs": 50,
        "optimizer": {
            "correct_bias": true,
            "eps": 1e-08,
            "lr": 1.9000671753502963e-05,
            "type": "huggingface_adamw",
            "weight_decay": 0.061748150962771656
        },
        "patience": 8,
        "use_amp": true,
        "validation_metric": "+f1-measure-overall"
    },
    "validation_data_path": "./data_processing/data/processed/wiki_test.conll"
}