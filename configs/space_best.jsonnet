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
        "type": "ger_reader"
    },
    "model": {
        "calculate_span_f1": true,
        "constrain_crf_decoding": true,
        "dropout": 0.7203112352716297,
        "encoder": {
            "input_dim": 768,
            "type": "pass_through"
        },
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "max_length": 512,
                    "model_name": "roberta-base",
                    "type": "pretrained_transformer_mismatched"
                }
            }
        },
        "type": "crf_tagger",
        "verbose_metrics": true
    },
    "numpy_seed": 42,
    "pytorch_seed": 42,
    "random_seed": 42,
    "train_data_path": "./data_processing/data/processed/spaceeval_train.conll",
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
            "lr": 5.337987218028524e-05,
            "type": "huggingface_adamw",
            "weight_decay": 0.033600041141681095
        },
        "patience": 8,
        "use_amp": true,
        "validation_metric": "+f1-measure-overall"
    },
    "validation_data_path": "./data_processing/data/processed/spaceeval_test.conll"
}
