{
    "data_loader": {
        "batch_sampler": {
            "batch_size": 8,
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
        "dropout": 0.09227093863122364,
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
            "lr": 4.2324999182073916e-05,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm\\.weight",
                        "layer_norm\\.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "type": "huggingface_adamw",
            "weight_decay": 0.004113041530454253
        },
        "patience": 8,
        "use_amp": true,
        "validation_metric": "+f1-measure-overall"
    },
    "validation_data_path": "./data_processing/data/processed/spaceeval_test.conll"
}