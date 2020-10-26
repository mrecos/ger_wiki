local base_dir = "./data_processing/data";

# fixed parameters
local transformer_model = "roberta-base";
local transformer_hidden_dim = 768;
local max_length = 512;
local epochs = 50;
local seed = 42;

# optimisable parameters
local lr = std.parseJson(std.extVar('lr'));
local batch_size = std.parseJson(std.extVar('batch_size'));
local weight_decay = std.parseJson(std.extVar('weight_decay'));
local dropout = std.parseJson(std.extVar('dropout'));

local patience = 8;
local lr_patience = 2;
local use_amp = true;
local eps = 1e-8;
local grad_norm = 1.0;

{
    "numpy_seed": seed,
    "pytorch_seed": seed,
    "random_seed": seed,
    "dataset_reader": {
        "type": "ger_reader",
        "token_indexers": {
          "tokens": {
            "type": "pretrained_transformer_mismatched",
            "model_name": transformer_model,
            "max_length": max_length
          },
        },
    },
    "train_data_path": base_dir + "/processed/spaceeval_train.conll",
    "validation_data_path": base_dir + "/processed/spaceeval_test.conll",
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        }
    },
    "model": {
        "type": "crf_tagger",
        "calculate_span_f1": true,
        "constrain_crf_decoding": true,
        "dropout": dropout,
        "include_start_end_transitions": false,
        "label_encoding": "BIOUL",
        "verbose_metrics": true,
        "encoder": {
            "type": "pass_through",
            "input_dim": transformer_hidden_dim,
        },
        "text_field_embedder": {
          "token_embedders": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": transformer_model,
                "max_length": max_length
            }
          }
        },
    },
    "trainer": {
        "cuda_device": 0,
        "optimizer": {
          "type": "huggingface_adamw",
          "lr": lr,
          "weight_decay": weight_decay,
          "correct_bias": true,
          "eps": eps
        },
        "learning_rate_scheduler": {
          "type": "reduce_on_plateau",
          "patience": lr_patience,
          "mode": "max"
        },
        "num_epochs": epochs,
        "validation_metric": "+f1-measure-overall",
        "patience": patience,
        "grad_norm": grad_norm,
        "use_amp": use_amp,
        "epoch_callbacks": [
            {
                "type": "optuna_pruner",
            },
        ],
    }
}
