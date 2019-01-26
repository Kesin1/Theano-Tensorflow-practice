hyperparams = {
    "CNN": {
        "out_channels": [20, 50],
        "filter_shapes": [(5, 5), (5, 5)],
        "hidden_layer_sizes": [500, 300],
    },
    "train": {
        "learning_rate": 10e-4,
        "reg": 10e-2,
        "decay": 0.99999,
        "momentum": 0.99,
        "epsilon": 10e-3,
        "batch_size": 30,
        "#validation_batch_size": 10000,
        "validation_batch_size": 30,
        "epochs": 3,
        "print_period": 10
    }
}
