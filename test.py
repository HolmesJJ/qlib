import qlib

from qlib.utils import flatten_dict
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.constant import REG_CN

# region in [REG_CN, REG_US]
provider_uri = "~/.qlib/qlib_data/my_data"
qlib.init(provider_uri=provider_uri, region=REG_CN)
market = "csi300"
benchmark = "SH000300"

data_handler_config = {
    "start_time": "2018-01-01",
    "end_time": "2020-12-01",
    "fit_start_time": "2018-01-01",
    "fit_end_time": "2020-12-31",
    "instruments": market,
}

task = {
    "model": {
        "class": "AutoFormerModel",
        "module_path": "qlib.contrib.model.pytorch_autoformer",
        "kwargs": {
            "seed": 42,
            "learning_rate": 0.0421,
            "task_name": "long_term_forecast",
            "model_id": "train",
            "model": "Autoformer",
            "features": "MS",
            "data": "stock",
            "freq": "d",
            "seq_len": 96,
            "label_len": 48,
            "pred_len": 96,
            "seasonal_patterns": "Monthly",
            "inverse": False,
            "mask_rate": 0.25,
            "anomaly_ratio": 0.25,
            "top_k": 5,
            "num_kernels": 6,
            "enc_in": 159,
            "dec_in": 159,
            "c_out": 159,
            "d_model": 512,
            "n_heads": 8,
            "e_layers": 2,
            "d_layers": 1,
            "d_ff": 2048,
            "moving_avg": 25,
            "factor": 1,
            "distil": True,
            "dropout": 0.1,
            "embed": "timeF",
            "activation": "gelu",
            "output_attention": False,
            "channel_independence": 0,
            "num_workers": 10,
            "itr": 1,
            "train_epochs": 10,
            "batch_size": 32,
            "patience": 3,
            "des": "test",
            "lradj": "type1",
            "use_amp": False,
            "use_gpu": True,
            "gpu": 0,
            "use_multi_gpu": False,
            "devices": "0,1,2,3",
            "p_hidden_dims": [128, 128],
            "p_hidden_layers": 2,
            "optimizer": "adam",
            "metric": "loss"
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2018-01-01", "2018-12-31"),
                "valid": ("2019-01-01", "2019-12-31"),
                "test": ("2020-01-01", "2020-12-31"),
            },
        },
    },
}

# Model initialization
model = init_instance_by_config(task["model"])
dataset = init_instance_by_config(task["dataset"])

# Start Experience
with R.start(experiment_name="workflow"):
    # Train
    R.log_params(**flatten_dict(task))
    model.fit(dataset)
    # Prediction
    recorder = R.get_recorder()
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()
