import argparse
import datetime as dte
import os

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer
tf.experimental.output_all_intermediates(True)

name, output_folder, use_tensorflow_with_gpu =  'op', '/gpfs/data/fs71801/lkapral66/tft', 'yes'

print("Using output folder {}".format(output_folder))

config = ExperimentConfig(name, output_folder)
formatter = config.make_data_formatter()


expt_name=name
use_gpu=use_tensorflow_with_gpu
model_folder=os.path.join(config.model_folder, "op")
data_csv_path=config.data_csv_path
data_formatter=formatter
use_testing_mode=False



if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
    raise ValueError(
        "Data formatters should inherit from" +
        "AbstractDataFormatter! Type={}".format(type(data_formatter)))

# Tensorflow setup
default_keras_session = tf.compat.v1.keras.backend.get_session()

if use_gpu:
    tf_config = utils.get_default_tensorflow_config(tf_device="gpu")


else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

print("*** Training from defined parameters for {} ***".format(expt_name))

print("Loading & splitting data...")
raw_data = pd.read_parquet(data_csv_path)
train, valid, test = data_formatter.split_data(raw_data)
train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

# Sets up default params


param_ranges = ModelClass.get_hyperparm_choices()

fixed_params = data_formatter.get_experiment_params()
params = data_formatter.get_default_model_params()
params["model_folder"] = model_folder


# Sets up hyperparam manager
print("*** Loading hyperparm manager ***")
opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                   fixed_params, model_folder)

# Training -- one iteration only
print("*** Running calibration ***")
print("Params Selected:")
for k in params:
    print("{}: {}".format(k, params[k]))
    
    
    
tf.compat.v1.reset_default_graph()
tf.Graph().as_default()
tf.compat.v1.disable_eager_execution()

params = opt_manager.get_next_parameters()
model = ModelClass(params, use_cudnn=use_gpu)


#model.load(opt_manager.hyperparam_folder)
best_loss = np.Inf
stopper = 0
model.cache_batched_data(valid, "valid", num_samples=valid_samples)

for episode in range(1000):

    model.cache_batched_data(train, "train", num_samples=train_samples)

    model.fit(init_epochs = episode*10)

    val_loss = model.evaluate()
    
    model.save(model_folder+"/latest")
    if(val_loss < best_loss):
        opt_manager.update_score(params, val_loss, model)
        best_loss = val_loss
        model.save(model_folder+"/save")
        stopper = 0
    else:
        stopper += 1
        if(stopper > 50):
            break
    with open("log.txt", "w") as file1:
        # Writing data to a file
        file1.write(str(stopper)+' '+str(val_loss))