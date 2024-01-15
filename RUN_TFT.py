#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from
scratch.

Usage:
python3 script_train_fixed_params {expt_name} {output_folder}

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved


"""

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


def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         use_testing_mode=False):
    """Trains tft based on defined model params.

    Args:
      expt_name: Name of experiment
      use_gpu: Whether to run tensorflow with GPU operations
      model_folder: Folder path where models are serialized
      data_csv_path: Path to csv file containing data
      data_formatter: Dataset-specific data fromatter (see
        expt_settings.dataformatter.GenericDataFormatter)
      use_testing_mode: Uses a smaller models and data sizes for testing purposes
        only -- switch to False to use original default settings
    """

    num_repeats = 100

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
    raw_data = pd.read_csv(data_csv_path)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder

    # Parameter overrides for testing only! Small sizes used to speed up script.
    if use_testing_mode:
        print('*** Test_mode activated ***')
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    # Sets up hyperparam manager
    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)

    # Training -- one iteration only
    print("*** Running calibration ***")
    print("Params Selected:")
    for k in params:
        print("{}: {}".format(k, params[k]))

        
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:

        tf.keras.backend.set_session(sess)

        params = opt_manager.get_next_parameters()
        model = ModelClass(params, use_cudnn=use_gpu)
        model.load(model_folder)
        best_loss = np.Inf
        for _ in range(num_repeats):


            if not model.training_data_cached():
                model.cache_batched_data(train, "train", num_samples=train_samples)
                model.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf.global_variables_initializer())
            model.fit()

            val_loss = model.evaluate()

            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, model)
                best_loss = val_loss

            tf.keras.backend.set_session(default_keras_session)

    print("*** Running tests ***")
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)

        model.load(opt_manager.hyperparam_folder)

        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        print("Computing test loss")
        output_map = model.predict(test, return_targets=True)
        targets = data_formatter.format_predictions(output_map["targets"])
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[
                col for col in data.columns
                if col not in {"forecast_time", "identifier"}
            ]]

        p50_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p50_forecast),
            0.5)
        p90_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p90_forecast),
            0.9)

        tf.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    print()
    print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(
        p50_loss.mean(), p90_loss.mean()))


if __name__ == "__main__":
    def get_args():
        """Gets settings from command line."""

        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            default="volatility",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names)))
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for data download")
        parser.add_argument(
            "use_gpu",
            metavar="g",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="no",
            help="Whether to use gpu for training.")

        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder

        return args.expt_name, root_folder, args.use_gpu == "yes"


    name, output_folder, use_tensorflow_with_gpu =  'op', '/gpfs/data/fs71801/lkapral66/tft', 'yes'

    print("Using output folder {}".format(output_folder))

    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    # Customise inputs to main() for new datasets.
    main(
        expt_name=name,
        use_gpu=use_tensorflow_with_gpu,
        model_folder=os.path.join(config.model_folder, "fixed"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        use_testing_mode=False)  # Change to false to use original default params


























# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Main hyperparameter optimisation script.

Performs random search to optimize hyperparameters on a single machine. For new
datasets, inputs to the main(...) should be customised.
"""

import argparse
import datetime as dte
import os

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt

import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.tft_model.TemporalFusionTransformer
tf.experimental.output_all_intermediates(True)

def main(expt_name, use_gpu, restart_opt, model_folder, hyperparam_iterations,
         data_csv_path, data_formatter):
    """Runs main hyperparameter optimization routine.

  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    restart_opt: Whether to run hyperparameter optimization from scratch
    model_folder: Folder path where models are serialized
    hyperparam_iterations: Number of iterations of random search
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
  """

    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError(
            "Data formatters should inherit from" +
            "AbstractDataFormatter! Type={}".format(type(data_formatter)))

    default_keras_session = tf.compat.v1.keras.backend.get_session

    if use_gpu:
        tf_config = utils.get_default_tensorflow_config(tf_device="gpu")

    else:
        tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

    print("### Running hyperparameter optimization for {} ###".format(expt_name))
    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    # Sets up default params
    fixed_params = data_formatter.get_experiment_params()
    param_ranges = ModelClass.get_hyperparm_choices()
    fixed_params["model_folder"] = model_folder

    print("*** Loading hyperparm manager ***")
    opt_manager = HyperparamOptManager(param_ranges, fixed_params, model_folder)

    success = opt_manager.load_results()
    if success and not restart_opt:
        print("Loaded results from previous training")
    else:
        print("Creating new hyperparameter optimisation")
    opt_manager.clear()

    print("*** Running calibration ***")
    while len(opt_manager.results.columns) < hyperparam_iterations:
        print("# Running hyperparam optimisation {} of {} for {}".format(
            len(opt_manager.results.columns) + 1, hyperparam_iterations, "TFT"))

        tf.compat.v1.reset_default_graph()
        with tf.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:

            tf.compat.v1.keras.backend.set_session(sess)

            params = opt_manager.get_next_parameters()
            model = ModelClass(params, use_cudnn=use_gpu)

            if not model.training_data_cached():
                model.cache_batched_data(train, "train", num_samples=train_samples)
                model.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf.compat.v1.global_variables_initializer())
            model.fit()

            val_loss = model.evaluate()
            
            if np.allclose(val_loss, 0.) or np.isnan(val_loss):
                # Set all invalid losses to infintiy.
                # N.b. val_loss only becomes 0. when the weights are nan.
                print("Skipping bad configuration....")
                val_loss = np.inf

            opt_manager.update_score(params, val_loss, model)

            tf.compat.v1.keras.backend.set_session(default_keras_session)



    print("*** Running tests ***")
    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)

        model.load(opt_manager.hyperparam_folder)

        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        print("Computing test loss")
        output_map = model.predict(test, return_targets=True)
        targets = data_formatter.format_predictions(output_map["targets"])
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[col for col in data.columns if col not in {"forecast_time", "identifier"}]]

        p50_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p50_forecast),
            0.5)
        p90_loss = utils.numpy_normalised_quantile_loss(
            extract_numerical_data(targets), extract_numerical_data(p90_forecast),
            0.9)

        tf.compat.v1.keras.backend.set_session(default_keras_session)

    print("Hyperparam optimisation completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    print("Params:")

    for k in best_params:
        print(k, " = ", best_params[k])
    print()
    print("Normalised Quantile Loss for Test Data: P50={}, P90={}".format(p50_loss.mean(), p90_loss.mean()))


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        experiment_names = ExperimentConfig.default_experiments

        parser = argparse.ArgumentParser(description="Data download configs")
        parser.add_argument(
            "expt_name",
            metavar="e",
            type=str,
            nargs="?",
            default="volatility",
            choices=experiment_names,
            help="Experiment Name. Default={}".format(",".join(experiment_names)))
        parser.add_argument(
            "output_folder",
            metavar="f",
            type=str,
            nargs="?",
            default=".",
            help="Path to folder for data download")
        parser.add_argument(
            "use_gpu",
            metavar="g",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="no",
            help="Whether to use gpu for training.")
        parser.add_argument(
            "restart_hyperparam_opt",
            metavar="o",
            type=str,
            nargs="?",
            choices=["yes", "no"],
            default="yes",
            help="Whether to re-run hyperparameter optimisation from scratch.")

        args = parser.parse_known_args()[0]

        root_folder = None if args.output_folder == "." else args.output_folder

        return args.expt_name, root_folder, args.use_gpu == "yes", \
            args.restart_hyperparam_opt == "yes"

    # Load settings for default experiments
    name, folder, use_tensorflow_with_gpu, restart = 'op', '/home/fs71801/lkapral66/Transformer/tft_tf2/tft_outputs', 'yes', 'yes'

    print("Using output folder {}".format(folder))

    config = ExperimentConfig(name, folder)
    formatter = config.make_data_formatter()

    # Customise inputs to main() for new datasets.
    main(
      expt_name=name,
      use_gpu=use_tensorflow_with_gpu,
      restart_opt=restart,
      model_folder=os.path.join(config.model_folder, "main"),
      hyperparam_iterations=config.hyperparam_iterations,
      data_csv_path=config.data_csv_path,
      data_formatter=formatter)


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors
def plot_patient_value_points(df, patient_nr, model, columns, label_names, scale_range,  window_length=10, prediction_points=[]):

    figure(figsize=(8, 6), dpi=300)
    indices = df['rand_id'].unique()
    pat_id = indices[patient_nr]
    
    df_pat = df[df['rand_id']==pat_id]
    
    print("Predicting")
    output_map = model.predict(df_pat, return_targets=True)
    targets = data_formatter.format_predictions(output_map["targets"])
    
    print(targets)



    x_axis = np.arange(0,len(df_pat_plot)-(window_length-1),1)

    legend = columns
    for column in columns:
        plt.plot(x_axis/4, df_pat_plot[column][0:len(df_pat)-(window_length-1)].apply(lambda x: rescale(x, scale, column)))
    '''
    for i, label in enumerate(label_names):
        plt.plot(x_axis/4, df_pat_plot[label][0:len(df_pat)-(window_length-1)].apply(lambda x: rescale(x, scale, column)))
        legend.append(label)
    '''
    for point in prediction_points:
        print(predicted_values[round(point/4),:])
        plt.plot(x_axis[round(point/4):round(point/4)+predicted_values.shape[1]], (rescale(predicted_values[round(point),:], scale, column)))
    legend.append('AI')
    #plt.ylim(0, 250)
    plt.xlabel('Time in [m]')    
    plt.ylabel('Mean blood pressure in [mmHg]')
    
    
    plt.legend(legend)

    plt.show()


# In[ ]:


#_(dropoutrate)_(earlystopping_patience)_(hiddenlayer_size)_[??]_InputSize_[0, 1, 2, 3]_[0, 1]_(LR)_(max_gradient_norm)_(minibatch_size


