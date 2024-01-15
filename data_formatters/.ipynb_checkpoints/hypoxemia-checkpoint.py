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
"""Custom formatting functions for Favorita dataset.

Defines dataset specific column definitions and data transformations.
"""

import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing
import numpy as np

DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class OPFormatter(data_formatters.base.GenericDataFormatter):
  """Defines and formats data for the Favorita dataset.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      
      
      ('rand_id', DataTypes.REAL_VALUED, InputTypes.ID),
      ('case_time', DataTypes.REAL_VALUED, InputTypes.TIME),            
      ('case_time', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('age', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
      ('gender', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('asa_score', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('urgency', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('surgery_type', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
      ('hf', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('pulse', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('spo2', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('etco2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('systolicbp', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('diastolicbp', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('meanbp', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('spo2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('invasivebp', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      ('insevo', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('exsevo', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('indes', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('exdes', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('berodual', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('cisatracurium', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('esketamin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('etomidat', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('fentanyl', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('midazolam', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('noradrenalin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('phenylephrin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('piritramid', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('propofol', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('remifentanil', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('rocuronium', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('succinylcholin', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('sufentanil', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('compliance', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('fio2', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('peep', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('plateau', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('pmax', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ppeak', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('pmean', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('resistance', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ventfreq', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('ventmode', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      ('vt', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('dobutamin_perfusor', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),      
      ('epinephrin_perfusor', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('levosimendan_perfusor', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('noradrenalin_perfusor', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('phenylephrin_perfusor', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('propofol_perfusor', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('remifentanil_perfusor', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('sufentanil_perfusor', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('vasopressin_perfusor', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
      ('phase', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT)
      
      #('hypotension60', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      #('MAP_period_id60', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      #('hypotension55', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      #('MAP_period_id55', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      #('hypotension65', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      #('MAP_period_id65', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      #('spo2_below_threshold', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      #('spo2_period_id', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
      #('hypoxemia', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT)     
  ]
                       


  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None

  def split_data(self, df):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')

    df = df.fillna(0)
    unique_case_id = df['rand_id'].drop_duplicates().tolist()
    ind_train = int(len(unique_case_id )*(0.7))
    ind_valid = int(len(unique_case_id )*(0.9))
    split_index_train = unique_case_id[ind_train]
    split_index_valid = unique_case_id[ind_valid] 
    first_entry_index = df.loc[df['rand_id'] == split_index_train].index[0]
    second_entry_index= df.loc[df['rand_id'] == split_index_valid].index[0]
   
    train = df.iloc[:first_entry_index]
    valid = df.iloc[first_entry_index:second_entry_index]
    test = df.iloc[second_entry_index:]
    self.set_scalers(train)

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID,
                                                   column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET,
                                                       column_definitions)

    # Extract identifiers in case required
    self.identifiers = list(df[id_column].unique())

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    data = df[real_inputs].values
    self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data)
    self._target_scaler = sklearn.preprocessing.StandardScaler().fit(
        df[[target_column]].values)  # used for predictions

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(
          srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """
    output = df.copy()

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    column_definitions = self.get_column_definition()

    real_inputs = utils.extract_cols_from_data_type(
        DataTypes.REAL_VALUED, column_definitions,
        {InputTypes.ID, InputTypes.TIME})
    
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    # Format real inputs
    output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

    # Format categorical inputs
    for col in categorical_inputs:
      string_df = df[col].apply(str)
      output[col] = self._cat_scalers[col].transform(string_df)

    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """
    
    output = predictions.copy()

    column_names = predictions.columns

    for col in column_names:
      if col not in {'forecast_time', 'identifier'}:
        output[col] = self._target_scaler.inverse_transform(predictions[[col]].values)

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 60,
        'num_encoder_steps': 32,
        'num_epochs': 10,
        'early_stopping_patience': 10,
        'multiprocessing_workers': 64
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.3,
        'hidden_layer_size': 240,
        'learning_rate': 0.0003,
        'minibatch_size': 128,
        'max_gradient_norm': 100.,
        'num_heads': 16,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    return 5000000, 1000000