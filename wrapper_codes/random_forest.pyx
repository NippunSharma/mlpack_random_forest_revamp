cimport arma
cimport arma_numpy
from io cimport IO
from io cimport SetParam, SetParamPtr, SetParamWithInfo, GetParamPtr
from io cimport EnableVerbose, DisableVerbose, DisableBacktrace, ResetTimers, EnableTimers
from matrix_utils import to_matrix, to_matrix_with_info
from serialization cimport SerializeIn, SerializeOut

import numpy as np
cimport numpy as np

import pandas as pd

from libcpp.string cimport string
from libcpp cimport bool as cbool
from libcpp.vector cimport vector

from cython.operator import dereference

# Here we import each mlpackMain function and wrap it
cdef extern from "</home/nippun/Desktop/mlpack-gsoc/mlpack/src/mlpack/methods/random_forest/random_forest_fit_main.cpp>" nogil:
  cdef void mlpackMainFit() nogil except +RuntimeError

cdef extern from "</home/nippun/Desktop/mlpack-gsoc/mlpack/src/mlpack/methods/random_forest/random_forest_predict.cpp>" nogil:
  cdef void mlpackMainPredict() nogil except +RuntimeError

cdef extern from "</home/nippun/Desktop/mlpack-gsoc/mlpack/src/mlpack/methods/random_forest/random_forest_model.hpp>" nogil:
  cdef cppclass RandomForestModel:
    RandomForestModel() nogil
  

cdef class RandomForestModelType:
  cdef RandomForestModel* modelptr

  def __cinit__(self):
    self.modelptr = new RandomForestModel()

  def __dealloc__(self):
    del self.modelptr

  def __getstate__(self):
    return SerializeOut(self.modelptr, "RandomForestModel")

  def __setstate__(self, state):
    SerializeIn(self.modelptr, state, "RandomForestModel")

  def __reduce_ex__(self, version):
    return (self.__class__, (), self.__getstate__())

# This is the class that the user will interact with.
# "fit()" method checks for parameters and sets them as passed, then trains the model.
# "predict()" method takes the test dataset and provides the predictions.
class RandomForestPy:
  def __init__(check_input_matrices=False,
           	   copy_all_inputs=False,
               maximum_depth=None,
               minimum_gain_split=None,
               minimum_leaf_size=None,
               num_trees=None,
               print_training_accuracy=False,
               seed=None,
               subspace_dim=None,
               verbose=False)

    self.params = dict()
    self.params["check_input_matrices"] = check_input_matrices
    self.params["copy_all_inputs"] = copy_all_inputs
    self.params["input_model"] = input_model
    self.params["maximum_depth"] = maximum_depth
    self.params["minimum_leaf_size"] = minimum_gain_split
    self.params["num_trees"] = num_trees
    self.params["seed"] = seed
    self.params["subspace_dim"] = subspace_dim
    self.params["verbose"] = verbose
    self.model_after_fit = None
 
  def fit(self, X, y):
    ResetTimers()
    EnableTimers()
    DisableBacktrace()
    DisableVerbose()

    IO.RestoreSettings("Random forests training")
    if isinstance(self.params["copy_all_inputs"], bool):
      if self.params["copy_all_inputs"]:
        SetParam[cbool](<const string> 'copy_all_inputs', self.params["copy_all_inputs"])
        IO.SetPassed(<const string> 'copy_all_inputs')
    else:
      raise TypeError("'copy_all_inputs' must have type 'bool'!")

    # Detect if the parameter was passed; set if so.
    if isinstance(self.params["check_input_matrices"], bool):
      if self.params["check_input_matrices"] is not False:
        SetParam[cbool](<const string> 'check_input_matrices', self.params["check_input_matrices"])
        IO.SetPassed(<const string> 'check_input_matrices')
    else:
      raise TypeError("'check_input_matrices' must have type 'bool'!")

    # Detect if the parameter was passed; set if so.
    if self.params["maximum_depth"] is not None:
      if isinstance(self.params["maximum_depth"], int):
        SetParam[int](<const string> 'maximum_depth', self.params["maximum_depth"])
        IO.SetPassed(<const string> 'maximum_depth')
      else:
        raise TypeError("'maximum_depth' must have type 'int'!")

    # Detect if the parameter was passed; set if so.
    if self.params["minimum_gain_split"] is not None:
      if isinstance(self.params["minimum_gain_split"], float):
        SetParam[double](<const string> 'minimum_gain_split', self.params["minimum_gain_split"])
        IO.SetPassed(<const string> 'minimum_gain_split')
      else:
        raise TypeError("'minimum_gain_split' must have type 'float'!")

    # Detect if the parameter was passed; set if so.
    if self.params["minimum_leaf_size"] is not None:
      if isinstance(self.params["minimum_leaf_size"], int):
        SetParam[int](<const string> 'minimum_leaf_size', self.params["minimum_leaf_size"])
        IO.SetPassed(<const string> 'minimum_leaf_size')
      else:
        raise TypeError("'minimum_leaf_size' must have type 'int'!")

    # Detect if the parameter was passed; set if so.
    if self.params["num_trees"] is not None:
      if isinstance(self.params["num_trees"], int):
        SetParam[int](<const string> 'num_trees', self.params["num_trees"])
        IO.SetPassed(<const string> 'num_trees')
      else:
        raise TypeError("'num_trees' must have type 'int'!")

    # Detect if the parameter was passed; set if so.
    if isinstance(self.params["print_training_accuracy"], bool):
      if self.params["print_training_accuracy"] is not False:
        SetParam[cbool](<const string> 'print_training_accuracy', self.params["print_training_accuracy"])
        IO.SetPassed(<const string> 'print_training_accuracy')
    else:
      raise TypeError("'print_training_accuracy' must have type 'bool'!")

    # Detect if the parameter was passed; set if so.
    if self.params["seed"] is not None:
      if isinstance(self.params["seed"], int):
        SetParam[int](<const string> 'seed', seed)
        IO.SetPassed(<const string> 'seed')
      else:
        raise TypeError("'seed' must have type 'int'!")

    # Detect if the parameter was passed; set if so.
    if self.params["subspace_dim"] is not None:
      if isinstance(self.params["subspace_dim"], int):
        SetParam[int](<const string> 'subspace_dim', self.params["subspace_dim"])
        IO.SetPassed(<const string> 'subspace_dim')
      else:
        raise TypeError("'subspace_dim' must have type 'int'!")

    # Detect if the parameter was passed; set if so.
    if isinstance(self.params["verbose"], bool):
      if self.params["verbose"] is not False:
        SetParam[cbool](<const string> 'verbose', self.params["verbose"])
        IO.SetPassed(<const string> 'verbose')
        EnableVerbose()
    else:
      raise TypeError("'verbose' must have type 'bool'!")

    # Mark all output options as passed.
    IO.SetPassed(<const string> 'output_model')
    if not isinstance(self.params["check_input_matrices"], bool):
      raise TypeError("'check_input_matrices' must have type 'bool'!")

    if X is not None:
      training_tuple = to_matrix(X, dtype=np.double, copy=IO.HasParam('copy_all_inputs'))
      if len(training_tuple[0].shape) < 2:
        training_tuple[0].shape = (training_tuple[0].shape[0], 1)
      training_mat = arma_numpy.numpy_to_mat_d(training_tuple[0], training_tuple[1])
      SetParam[arma.Mat[double]](<const string> 'training', dereference(training_mat))
      IO.SetPassed(<const string> 'training')
      del training_mat

    if y is not None:
      labels_tuple = to_matrix(y, dtype=np.intp, copy=IO.HasParam('copy_all_inputs'))
      if len(labels_tuple[0].shape) > 1:
        if labels_tuple[0].shape[0] == 1 or labels_tuple[0].shape[1] == 1:
          labels_tuple[0].shape = (labels_tuple[0].size,)
      labels_mat = arma_numpy.numpy_to_row_s(labels_tuple[0], labels_tuple[1])
      SetParam[arma.Row[size_t]](<const string> 'labels', dereference(labels_mat))
      IO.SetPassed(<const string> 'labels')
      del labels_mat

    if IO.check_input_matrices:
      CheckInputMatrices()

    mlpackMainFit()

    self.model_after_fit = RandomForestModelType()
    (<RandomForestModelType?> self.model_after_fit).modelptr = GetParamPtr[RandomForestModel]('output_model')
    if input_model is not None:
      if (<RandomForestModelType> self.model_after_fit).modelptr == (<RandomForestModelType> input_model).modelptr:
        (<RandomForestModelType> self.model_after_fit).modelptr = <RandomForestModel*> 0
        self.model_after_fit = input_model

    IO.ClearSettings()

  def predict(self, X_test):
    ResetTimers()
    EnableTimers()
    DisableBacktrace()
    DisableVerbose()
    IO.RestoreSettings("Random forests prediction")

    if isinstance(self.params["copy_all_inputs"], bool):
      if self.params["copy_all_inputs"]:
        SetParam[cbool](<const string> 'copy_all_inputs', self.params["copy_all_inputs"])
        IO.SetPassed(<const string> 'copy_all_inputs')
    else:
      raise TypeError("'copy_all_inputs' must have type 'bool'!")

    # Detect if the parameter was passed; set if so.
    if isinstance(self.params["check_input_matrices"], bool):
      if self.params["check_input_matrices"] is not False:
        SetParam[cbool](<const string> 'check_input_matrices', self.params["check_input_matrices"])
        IO.SetPassed(<const string> 'check_input_matrices')
    else:
      raise TypeError("'check_input_matrices' must have type 'bool'!")

    # Detect if the parameter was passed; set if so.
    if self.model_after_fit is not None:
      try:
        SetParamPtr[RandomForestModel]('input_model', (<RandomForestModelType?> self.model_after_fit).modelptr, IO.HasParam('copy_all_inputs'))
      except TypeError as e:
        if type(self.model_after_fit).__name__ == 'RandomForestModelType':
          SetParamPtr[RandomForestModel]('input_model', (<RandomForestModelType> self.model_after_fit).modelptr, IO.HasParam('copy_all_inputs'))
        else:
          raise e
      IO.SetPassed(<const string> 'input_model')

    # Detect if the parameter was passed; set if so.
    if X_test is not None:
      test_tuple = to_matrix(X_test, dtype=np.double, copy=IO.HasParam('copy_all_inputs'))
      if len(test_tuple[0].shape) < 2:
        test_tuple[0].shape = (test_tuple[0].shape[0], 1)
      test_mat = arma_numpy.numpy_to_mat_d(test_tuple[0], test_tuple[1])
      SetParam[arma.Mat[double]](<const string> 'test', dereference(test_mat))
      IO.SetPassed(<const string> 'test')
      del test_mat

    # Detect if the parameter was passed; set if so.
    if isinstance(self.params["verbose"], bool):
      if self.params["verbose"] is not False:
        SetParam[cbool](<const string> 'verbose', self.params["verbose"])
        IO.SetPassed(<const string> 'verbose')
        EnableVerbose()
    else:
      raise TypeError("'verbose' must have type 'bool'!")

    # Mark all output options as passed.
    IO.SetPassed(<const string> 'predictions')
    if not isinstance(check_input_matrices, bool):
      raise TypeError("'check_input_matrices' must have type 'bool'!")

    if self.params["check_input_matrices"]:
      IO.CheckInputMatrices()
    # Call the mlpack program.

    mlpackMainPredict()

    predictions = arma_numpy.row_to_numpy_s(IO.GetParam[arma.Row[size_t]]('predictions'))
    IO.ClearSettings()
    return predictions
