/**
 * @file methods/random_forest/random_forest_main.cpp
 * @author Ryan Curtin
 *
 * A program to build and evaluate random forests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/random_forest/random_forest_model.hpp>
#include <mlpack/core/util/mlpack_main_predict.hpp>

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_NAME("Random forests prediction");

// Short description.
BINDING_SHORT_DESC(
    "Random Forest Prediction.");

// Long description.
BINDING_LONG_DESC(
    "Some description for the CLI.");

BINDING_EXAMPLE(
    "To use the model to classify points in " +
    PRINT_DATASET("test_set") + " ,while saving the "
    "predictions for each point to " + PRINT_DATASET("predictions") + ", one "
    "could call. ");

PARAM_MATRIX_IN("test", "Test dataset to produce predictions for.", "T");

PARAM_UROW_OUT("predictions", "Predicted classes for each point in the test "
    "set.", "p");

PARAM_MODEL_IN(RandomForestModel, "input_model", "Pre-trained random forest to "
    "use for classification.", "m");

static void mlpackMainPredict()
{
  arma::mat test = std::move(IO::GetParam<arma::mat>("test"));
  arma::Row<size_t> classes;

  RandomForestModel* rfModel = IO::GetParam<RandomForestModel*>("input_model");
  Timer::Start("rf_prediction");
  rfModel->rf.Classify(test, classes);
  Timer::Stop("rf_prediction");
  IO::GetParam<arma::Row<size_t>>("predictions") = classes;
}
