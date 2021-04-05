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
#include <mlpack/methods/decision_tree/random_dimension_select.hpp>
#include <mlpack/core/util/mlpack_main_fit.hpp>

using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::util;
using namespace std;

// Program Name.
BINDING_NAME("Random forests training");

// Short description for CLI.
BINDING_SHORT_DESC(
    "An implementation of the standard random forest algorithm by Leo Breiman "
    "for classification. Given labeled data, a random forest can be trained "
    "and saved for future use.");

// Short Description for wrapper in other languages.
// This is commented because BINDING_SHORT_DESC_WRAPPER is not yet implemented.
// Note that here I have not written wether the model can be saved or used for
// future use because all this will be present inside the documentation of the
// "fit" and "predict" methods respectively.
/*
BINDING_SHORT_DESC_WRAPPER(
    "An implementation of the standard random forest algorithm by Leo Breiman "
    "for classification.");
)
*/

// Long description.
BINDING_LONG_DESC(
    "Some long description for this functionality in CLI.");

// Long Description for wrapper in other languages.
// This is commented because BINDING_LONG_DESC_WRAPPER is not yet implemented.
/*
BINDING_LONG_DESC_WRAPPER(
    "Some long description for this functionality in other languages.");
)
*/

// Example for use in CLI.
BINDING_EXAMPLE(
    "For example, to train a random forest with a minimum leaf size of 20 "
    "using 10 trees on the dataset contained in " + PRINT_DATASET("data") +
    "with labels " + PRINT_DATASET("labels") + ", saving the output random "
    "forest to " + PRINT_MODEL("rf_model") + " and printing the training "
    "error, one could call"
    "\n\n" +
    PRINT_CALL("random_forest", "training", "data", "labels", "labels",
        "minimum_leaf_size", 20, "num_trees", 10, "output_model", "rf_model",
        "print_training_accuracy", true));

// Example for use of wrapper in other languages.
// This is commented because BINDING_EXAMPLE_WRAPPER is not yet implemented.
// Examples will change based on which language is used, for this 
// we can make a modified version of the PRINT_CALL macro.
/*
BINDING_EXAMPLE_WRAPPER(
    "Example for calling this through other languages.");
)
*/

PARAM_MATRIX_IN("training", "Training dataset.", "t");
PARAM_UROW_IN("labels", "Labels for training dataset.", "l");

PARAM_FLAG("print_training_accuracy", "If set, then the accuracy of the model "
    "on the training set will be predicted (verbose must also be specified).",
    "a");

PARAM_INT_IN("num_trees", "Number of trees in the random forest.", "N", 10);
PARAM_INT_IN("minimum_leaf_size", "Minimum number of points in each leaf "
    "node.", "n", 1);
PARAM_INT_IN("maximum_depth", "Maximum depth of the tree (0 means no limit).",
    "D", 0);

PARAM_DOUBLE_IN("minimum_gain_split", "Minimum gain needed to make a split "
    "when building a tree.", "g", 0);
PARAM_INT_IN("subspace_dim", "Dimensionality of random subspace to use for "
    "each split.  '0' will autoselect the square root of data dimensionality.",
    "d", 0);

PARAM_INT_IN("seed", "Random seed.  If 0, 'std::time(NULL)' is used.", "s", 0);


PARAM_MODEL_IN(RandomForestModel, "input_model", "Pre-trained random forest to "
    "use for classification.", "m");
PARAM_MODEL_OUT(RandomForestModel, "output_model", "Model to save trained "
    "random forest to.", "M");

static void mlpackMainFit()
{
  if (IO::GetParam<int>("seed") != 0)
    math::RandomSeed((size_t) IO::GetParam<int>("seed"));
  else
    math::RandomSeed((size_t) std::time(NULL));

  arma::mat data = std::move(IO::GetParam<arma::mat>("training"));
  arma::Row<size_t> labels = std::move(IO::GetParam<arma::Row<size_t>>("labels"));

  const size_t numTrees = (size_t) IO::GetParam<int>("num_trees");
  const size_t minimumLeafSize =
    (size_t) IO::GetParam<int>("minimum_leaf_size");
  const size_t maxDepth = (size_t) IO::GetParam<int>("maximum_depth");
  const double minimumGainSplit = IO::GetParam<double>("minimum_gain_split");
  const size_t randomDims = (IO::GetParam<int>("subspace_dim") == 0) ?
    (size_t) std::sqrt(data.n_rows) :
    (size_t) IO::GetParam<int>("subspace_dim");
  MultipleRandomDimensionSelect mrds(randomDims);

  const size_t numClasses = arma::max(labels) + 1;

  RandomForestModel* rfModel;
  if(IO::HasParam("input_model"))
    rfModel = IO::GetParam<RandomForestModel*>("input_model");
  else
    rfModel = new RandomForestModel();

  Timer::Start("rf_training");
  rfModel->rf.Train(data, labels, numClasses, numTrees, minimumLeafSize,
        minimumGainSplit, maxDepth, mrds);
  Timer::Stop("rf_training");

  // Did we want training accuracy?
  if (IO::HasParam("print_training_accuracy"))
  {
    Timer::Start("rf_prediction");
    arma::Row<size_t> predictions;
    rfModel->rf.Classify(data, predictions);

    const size_t correct = arma::accu(predictions == labels);

    Log::Info << correct << " of " << labels.n_elem << " correct on training"
        << " set (" << (double(correct) / double(labels.n_elem) * 100) << ")."
        << endl;
    Timer::Stop("rf_prediction");
  }

  IO::GetParam<RandomForestModel*>("output_model") = rfModel;
}
