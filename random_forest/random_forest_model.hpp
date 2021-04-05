#ifndef MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_MODEL_HPP
#define MLPACK_METHODS_RANDOM_FOREST_RANDOM_FOREST_MODEL_HPP
#include <mlpack/methods/random_forest/random_forest.hpp>
using namespace mlpack::tree;

/**
 * This is the class that we will serialize.  It is a pretty simple wrapper
 * around DecisionTree<>.  In order to support categoricals, it will need to
 * also hold and serialize a DatasetInfo.
 */
class RandomForestModel
{
 public:
  // The tree itself, left public for direct access by this program.
  RandomForest<> rf;

  // Create the model.
  RandomForestModel() { /* Nothing to do. */ }

  // Serialize the model.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */)
  {
    ar(CEREAL_NVP(rf));
  }
};

#endif
