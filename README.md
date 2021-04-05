# mlpack_random_forest_revamp
# This code is for demonstration purposes only.

"random_forest" directory contains the refactored code for `_main.cpp` files.

"wrapper_codes" directory contains the code for generated wrappers (for some languages).

"mlpack_mains" directory contains the different "mlpack_main" versions that are defined in
the respective "_main.cpp" files. For example: "random_forest_fit_main.cpp" includes "mlpack_main_fit.hpp"
because "mlpack_main_fit.hpp" has a "mlpackMainFit()" function declared along with other things.

Note: "mlpack_main_fit.hpp", "mlpack_main_predict.hpp" only differ in the name of functions declared inside
i.e. "mlpackMainFit()", "mlpackMainPredict()" respectively.

Also, take a look at the "CMakeLists.txt" inside "random_forest" directory, to know how the CMake macros are
called, and how is grouping done.
