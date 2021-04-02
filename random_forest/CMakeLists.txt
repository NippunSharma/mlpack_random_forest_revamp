# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  bootstrap.hpp
  random_forest.hpp
  random_forest_impl.hpp
  random_forest_model.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(random_forest_fit)
add_python_binding(random_forest_fit)
add_julia_binding(random_forest_fit)
add_go_binding(random_forest_fit)
add_r_binding(random_forest_fit)
add_markdown_docs(random_forest_fit "cli;python;julia;go;r" "classification")

add_cli_executable(random_forest_predict)
add_python_binding(random_forest_predict)
add_julia_binding(random_forest_predict)
add_go_binding(random_forest_predict)
add_r_binding(random_forest_predict)
add_markdown_docs(random_forest_predict "cli;python;julia;go;r" "classification")
