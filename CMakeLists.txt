
cmake_minimum_required(VERSION 3.6)

project(gann VERSION 1.0.0 DESCRIPTION "g artifical neuron networks")

set(GFC				"/Users/christian/export/local/works/doublegsoft.open/gfc/03.Development/gfc")
set(GFC_INC		"${GFC}/include")
set(GFC_LIB		"${GFC}/build/darwin")

set(GNUM			"/Users/christian/export/local/works/doublegsoft.open/gnum/03.Development/gnum")
set(GNUM_INC	"${GNUM}/include")
set(GNUM_LIB	"${GNUM}/build/darwin")

include_directories(include ${GFC_INC})

add_library(gann STATIC
  src/gann-mlp.c
  src/gann.c
)

target_link_libraries(gann PRIVATE ${GNUM_LIB}/libgnum.a ${GNUM_LIB}/libgfc.a)

add_executable(gann-mlp-test-iris
  src/gann-mlp.c
  src/gann.c
  test/gann-mlp-test-iris.c
)

add_executable(gann-w2v-test-analogy
  src/gann-w2v.c
  src/gann.c
  test/gann-w2v-test-analogy.c
)

target_link_libraries(gann-w2v-test-analogy PRIVATE ${GFC_LIB}/libgfc.a)