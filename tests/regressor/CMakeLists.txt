# ガウス過程回帰テストの設定

add_executable(gp_regression_test
    "${CMAKE_CURRENT_SOURCE_DIR}/gp_regression_test.cpp"
)

target_include_directories(gp_regression_test PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(gp_regression_test PRIVATE
    GTest::gtest
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET gp_regression_test PROPERTY CXX_STANDARD 20)
endif()

gtest_discover_tests(gp_regression_test)
