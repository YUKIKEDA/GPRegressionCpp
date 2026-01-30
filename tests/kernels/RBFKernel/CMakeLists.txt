# RBFKernel テストの設定

add_executable(rbf_kernel_test
    "${CMAKE_CURRENT_SOURCE_DIR}/rbf_kernel_test.cpp"
)

target_include_directories(rbf_kernel_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
)

target_link_libraries(rbf_kernel_test PRIVATE
    GTest::gtest
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET rbf_kernel_test PROPERTY CXX_STANDARD 20)
endif()

gtest_discover_tests(rbf_kernel_test)
