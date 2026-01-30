# ConstantKernel テストの設定

add_executable(constant_kernel_test
    "${CMAKE_CURRENT_SOURCE_DIR}/constant_kernel_test.cpp"
)

target_include_directories(constant_kernel_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
)

target_link_libraries(constant_kernel_test PRIVATE
    GTest::gtest
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET constant_kernel_test PROPERTY CXX_STANDARD 20)
endif()

gtest_discover_tests(constant_kernel_test)
