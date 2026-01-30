# KernelOperator テストの設定

add_executable(kernel_oprator_test
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_oprator_test.cpp"
)

target_include_directories(kernel_oprator_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
)

target_link_libraries(kernel_oprator_test PRIVATE
    GTest::gtest
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET kernel_oprator_test PROPERTY CXX_STANDARD 20)
endif()

gtest_discover_tests(kernel_oprator_test)
