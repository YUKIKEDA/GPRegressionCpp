# K-means テストの設定

add_executable(kmeans_test
  "${CMAKE_CURRENT_SOURCE_DIR}/kmeans_test.cpp"
)

target_include_directories(kmeans_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
)

target_link_libraries(kmeans_test PRIVATE
  GTest::gtest
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET kmeans_test PROPERTY CXX_STANDARD 20)
endif()

gtest_discover_tests(kmeans_test)
