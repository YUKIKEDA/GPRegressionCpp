# PCA テストの設定

add_executable(pca_test
  "${CMAKE_CURRENT_SOURCE_DIR}/pca_test.cpp"
)

target_include_directories(pca_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
)

# main を自前で定義しているため gtest のみリンク
target_link_libraries(pca_test PRIVATE
  GTest::gtest
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET pca_test PROPERTY CXX_STANDARD 20)
endif()

gtest_discover_tests(pca_test)
