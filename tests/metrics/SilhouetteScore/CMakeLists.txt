# SilhouetteScore テストの設定

add_executable(silhouette_score_test
  "${CMAKE_CURRENT_SOURCE_DIR}/silhouette_score_test.cpp"
)

target_include_directories(silhouette_score_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
)

target_link_libraries(silhouette_score_test PRIVATE
  GTest::gtest
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET silhouette_score_test PROPERTY CXX_STANDARD 20)
endif()

gtest_discover_tests(silhouette_score_test)
