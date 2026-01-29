# Dual Annealingテストの設定

# Dual Annealingテスト
add_executable(dual_annealing_test 
    "${CMAKE_CURRENT_SOURCE_DIR}/dual_annealing_test.cpp"
)

target_include_directories(dual_annealing_test PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../include"
)

# main関数がテストファイルに定義されているため、gtest_mainではなくgtestのみをリンク
target_link_libraries(dual_annealing_test PRIVATE
    GTest::gtest
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET dual_annealing_test PROPERTY CXX_STANDARD 20)
endif()

# Google Testの個別テストケースを自動検出して登録
# これにより、TestAllCases、TestLowDimensionalCasesなどが個別に表示される
gtest_discover_tests(dual_annealing_test)
