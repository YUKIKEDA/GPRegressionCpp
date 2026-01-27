# テストの設定

# GTestの検索
find_package(GTest CONFIG REQUIRED)

# GoogleTestモジュールの読み込み（gtest_discover_testsを使用するため）
include(GoogleTest)

# Nelder-Meadテスト
add_executable(nelder_mead_test 
    "optimize/NelderMead/nelder_mead_test.cpp"
)

target_include_directories(nelder_mead_test PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/../include"
)

# main関数がテストファイルに定義されているため、gtest_mainではなくgtestのみをリンク
target_link_libraries(nelder_mead_test PRIVATE
    GTest::gtest
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET nelder_mead_test PROPERTY CXX_STANDARD 20)
endif()

# Google Testの個別テストケースを自動検出して登録
# これにより、TestAllCases、TestLowDimensionalCasesなどが個別に表示される
gtest_discover_tests(nelder_mead_test)
