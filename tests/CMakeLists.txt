# テストの設定

# GTestの検索
find_package(GTest CONFIG REQUIRED)

# GoogleTestモジュールの読み込み（gtest_discover_testsを使用するため）
include(GoogleTest)

# サブディレクトリのテストを追加
add_subdirectory(decomposition/PCA)
add_subdirectory(kernels/ConstantKernel)
add_subdirectory(kernels/KernelOperator)
add_subdirectory(kernels/RBFKernel)
add_subdirectory(kernels/WhiteNoiseKernel)
add_subdirectory(optimize/NelderMead)
add_subdirectory(optimize/DualAnnealing)
add_subdirectory(regressor)
