/**
 * @file constant_kernel_test.cpp
 * @brief 定数カーネル (ConstantKernel) の GoogleTest テスト
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "kernels/ConstantKernel/constant_kernel.hpp"
#include "kernels/kernel.hpp"

using namespace gprcpp::kernels;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
  const double kTol = 1e-10;
}

// ---------------------------------------------------------------------------
// コンストラクタ・基本性質
// ---------------------------------------------------------------------------

// デフォルトコンストラクタで、ハイパーパラメータ数が1・定常・theta=log(1)=0 となること
TEST(ConstantKernelTest, DefaultConstructor)
{
  ConstantKernel kernel;
  EXPECT_EQ(kernel.num_hyperparameters(), 1);
  EXPECT_TRUE(kernel.is_stationary());
  // デフォルト値 1.0 → get_hyperparameters() は log(1) = 0
  VectorXd theta = kernel.get_hyperparameters();
  ASSERT_EQ(theta.size(), 1);
  EXPECT_NEAR(theta(0), 0.0, kTol);
}

// 定数値を指定したコンストラクタで、get_hyperparameters() が log(定数値) を返すこと
TEST(ConstantKernelTest, ConstructorWithValue)
{
  ConstantKernel kernel(2.5);
  VectorXd theta = kernel.get_hyperparameters();
  ASSERT_EQ(theta.size(), 1);
  EXPECT_NEAR(theta(0), std::log(2.5), kTol);
}

// 下限・上限を指定したコンストラクタで、ハイパーパラメータが正しく設定されること
TEST(ConstantKernelTest, ConstructorWithBounds)
{
  ConstantKernel kernel(3.0, 1e-6, 1e4);
  VectorXd theta = kernel.get_hyperparameters();
  EXPECT_NEAR(theta(0), std::log(3.0), kTol);
}

// ---------------------------------------------------------------------------
// カーネル行列 operator()
// ---------------------------------------------------------------------------

// K(X, X) が正方行列で、全要素が定数値であること
TEST(ConstantKernelTest, KernelMatrixSameInput)
{
  ConstantKernel kernel(2.0);
  MatrixXd X(3, 2);
  X << 0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0;

  MatrixXd K = kernel(X, X);
  EXPECT_EQ(K.rows(), 3);
  EXPECT_EQ(K.cols(), 3);
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      EXPECT_NEAR(K(i, j), 2.0, kTol) << "i=" << i << " j=" << j;
    }
  }
}

// K(X1, X2) の形状が (X1の行数, X2の行数) で、全要素が定数値であること
TEST(ConstantKernelTest, KernelMatrixTwoInputs)
{
  ConstantKernel kernel(5.0);
  MatrixXd X1(2, 1);
  X1 << 1.0, 2.0;
  MatrixXd X2(4, 1);
  X2 << 0.0, 1.0, 2.0, 3.0;

  MatrixXd K = kernel(X1, X2);
  EXPECT_EQ(K.rows(), 2);
  EXPECT_EQ(K.cols(), 4);
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      EXPECT_NEAR(K(i, j), 5.0, kTol);
    }
  }
}

// 第2引数を省略した kernel(X) が K(X, X) と同様に 1x1 の場合は定数になること
TEST(ConstantKernelTest, KernelMatrixSingleArgument)
{
  ConstantKernel kernel(1.0);
  MatrixXd X(1, 3);
  X << 1.0, 2.0, 3.0;

  MatrixXd K = kernel(X);
  EXPECT_EQ(K.rows(), 1);
  EXPECT_EQ(K.cols(), 1);
  EXPECT_NEAR(K(0, 0), 1.0, kTol);
}

// カーネル行列の値が入力 X1, X2 の内容に依存せず、定数値のみで決まること
TEST(ConstantKernelTest, KernelMatrixIndependentOfInputValues)
{
  ConstantKernel kernel(7.0);
  MatrixXd A(2, 2);
  A << 0.0, 0.0, 100.0, -100.0;
  MatrixXd B(2, 2);
  B << 1.0, 2.0, 3.0, 4.0;

  MatrixXd K = kernel(A, B);
  EXPECT_EQ(K.rows(), 2);
  EXPECT_EQ(K.cols(), 2);
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 2; ++j)
    {
      EXPECT_NEAR(K(i, j), 7.0, kTol);
    }
  }
}

// ---------------------------------------------------------------------------
// 対角 diag()
// ---------------------------------------------------------------------------

// diag(X) の長さが X の行数で、全要素が定数値であること
TEST(ConstantKernelTest, Diag)
{
  ConstantKernel kernel(3.0);
  MatrixXd X(4, 2);
  X << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;

  VectorXd d = kernel.diag(X);
  EXPECT_EQ(d.size(), 4);
  for (int i = 0; i < 4; ++i)
  {
    EXPECT_NEAR(d(i), 3.0, kTol);
  }
}

// diag(X) の各要素が K(X, X) の対角成分と一致すること
TEST(ConstantKernelTest, DiagConsistentWithKernelMatrix)
{
  ConstantKernel kernel(2.5);
  MatrixXd X(3, 1);
  X << 1.0, 2.0, 3.0;

  VectorXd d = kernel.diag(X);
  MatrixXd K = kernel(X, X);
  for (int i = 0; i < 3; ++i)
  {
    EXPECT_NEAR(d(i), K(i, i), kTol);
  }
}

// ---------------------------------------------------------------------------
// ハイパーパラメータ (対数スケール)
// ---------------------------------------------------------------------------

// get_hyperparameters() で取得した theta を set_hyperparameters() に渡しても値が変わらないこと
TEST(ConstantKernelTest, GetSetHyperparametersRoundtrip)
{
  ConstantKernel kernel(4.0);
  VectorXd theta = kernel.get_hyperparameters();
  kernel.set_hyperparameters(theta);
  VectorXd theta2 = kernel.get_hyperparameters();
  EXPECT_EQ(theta.size(), theta2.size());
  EXPECT_NEAR(theta(0), theta2(0), kTol);
}

// set_hyperparameters() でハイパーパラメータを設定した後、カーネル行列の値が変わること
TEST(ConstantKernelTest, SetHyperparametersChangesKernelValue)
{
  ConstantKernel kernel(1.0);
  VectorXd theta(1);
  theta(0) = std::log(10.0);
  kernel.set_hyperparameters(theta);

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = kernel(X);
  EXPECT_NEAR(K(0, 0), 10.0, kTol);
}

// ---------------------------------------------------------------------------
// clone / to_string
// ---------------------------------------------------------------------------

// clone() が元のカーネルと同じ定数値でカーネル行列を返す独立したインスタンスを返すこと
TEST(ConstantKernelTest, Clone)
{
  ConstantKernel kernel(6.0);
  std::shared_ptr<Kernel> cloned = kernel.clone();

  ASSERT_NE(cloned.get(), nullptr);
  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K_orig = kernel(X);
  MatrixXd K_clone = (*cloned)(X);
  EXPECT_NEAR(K_orig(0, 0), K_clone(0, 0), kTol);
  EXPECT_NEAR(K_clone(0, 0), 6.0, kTol);
}

// to_string() が空でなく、末尾が "^2" の形式であること
TEST(ConstantKernelTest, ToString)
{
  ConstantKernel kernel(2.0);
  std::string s = kernel.to_string();
  EXPECT_FALSE(s.empty());
  EXPECT_TRUE(s.size() >= 3u && s.substr(s.size() - 2) == "^2");
}

// ---------------------------------------------------------------------------
// main (GTest 実行用)
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
