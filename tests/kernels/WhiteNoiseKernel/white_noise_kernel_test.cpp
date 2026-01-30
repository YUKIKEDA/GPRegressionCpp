/**
 * @file white_noise_kernel_test.cpp
 * @brief ホワイトノイズカーネル (WhiteKernel) の GoogleTest テスト
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "kernels/WhiteNoiseKernel/white_noise_kernel.hpp"
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
TEST(WhiteKernelTest, DefaultConstructor)
{
  WhiteKernel kernel;
  EXPECT_EQ(kernel.num_hyperparameters(), 1);
  EXPECT_TRUE(kernel.is_stationary());
  // デフォルト値 1.0 → get_hyperparameters() は log(1) = 0
  VectorXd theta = kernel.get_hyperparameters();
  ASSERT_EQ(theta.size(), 1);
  EXPECT_NEAR(theta(0), 0.0, kTol);
}

// ノイズレベルを指定したコンストラクタで、get_hyperparameters() が log(ノイズレベル) を返すこと
TEST(WhiteKernelTest, ConstructorWithNoiseLevel)
{
  WhiteKernel kernel(2.5);
  VectorXd theta = kernel.get_hyperparameters();
  ASSERT_EQ(theta.size(), 1);
  EXPECT_NEAR(theta(0), std::log(2.5), kTol);
}

// 下限・上限を指定したコンストラクタで、ハイパーパラメータが正しく設定されること
TEST(WhiteKernelTest, ConstructorWithBounds)
{
  WhiteKernel kernel(3.0, 1e-6, 1e4);
  VectorXd theta = kernel.get_hyperparameters();
  EXPECT_NEAR(theta(0), std::log(3.0), kTol);
}

// ---------------------------------------------------------------------------
// カーネル行列 operator() - 自己共分散 (x2 が空)
// ---------------------------------------------------------------------------

// K(X) が単位行列 × noise_level であること
TEST(WhiteKernelTest, KernelMatrixSelfCovariance)
{
  WhiteKernel kernel(2.0);
  MatrixXd X(3, 2);
  X << 0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0;

  MatrixXd K = kernel(X);
  EXPECT_EQ(K.rows(), 3);
  EXPECT_EQ(K.cols(), 3);
  // 対角成分は noise_level
  for (int i = 0; i < 3; ++i)
  {
    EXPECT_NEAR(K(i, i), 2.0, kTol) << "i=" << i;
  }
  // 非対角成分は 0
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      if (i != j)
      {
        EXPECT_NEAR(K(i, j), 0.0, kTol) << "i=" << i << " j=" << j;
      }
    }
  }
}

// 1x1 の場合も対角成分が noise_level であること
TEST(WhiteKernelTest, KernelMatrixSelfCovarianceSinglePoint)
{
  WhiteKernel kernel(5.0);
  MatrixXd X(1, 2);
  X << 1.0, 2.0;

  MatrixXd K = kernel(X);
  EXPECT_EQ(K.rows(), 1);
  EXPECT_EQ(K.cols(), 1);
  EXPECT_NEAR(K(0, 0), 5.0, kTol);
}

// ---------------------------------------------------------------------------
// カーネル行列 operator() - クロス共分散 (x2 が非空)
// ---------------------------------------------------------------------------

// K(X1, X2) が全要素 0 のゼロ行列であること
TEST(WhiteKernelTest, KernelMatrixCrossCovariance)
{
  WhiteKernel kernel(5.0);
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
      EXPECT_NEAR(K(i, j), 0.0, kTol) << "i=" << i << " j=" << j;
    }
  }
}

// 同じ入力を X1, X2 として渡してもクロス共分散はゼロであること
TEST(WhiteKernelTest, KernelMatrixCrossCovarianceSameInput)
{
  WhiteKernel kernel(3.0);
  MatrixXd X(3, 2);
  X << 0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0;

  MatrixXd K = kernel(X, X);
  EXPECT_EQ(K.rows(), 3);
  EXPECT_EQ(K.cols(), 3);
  // クロス共分散では全要素が 0
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      EXPECT_NEAR(K(i, j), 0.0, kTol) << "i=" << i << " j=" << j;
    }
  }
}

// ---------------------------------------------------------------------------
// 対角 diag()
// ---------------------------------------------------------------------------

// diag(X) の長さが X の行数で、全要素が noise_level であること
TEST(WhiteKernelTest, Diag)
{
  WhiteKernel kernel(3.0);
  MatrixXd X(4, 2);
  X << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;

  VectorXd d = kernel.diag(X);
  EXPECT_EQ(d.size(), 4);
  for (int i = 0; i < 4; ++i)
  {
    EXPECT_NEAR(d(i), 3.0, kTol);
  }
}

// diag(X) の各要素が K(X) (自己共分散) の対角成分と一致すること
TEST(WhiteKernelTest, DiagConsistentWithKernelMatrix)
{
  WhiteKernel kernel(2.5);
  MatrixXd X(3, 1);
  X << 1.0, 2.0, 3.0;

  VectorXd d = kernel.diag(X);
  MatrixXd K = kernel(X);
  for (int i = 0; i < 3; ++i)
  {
    EXPECT_NEAR(d(i), K(i, i), kTol);
  }
}

// ---------------------------------------------------------------------------
// ハイパーパラメータ (対数スケール)
// ---------------------------------------------------------------------------

// get_hyperparameters() で取得した theta を set_hyperparameters() に渡しても値が変わらないこと
TEST(WhiteKernelTest, GetSetHyperparametersRoundtrip)
{
  WhiteKernel kernel(4.0);
  VectorXd theta = kernel.get_hyperparameters();
  kernel.set_hyperparameters(theta);
  VectorXd theta2 = kernel.get_hyperparameters();
  EXPECT_EQ(theta.size(), theta2.size());
  EXPECT_NEAR(theta(0), theta2(0), kTol);
}

// set_hyperparameters() でハイパーパラメータを設定した後、カーネル行列の値が変わること
TEST(WhiteKernelTest, SetHyperparametersChangesKernelValue)
{
  WhiteKernel kernel(1.0);
  VectorXd theta(1);
  theta(0) = std::log(10.0);
  kernel.set_hyperparameters(theta);

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = kernel(X);
  EXPECT_NEAR(K(0, 0), 10.0, kTol);
}

// set_hyperparameters() で設定した値が diag() にも反映されること
TEST(WhiteKernelTest, SetHyperparametersAffectsDiag)
{
  WhiteKernel kernel(1.0);
  VectorXd theta(1);
  theta(0) = std::log(7.5);
  kernel.set_hyperparameters(theta);

  MatrixXd X(2, 1);
  X << 0.0, 1.0;
  VectorXd d = kernel.diag(X);
  EXPECT_NEAR(d(0), 7.5, kTol);
  EXPECT_NEAR(d(1), 7.5, kTol);
}

// ---------------------------------------------------------------------------
// clone / to_string
// ---------------------------------------------------------------------------

// clone() が元のカーネルと同じノイズレベルでカーネル行列を返す独立したインスタンスを返すこと
TEST(WhiteKernelTest, Clone)
{
  WhiteKernel kernel(6.0);
  std::shared_ptr<Kernel> cloned = kernel.clone();

  ASSERT_NE(cloned.get(), nullptr);
  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K_orig = kernel(X);
  MatrixXd K_clone = (*cloned)(X);
  EXPECT_NEAR(K_orig(0, 0), K_clone(0, 0), kTol);
  EXPECT_NEAR(K_clone(0, 0), 6.0, kTol);
}

// clone() で作成したインスタンスが元のインスタンスから独立していること
TEST(WhiteKernelTest, CloneIsIndependent)
{
  WhiteKernel kernel(2.0);
  std::shared_ptr<Kernel> cloned = kernel.clone();

  // クローンのハイパーパラメータを変更
  VectorXd theta(1);
  theta(0) = std::log(8.0);
  cloned->set_hyperparameters(theta);

  // 元のカーネルには影響しないこと
  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K_orig = kernel(X);
  MatrixXd K_clone = (*cloned)(X);
  EXPECT_NEAR(K_orig(0, 0), 2.0, kTol);
  EXPECT_NEAR(K_clone(0, 0), 8.0, kTol);
}

// to_string() が "WhiteKernel(" を含む文字列を返すこと
TEST(WhiteKernelTest, ToString)
{
  WhiteKernel kernel(2.0);
  std::string s = kernel.to_string();
  EXPECT_FALSE(s.empty());
  EXPECT_NE(s.find("WhiteKernel("), std::string::npos);
}

// to_string() がノイズレベルの値を含むこと
TEST(WhiteKernelTest, ToStringContainsNoiseLevel)
{
  WhiteKernel kernel(3.5);
  std::string s = kernel.to_string();
  // "3.5" または "3.500000" のような値が含まれること
  EXPECT_NE(s.find("3.5"), std::string::npos);
}

// ---------------------------------------------------------------------------
// main (GTest 実行用)
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
