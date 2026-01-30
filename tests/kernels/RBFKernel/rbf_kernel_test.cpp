/**
 * @file rbf_kernel_test.cpp
 * @brief RBFカーネル (RBF) の GoogleTest テスト
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "kernels/RBFKernel/rbf_kernel.hpp"
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

// デフォルトコンストラクタで、ハイパーパラメータ数が1・定常・等方・theta=log(1)=0 となること
TEST(RBFKernelTest, DefaultConstructor)
{
  RBF kernel;
  EXPECT_EQ(kernel.num_hyperparameters(), 1);
  EXPECT_TRUE(kernel.is_stationary());
  EXPECT_TRUE(kernel.is_isotropic());
  VectorXd theta = kernel.get_hyperparameters();
  ASSERT_EQ(theta.size(), 1);
  EXPECT_NEAR(theta(0), 0.0, kTol);
}

// スカラー長さスケールを指定したコンストラクタで、get_hyperparameters() が log(l) を返すこと
TEST(RBFKernelTest, ConstructorWithScalarLengthScale)
{
  RBF kernel(2.5);
  VectorXd theta = kernel.get_hyperparameters();
  ASSERT_EQ(theta.size(), 1);
  EXPECT_NEAR(theta(0), std::log(2.5), kTol);
  EXPECT_TRUE(kernel.is_isotropic());
}

// 下限・上限を指定したコンストラクタで、ハイパーパラメータが正しく設定されること
TEST(RBFKernelTest, ConstructorWithBounds)
{
  RBF kernel(3.0, 1e-6, 1e4);
  VectorXd theta = kernel.get_hyperparameters();
  EXPECT_NEAR(theta(0), std::log(3.0), kTol);
}

// ベクトル長さスケール（ARD）でコンストラクタを呼ぶと、異方・ハイパーパラメータ数が次元数になること
TEST(RBFKernelTest, ConstructorWithVectorLengthScale)
{
  VectorXd l(3);
  l << 1.0, 2.0, 0.5;
  RBF kernel(l);
  EXPECT_EQ(kernel.num_hyperparameters(), 3);
  EXPECT_FALSE(kernel.is_isotropic());
  VectorXd theta = kernel.get_hyperparameters();
  ASSERT_EQ(theta.size(), 3);
  EXPECT_NEAR(theta(0), std::log(1.0), kTol);
  EXPECT_NEAR(theta(1), std::log(2.0), kTol);
  EXPECT_NEAR(theta(2), std::log(0.5), kTol);
}

// ---------------------------------------------------------------------------
// カーネル行列 operator() - K(X, X)
// ---------------------------------------------------------------------------

// K(X, X) が正方行列で、対角成分が 1、非負かつ対称であること
TEST(RBFKernelTest, KernelMatrixSameInput)
{
  RBF kernel(1.0);
  MatrixXd X(3, 2);
  X << 0.0, 0.0,
      1.0, 0.0,
      0.0, 1.0;

  MatrixXd K = kernel(X, X);
  EXPECT_EQ(K.rows(), 3);
  EXPECT_EQ(K.cols(), 3);
  for (int i = 0; i < 3; ++i)
  {
    EXPECT_NEAR(K(i, i), 1.0, kTol) << "i=" << i;
    for (int j = 0; j < 3; ++j)
    {
      EXPECT_GE(K(i, j), 0.0);
      EXPECT_LE(K(i, j), 1.0);
      EXPECT_NEAR(K(i, j), K(j, i), kTol) << "i=" << i << " j=" << j;
    }
  }
}

// 第2引数を省略した kernel(X) が K(X, X) と一致すること
TEST(RBFKernelTest, KernelMatrixSingleArgument)
{
  RBF kernel(1.0);
  MatrixXd X(2, 1);
  X << 0.0, 1.0;

  MatrixXd K1 = kernel(X);
  MatrixXd K2 = kernel(X, X);
  EXPECT_EQ(K1.rows(), K2.rows());
  EXPECT_EQ(K1.cols(), K2.cols());
  for (Eigen::Index i = 0; i < K1.size(); ++i)
  {
    EXPECT_NEAR(K1(i), K2(i), kTol);
  }
}

// 同一点では k(x,x) = 1 であること
TEST(RBFKernelTest, KernelMatrixSamePointIsOne)
{
  RBF kernel(1.0);
  MatrixXd X(1, 2);
  X << 1.5, -2.3;

  MatrixXd K = kernel(X);
  EXPECT_EQ(K.rows(), 1);
  EXPECT_EQ(K.cols(), 1);
  EXPECT_NEAR(K(0, 0), 1.0, kTol);
}

// 既知の値: 1次元で x=0, y=1, l=1 のとき k = exp(-0.5) であること
TEST(RBFKernelTest, KernelMatrixKnownValue1D)
{
  RBF kernel(1.0);
  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd Y(1, 1);
  Y << 1.0;

  MatrixXd K = kernel(X, Y);
  EXPECT_EQ(K.rows(), 1);
  EXPECT_EQ(K.cols(), 1);
  double expected = std::exp(-0.5 * 1.0 * 1.0); // exp(-0.5 * (1/l)^2), l=1
  EXPECT_NEAR(K(0, 0), expected, kTol);
}

// 長さスケール l を大きくすると同じ距離でもカーネル値が大きくなること（より滑らか）
TEST(RBFKernelTest, KernelMatrixLargerLengthScaleIncreasesValue)
{
  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd Y(1, 1);
  Y << 1.0;

  RBF kernel_small(0.5);
  RBF kernel_large(2.0);
  double k_small = kernel_small(X, Y)(0, 0);
  double k_large = kernel_large(X, Y)(0, 0);
  EXPECT_GT(k_large, k_small);
  EXPECT_NEAR(k_small, std::exp(-0.5 * (1.0 / 0.5) * (1.0 / 0.5)), kTol); // exp(-2)
  EXPECT_NEAR(k_large, std::exp(-0.5 * (1.0 / 2.0) * (1.0 / 2.0)), kTol); // exp(-0.125)
}

// K(X1, X2) の形状が (X1 の行数, X2 の行数) であること
TEST(RBFKernelTest, KernelMatrixTwoInputsShape)
{
  RBF kernel(1.0);
  MatrixXd X1(2, 1);
  X1 << 1.0, 2.0;
  MatrixXd X2(4, 1);
  X2 << 0.0, 1.0, 2.0, 3.0;

  MatrixXd K = kernel(X1, X2);
  EXPECT_EQ(K.rows(), 2);
  EXPECT_EQ(K.cols(), 4);
}

// 異方（ARD）カーネルで次元ごとの長さスケールが効いていること
TEST(RBFKernelTest, KernelMatrixARD)
{
  VectorXd l(2);
  l << 1.0, 2.0; // 第1次元 l=1、第2次元 l=2
  RBF kernel(l);

  // 第1次元だけ 1 離れた2点: スケール後距離 1 → k = exp(-0.5)
  MatrixXd A(1, 2);
  A << 0.0, 0.0;
  MatrixXd B(1, 2);
  B << 1.0, 0.0;
  double k1 = kernel(A, B)(0, 0);
  EXPECT_NEAR(k1, std::exp(-0.5), kTol);

  // 第2次元だけ 2 離れた2点: スケール後距離 1 → k = exp(-0.5)
  MatrixXd C(1, 2);
  C << 0.0, 0.0;
  MatrixXd D(1, 2);
  D << 0.0, 2.0;
  double k2 = kernel(C, D)(0, 0);
  EXPECT_NEAR(k2, std::exp(-0.5), kTol);
}

// ---------------------------------------------------------------------------
// 対角 diag()
// ---------------------------------------------------------------------------

// diag(X) の長さが X の行数で、全要素が 1 であること（RBF の対角は常に exp(0)=1）
TEST(RBFKernelTest, Diag)
{
  RBF kernel(1.0);
  MatrixXd X(4, 2);
  X << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;

  VectorXd d = kernel.diag(X);
  EXPECT_EQ(d.size(), 4);
  for (int i = 0; i < 4; ++i)
  {
    EXPECT_NEAR(d(i), 1.0, kTol);
  }
}

// diag(X) の各要素が K(X, X) の対角成分と一致すること
TEST(RBFKernelTest, DiagConsistentWithKernelMatrix)
{
  RBF kernel(2.0);
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
TEST(RBFKernelTest, GetSetHyperparametersRoundtrip)
{
  RBF kernel(4.0);
  VectorXd theta = kernel.get_hyperparameters();
  kernel.set_hyperparameters(theta);
  VectorXd theta2 = kernel.get_hyperparameters();
  EXPECT_EQ(theta.size(), theta2.size());
  EXPECT_NEAR(theta(0), theta2(0), kTol);
}

// set_hyperparameters() でハイパーパラメータを設定した後、カーネル行列の値が変わること
TEST(RBFKernelTest, SetHyperparametersChangesKernelValue)
{
  RBF kernel(1.0);
  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd Y(1, 1);
  Y << 1.0;
  double k_before = kernel(X, Y)(0, 0);

  VectorXd theta(1);
  theta(0) = std::log(10.0); // 非常に大きい l → ほぼ 1 に近い
  kernel.set_hyperparameters(theta);
  double k_after = kernel(X, Y)(0, 0);
  EXPECT_GT(k_after, k_before);
  EXPECT_NEAR(k_after, std::exp(-0.5 * 0.1 * 0.1), kTol); // (1/10)^2 * 0.5
}

// set_hyperparameters() でサイズが一致しない場合は無視されること（実装の仕様）
TEST(RBFKernelTest, SetHyperparametersWrongSizeIgnored)
{
  RBF kernel(1.0);
  VectorXd theta_orig = kernel.get_hyperparameters();
  VectorXd theta_wrong(2);
  theta_wrong << 0.0, 0.0;
  kernel.set_hyperparameters(theta_wrong);
  VectorXd theta_after = kernel.get_hyperparameters();
  EXPECT_EQ(theta_after.size(), 1);
  EXPECT_NEAR(theta_after(0), theta_orig(0), kTol);
}

// 異方カーネルの get/set ラウンドトリップ
TEST(RBFKernelTest, GetSetHyperparametersRoundtripARD)
{
  VectorXd l(2);
  l << 1.5, 2.5;
  RBF kernel(l);
  VectorXd theta = kernel.get_hyperparameters();
  kernel.set_hyperparameters(theta);
  VectorXd theta2 = kernel.get_hyperparameters();
  EXPECT_EQ(theta.size(), theta2.size());
  for (int i = 0; i < theta.size(); ++i)
  {
    EXPECT_NEAR(theta(i), theta2(i), kTol);
  }
}

// ---------------------------------------------------------------------------
// clone / to_string
// ---------------------------------------------------------------------------

// clone() が元のカーネルと同じ長さスケールでカーネル行列を返す独立したインスタンスを返すこと
TEST(RBFKernelTest, Clone)
{
  RBF kernel(2.0);
  std::shared_ptr<Kernel> cloned = kernel.clone();

  ASSERT_NE(cloned.get(), nullptr);
  MatrixXd X(2, 1);
  X << 0.0, 1.0;
  MatrixXd K_orig = kernel(X, X);
  MatrixXd K_clone = (*cloned)(X, X);
  EXPECT_EQ(K_orig.rows(), K_clone.rows());
  EXPECT_EQ(K_orig.cols(), K_clone.cols());
  for (Eigen::Index i = 0; i < K_orig.size(); ++i)
  {
    EXPECT_NEAR(K_orig(i), K_clone(i), kTol);
  }
}

// clone() で作成したインスタンスが元のインスタンスから独立していること
TEST(RBFKernelTest, CloneIsIndependent)
{
  RBF kernel(1.0);
  std::shared_ptr<Kernel> cloned = kernel.clone();

  VectorXd theta(1);
  theta(0) = std::log(5.0);
  cloned->set_hyperparameters(theta);

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd Y(1, 1);
  Y << 1.0;
  double k_orig = kernel(X, Y)(0, 0);
  double k_clone = (*cloned)(X, Y)(0, 0);
  EXPECT_NEAR(k_orig, std::exp(-0.5), kTol); // l=1
  EXPECT_GT(k_clone, k_orig);                // l=5 の方が大きい
}

// to_string() が "RBF(" を含むこと
TEST(RBFKernelTest, ToString)
{
  RBF kernel(2.0);
  std::string s = kernel.to_string();
  EXPECT_FALSE(s.empty());
  EXPECT_NE(s.find("RBF("), std::string::npos);
  EXPECT_NE(s.find("length_scale"), std::string::npos);
}

// to_string() が等方のときスカラー表示であること
TEST(RBFKernelTest, ToStringIsotropic)
{
  RBF kernel(3.5);
  std::string s = kernel.to_string();
  EXPECT_NE(s.find("3.5"), std::string::npos);
}

// to_string() が異方のときベクトル表示であること
TEST(RBFKernelTest, ToStringARD)
{
  VectorXd l(2);
  l << 1.0, 2.0;
  RBF kernel(l);
  std::string s = kernel.to_string();
  EXPECT_NE(s.find("RBF("), std::string::npos);
  EXPECT_NE(s.find("length_scale"), std::string::npos);
}

// ---------------------------------------------------------------------------
// main (GTest 実行用)
// ---------------------------------------------------------------------------

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
