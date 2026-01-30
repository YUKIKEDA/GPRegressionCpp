/**
 * @file kernel_oprator_test.cpp
 * @brief カーネル演算子 (SumKernelOperator, ProductKernelOperator) の GoogleTest テスト
 */

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <cmath>
#include <memory>

#include "kernels/kernel.hpp"
#include "kernels/kernel_oprator.hpp"
#include "kernels/ConstantKernel/constant_kernel.hpp"
#include "kernels/RBFKernel/rbf_kernel.hpp"

using namespace gprcpp::kernels;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
  const double kTol = 1e-10;
}

// ---------------------------------------------------------------------------
// SumKernelOperator: コンストラクタ・基本性質
// ---------------------------------------------------------------------------

TEST(KernelOperatorTest, SumKernelOperator_Constructor)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<ConstantKernel>(2.0);
  SumKernelOperator sum_op(k1, k2);

  EXPECT_EQ(sum_op.num_hyperparameters(), 2);
  EXPECT_TRUE(sum_op.is_stationary());

  VectorXd theta = sum_op.get_hyperparameters();
  ASSERT_EQ(theta.size(), 2);
  EXPECT_NEAR(theta(0), std::log(1.0), kTol);
  EXPECT_NEAR(theta(1), std::log(2.0), kTol);
}

TEST(KernelOperatorTest, SumKernelOperator_KernelMatrixSameInput)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<ConstantKernel>(3.0);
  SumKernelOperator sum_op(k1, k2);

  MatrixXd X(2, 2);
  X << 0.0, 0.0,
      1.0, 1.0;

  MatrixXd K = sum_op(X, X);
  EXPECT_EQ(K.rows(), 2);
  EXPECT_EQ(K.cols(), 2);
  // K = 1 + 3 = 4 が全要素
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      EXPECT_NEAR(K(i, j), 4.0, kTol) << "i=" << i << " j=" << j;
}

TEST(KernelOperatorTest, SumKernelOperator_KernelMatrixDifferentInput)
{
  auto k1 = std::make_shared<ConstantKernel>(2.0);
  auto k2 = std::make_shared<ConstantKernel>(1.0);
  SumKernelOperator sum_op(k1, k2);

  MatrixXd X1(2, 1), X2(3, 1);
  X1 << 0.0, 1.0;
  X2 << 0.0, 1.0, 2.0;

  MatrixXd K = sum_op(X1, X2);
  EXPECT_EQ(K.rows(), 2);
  EXPECT_EQ(K.cols(), 3);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_NEAR(K(i, j), 3.0, kTol);
}

TEST(KernelOperatorTest, SumKernelOperator_Diag)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<ConstantKernel>(2.0);
  SumKernelOperator sum_op(k1, k2);

  MatrixXd X(3, 2);
  X << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;

  VectorXd d = sum_op.diag(X);
  ASSERT_EQ(d.size(), 3);
  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(d(i), 3.0, kTol);
}

TEST(KernelOperatorTest, SumKernelOperator_SetGetHyperparameters)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<ConstantKernel>(2.0);
  SumKernelOperator sum_op(k1, k2);

  VectorXd theta_new(2);
  theta_new << std::log(5.0), std::log(10.0);
  sum_op.set_hyperparameters(theta_new);

  VectorXd theta = sum_op.get_hyperparameters();
  EXPECT_NEAR(theta(0), std::log(5.0), kTol);
  EXPECT_NEAR(theta(1), std::log(10.0), kTol);

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = sum_op(X, X);
  EXPECT_NEAR(K(0, 0), 5.0 + 10.0, kTol);
}

TEST(KernelOperatorTest, SumKernelOperator_Clone)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<ConstantKernel>(2.0);
  SumKernelOperator sum_op(k1, k2);

  std::shared_ptr<Kernel> cloned = sum_op.clone();
  ASSERT_NE(cloned.get(), nullptr);

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K_orig = sum_op(X, X);
  MatrixXd K_clone = (*cloned)(X, X);
  EXPECT_NEAR(K_orig(0, 0), K_clone(0, 0), kTol);
}

TEST(KernelOperatorTest, SumKernelOperator_ToString)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<ConstantKernel>(2.0);
  SumKernelOperator sum_op(k1, k2);

  std::string s = sum_op.to_string();
  EXPECT_TRUE(s.find('+') != std::string::npos);
}

// ---------------------------------------------------------------------------
// ProductKernelOperator: コンストラクタ・基本性質
// ---------------------------------------------------------------------------

TEST(KernelOperatorTest, ProductKernelOperator_Constructor)
{
  auto k1 = std::make_shared<ConstantKernel>(2.0);
  auto k2 = std::make_shared<ConstantKernel>(3.0);
  ProductKernelOperator prod_op(k1, k2);

  EXPECT_EQ(prod_op.num_hyperparameters(), 2);
  EXPECT_TRUE(prod_op.is_stationary());

  VectorXd theta = prod_op.get_hyperparameters();
  ASSERT_EQ(theta.size(), 2);
  EXPECT_NEAR(theta(0), std::log(2.0), kTol);
  EXPECT_NEAR(theta(1), std::log(3.0), kTol);
}

TEST(KernelOperatorTest, ProductKernelOperator_KernelMatrixSameInput)
{
  auto k1 = std::make_shared<ConstantKernel>(2.0);
  auto k2 = std::make_shared<ConstantKernel>(3.0);
  ProductKernelOperator prod_op(k1, k2);

  MatrixXd X(2, 2);
  X << 0.0, 0.0, 1.0, 1.0;

  MatrixXd K = prod_op(X, X);
  EXPECT_EQ(K.rows(), 2);
  EXPECT_EQ(K.cols(), 2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      EXPECT_NEAR(K(i, j), 6.0, kTol);
}

TEST(KernelOperatorTest, ProductKernelOperator_Diag)
{
  auto k1 = std::make_shared<ConstantKernel>(2.0);
  auto k2 = std::make_shared<ConstantKernel>(4.0);
  ProductKernelOperator prod_op(k1, k2);

  MatrixXd X(3, 1);
  X << 0.0, 1.0, 2.0;

  VectorXd d = prod_op.diag(X);
  ASSERT_EQ(d.size(), 3);
  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(d(i), 8.0, kTol);
}

TEST(KernelOperatorTest, ProductKernelOperator_SetGetHyperparameters)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<ConstantKernel>(1.0);
  ProductKernelOperator prod_op(k1, k2);

  VectorXd theta_new(2);
  theta_new << std::log(3.0), std::log(4.0);
  prod_op.set_hyperparameters(theta_new);

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = prod_op(X, X);
  EXPECT_NEAR(K(0, 0), 12.0, kTol);
}

TEST(KernelOperatorTest, ProductKernelOperator_ToString)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<ConstantKernel>(2.0);
  ProductKernelOperator prod_op(k1, k2);

  std::string s = prod_op.to_string();
  EXPECT_TRUE(s.find('*') != std::string::npos);
}

// ---------------------------------------------------------------------------
// Sum: Constant + RBF の組み合わせ（形状・対角の一致）
// ---------------------------------------------------------------------------

TEST(KernelOperatorTest, SumKernelOperator_ConstantPlusRBF)
{
  auto k_const = std::make_shared<ConstantKernel>(1.0);
  auto k_rbf = std::make_shared<RBF>(1.0);
  SumKernelOperator sum_op(k_const, k_rbf);

  MatrixXd X(3, 2);
  X << 0.0, 0.0, 1.0, 0.0, 0.0, 1.0;

  MatrixXd K = sum_op(X, X);
  EXPECT_EQ(K.rows(), 3);
  EXPECT_EQ(K.cols(), 3);

  // 対角は K_const(i,i) + K_rbf(i,i) = 1 + 1 = 2
  VectorXd d = sum_op.diag(X);
  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(d(i), 2.0, kTol);

  // 対角と行列の対角要素が一致
  for (int i = 0; i < 3; ++i)
    EXPECT_NEAR(K(i, i), d(i), kTol);
}

// ---------------------------------------------------------------------------
// Product: Constant * RBF
// ---------------------------------------------------------------------------

TEST(KernelOperatorTest, ProductKernelOperator_ConstantTimesRBF)
{
  auto k_const = std::make_shared<ConstantKernel>(2.0);
  auto k_rbf = std::make_shared<RBF>(1.0);
  ProductKernelOperator prod_op(k_const, k_rbf);

  MatrixXd X(2, 1);
  X << 0.0, 1.0;

  MatrixXd K = prod_op(X, X);
  EXPECT_EQ(K.rows(), 2);
  EXPECT_EQ(K.cols(), 2);
  // 対角: 2 * 1 = 2
  EXPECT_NEAR(K(0, 0), 2.0, kTol);
  EXPECT_NEAR(K(1, 1), 2.0, kTol);
  // RBF(0,1) = exp(-0.5), K(0,1) = 2 * exp(-0.5)
  double rbf_01 = std::exp(-0.5);
  EXPECT_NEAR(K(0, 1), 2.0 * rbf_01, kTol);
  EXPECT_NEAR(K(1, 0), 2.0 * rbf_01, kTol);
}

// ---------------------------------------------------------------------------
// operator+ (Kernel + Kernel)
// ---------------------------------------------------------------------------

TEST(KernelOperatorTest, OperatorPlus_TwoKernels)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<ConstantKernel>(2.0);
  std::shared_ptr<Kernel> sum_k = k1 + k2;

  ASSERT_NE(sum_k.get(), nullptr);
  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = (*sum_k)(X, X);
  EXPECT_NEAR(K(0, 0), 3.0, kTol);
}

// ---------------------------------------------------------------------------
// operator* (Kernel * Kernel)
// ---------------------------------------------------------------------------

TEST(KernelOperatorTest, OperatorMultiply_TwoKernels)
{
  auto k1 = std::make_shared<ConstantKernel>(2.0);
  auto k2 = std::make_shared<ConstantKernel>(3.0);
  std::shared_ptr<Kernel> prod_k = k1 * k2;

  ASSERT_NE(prod_k.get(), nullptr);
  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = (*prod_k)(X, X);
  EXPECT_NEAR(K(0, 0), 6.0, kTol);
}

// ---------------------------------------------------------------------------
// operator* (double * Kernel), (Kernel * double)
// ---------------------------------------------------------------------------

TEST(KernelOperatorTest, OperatorMultiply_DoubleTimesKernel)
{
  auto k = std::make_shared<ConstantKernel>(1.0);
  std::shared_ptr<Kernel> scaled = 5.0 * k;

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = (*scaled)(X, X);
  EXPECT_NEAR(K(0, 0), 5.0, kTol);
}

TEST(KernelOperatorTest, OperatorMultiply_KernelTimesDouble)
{
  auto k = std::make_shared<ConstantKernel>(1.0);
  std::shared_ptr<Kernel> scaled = k * 4.0;

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = (*scaled)(X, X);
  EXPECT_NEAR(K(0, 0), 4.0, kTol);
}

// ---------------------------------------------------------------------------
// operator+ (double + Kernel), (Kernel + double)
// ---------------------------------------------------------------------------

TEST(KernelOperatorTest, OperatorPlus_DoublePlusKernel)
{
  auto k = std::make_shared<ConstantKernel>(2.0);
  std::shared_ptr<Kernel> shifted = 3.0 + k;

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = (*shifted)(X, X);
  EXPECT_NEAR(K(0, 0), 5.0, kTol);
}

TEST(KernelOperatorTest, OperatorPlus_KernelPlusDouble)
{
  auto k = std::make_shared<ConstantKernel>(2.0);
  std::shared_ptr<Kernel> shifted = k + 3.0;

  MatrixXd X(1, 1);
  X << 0.0;
  MatrixXd K = (*shifted)(X, X);
  EXPECT_NEAR(K(0, 0), 5.0, kTol);
}

// ---------------------------------------------------------------------------
// is_stationary: 非定常カーネルが含まれる場合は false になる想定
// （現状 Constant と RBF はどちらも定常なので、両方使うと true）
// ---------------------------------------------------------------------------

TEST(KernelOperatorTest, IsStationary_BothStationary)
{
  auto k1 = std::make_shared<ConstantKernel>(1.0);
  auto k2 = std::make_shared<RBF>(1.0);
  SumKernelOperator sum_op(k1, k2);
  EXPECT_TRUE(sum_op.is_stationary());

  ProductKernelOperator prod_op(k1, k2);
  EXPECT_TRUE(prod_op.is_stationary());
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
