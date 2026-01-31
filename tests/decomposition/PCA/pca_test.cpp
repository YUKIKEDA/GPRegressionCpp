/**
 * @file pca_test.cpp
 * @brief 特異値分解を用いた主成分分析（PCA）のGoogleTestテスト
 *
 * JSONファイルからテストデータを読み込み、scikit-learn と同様の期待値と比較します。
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>

#include "decomposition/PCA/pca.hpp"

using json = nlohmann::json;
using namespace gprcpp::decomposition;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace
{
  json loadTestData(const std::string &filepath)
  {
    std::ifstream file(filepath);
    if (!file.is_open())
    {
      throw std::runtime_error("Failed to open test data file: " + filepath);
    }
    json data;
    file >> data;
    return data;
  }

  MatrixXd jsonToMatrix(const json &j)
  {
    if (j.empty() || !j[0].is_array())
    {
      return MatrixXd(0, 0);
    }
    const size_t rows = j.size();
    const size_t cols = j[0].size();
    MatrixXd m(rows, cols);
    for (size_t i = 0; i < rows; ++i)
    {
      for (size_t k = 0; k < cols; ++k)
      {
        m(i, k) = j[i][k].get<double>();
      }
    }
    return m;
  }

  VectorXd jsonToVector(const json &j)
  {
    VectorXd v(static_cast<Eigen::Index>(j.size()));
    for (size_t i = 0; i < j.size(); ++i)
    {
      v(static_cast<Eigen::Index>(i)) = j[i].get<double>();
    }
    return v;
  }

  const double kTol = 1e-10;
}

class PCATest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::filesystem::path p(__FILE__);
    std::filesystem::path data_dir = p.parent_path() / "data";
    std::filesystem::path json_file = data_dir / "pca_test_data.json";
    test_data_ = loadTestData(json_file.string());
    test_cases_ = test_data_["test_cases"];
  }

  json test_data_;
  json test_cases_;
};

TEST_F(PCATest, AllCasesMatchExpected)
{
  for (auto &tc : test_cases_)
  {
    const std::string name = tc["name"].get<std::string>();
    const json &config = tc["config"];
    const int n_components = config["n_components"].get<int>();
    const bool whiten = config["whiten"].get<bool>();
    const MatrixXd X = jsonToMatrix(tc["data"]["X"]);
    const json &expected = tc["expected"];

    PCA pca(n_components, whiten);
    pca.fit(X);

    const VectorXd exp_mean = jsonToVector(expected["mean"]);
    const MatrixXd exp_components = jsonToMatrix(expected["components"]);
    const VectorXd exp_singular = jsonToVector(expected["singular_values"]);
    const VectorXd exp_var = jsonToVector(expected["explained_variance"]);
    const VectorXd exp_ratio = jsonToVector(expected["explained_variance_ratio"]);
    const double exp_noise = expected["noise_variance"].get<double>();
    const MatrixXd exp_X_transformed = jsonToMatrix(expected["X_transformed"]);

    EXPECT_EQ(pca.mean().size(), exp_mean.size()) << "case: " << name;
    for (Eigen::Index i = 0; i < pca.mean().size(); ++i)
    {
      EXPECT_NEAR(pca.mean()(i), exp_mean(i), kTol) << "case: " << name << " mean[" << i << "]";
    }

    EXPECT_EQ(pca.components().rows(), exp_components.rows());
    EXPECT_EQ(pca.components().cols(), exp_components.cols());
    for (Eigen::Index i = 0; i < pca.components().rows(); ++i)
    {
      for (Eigen::Index j = 0; j < pca.components().cols(); ++j)
      {
        EXPECT_NEAR(pca.components()(i, j), exp_components(i, j), kTol)
            << "case: " << name << " components[" << i << "," << j << "]";
      }
    }

    for (Eigen::Index i = 0; i < pca.singular_values().size(); ++i)
    {
      EXPECT_NEAR(pca.singular_values()(i), exp_singular(i), kTol)
          << "case: " << name << " singular_values[" << i << "]";
    }

    for (Eigen::Index i = 0; i < pca.explained_variance().size(); ++i)
    {
      EXPECT_NEAR(pca.explained_variance()(i), exp_var(i), kTol)
          << "case: " << name << " explained_variance[" << i << "]";
    }

    for (Eigen::Index i = 0; i < pca.explained_variance_ratio().size(); ++i)
    {
      EXPECT_NEAR(pca.explained_variance_ratio()(i), exp_ratio(i), kTol)
          << "case: " << name << " explained_variance_ratio[" << i << "]";
    }

    EXPECT_NEAR(pca.noise_variance(), exp_noise, kTol) << "case: " << name;

    MatrixXd X_transformed = pca.transform(X);
    EXPECT_EQ(X_transformed.rows(), exp_X_transformed.rows());
    EXPECT_EQ(X_transformed.cols(), exp_X_transformed.cols());
    for (Eigen::Index i = 0; i < X_transformed.rows(); ++i)
    {
      for (Eigen::Index j = 0; j < X_transformed.cols(); ++j)
      {
        EXPECT_NEAR(X_transformed(i, j), exp_X_transformed(i, j), kTol)
            << "case: " << name << " X_transformed[" << i << "," << j << "]";
      }
    }
  }
}

TEST_F(PCATest, FitTransformEqualsFitThenTransform)
{
  for (auto &tc : test_cases_)
  {
    const json &config = tc["config"];
    const int n_components = config["n_components"].get<int>();
    const bool whiten = config["whiten"].get<bool>();
    const MatrixXd X = jsonToMatrix(tc["data"]["X"]);

    PCA pca1(n_components, whiten);
    pca1.fit(X);
    MatrixXd t1 = pca1.transform(X);

    PCA pca2(n_components, whiten);
    MatrixXd t2 = pca2.fit_transform(X);

    EXPECT_EQ(t1.rows(), t2.rows());
    EXPECT_EQ(t1.cols(), t2.cols());
    for (Eigen::Index i = 0; i < t1.rows(); ++i)
    {
      for (Eigen::Index j = 0; j < t1.cols(); ++j)
      {
        EXPECT_NEAR(t1(i, j), t2(i, j), kTol);
      }
    }
  }
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
