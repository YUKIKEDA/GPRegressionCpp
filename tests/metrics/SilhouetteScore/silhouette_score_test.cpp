/**
 * @file silhouette_score_test.cpp
 * @brief シルエット係数（Silhouette Coefficient）の GoogleTest テスト
 *
 * JSON ファイルからテストデータを読み込み、scikit-learn と同様の期待値と比較します。
 * X + labels による silhouette_samples / silhouette_score と、
 * 事前計算距離行列 D + labels による silhouette_samples_precomputed / silhouette_score_precomputed を検証します。
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>

#include "metrics/SilhouetteScore/silhouette_score.hpp"

using json = nlohmann::json;
using namespace gprcpp::metrics;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::VectorXi;

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

  VectorXi jsonToVectorXi(const json &j)
  {
    VectorXi v(static_cast<Eigen::Index>(j.size()));
    for (size_t i = 0; i < j.size(); ++i)
    {
      v(static_cast<Eigen::Index>(i)) = j[i].get<int>();
    }
    return v;
  }

  // 高次元・多数サンプル時の浮動小数点誤差蓄積を許容（C++ vs scikit-learn の実装差）
  const double kTol = 1e-8;
}

class SilhouetteScoreTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::filesystem::path p(__FILE__);
    std::filesystem::path data_dir = p.parent_path() / "data";
    std::filesystem::path json_file = data_dir / "silhouette_score_test_data.json";
    test_data_ = loadTestData(json_file.string());
    test_cases_ = test_data_["test_cases"];
  }

  json test_data_;
  json test_cases_;
};

TEST_F(SilhouetteScoreTest, SilhouetteSamplesAndScoreFromX)
{
  for (auto &tc : test_cases_)
  {
    const std::string name = tc["name"].get<std::string>();
    const json &data = tc["data"];

    // precomputed ケースは X がないのでスキップ
    if (data.find("X") == data.end())
    {
      continue;
    }

    const MatrixXd X = jsonToMatrix(data["X"]);
    const VectorXi labels = jsonToVectorXi(data["labels"]);
    const json &expected = tc["expected"];

    const VectorXd exp_samples = jsonToVector(expected["silhouette_samples"]);
    const double exp_score = expected["silhouette_score"].get<double>();

    Eigen::VectorXd samples = silhouette_samples(X, labels);
    double score = silhouette_score(X, labels);

    EXPECT_EQ(samples.size(), exp_samples.size()) << "case: " << name;
    for (Eigen::Index i = 0; i < samples.size(); ++i)
    {
      EXPECT_NEAR(samples(i), exp_samples(i), kTol)
          << "case: " << name << " sample[" << i << "]";
    }
    EXPECT_NEAR(score, exp_score, kTol) << "case: " << name;
  }
}

TEST_F(SilhouetteScoreTest, SilhouetteSamplesAndScorePrecomputed)
{
  for (auto &tc : test_cases_)
  {
    const std::string name = tc["name"].get<std::string>();
    const json &data = tc["data"];

    // precomputed ケースのみ（data.D がある場合）
    if (data.find("D") == data.end())
    {
      continue;
    }

    const MatrixXd D = jsonToMatrix(data["D"]);
    const VectorXi labels = jsonToVectorXi(data["labels"]);
    const json &expected = tc["expected"];

    const VectorXd exp_samples = jsonToVector(expected["silhouette_samples"]);
    const double exp_score = expected["silhouette_score"].get<double>();

    Eigen::VectorXd samples = silhouette_samples_precomputed(D, labels);
    double score = silhouette_score_precomputed(D, labels);

    EXPECT_EQ(samples.size(), exp_samples.size()) << "case: " << name;
    for (Eigen::Index i = 0; i < samples.size(); ++i)
    {
      EXPECT_NEAR(samples(i), exp_samples(i), kTol)
          << "case: " << name << " sample[" << i << "]";
    }
    EXPECT_NEAR(score, exp_score, kTol) << "case: " << name;
  }
}

TEST_F(SilhouetteScoreTest, InvalidInputThrows)
{
  Eigen::MatrixXd X(5, 2);
  X.setZero();
  Eigen::VectorXi labels_bad_size(3);
  labels_bad_size.setZero();

  EXPECT_THROW(silhouette_samples(X, labels_bad_size), std::invalid_argument);
  EXPECT_THROW(silhouette_score(X, labels_bad_size), std::invalid_argument);

  Eigen::MatrixXd empty_X(0, 0);
  Eigen::VectorXi empty_labels(0);
  EXPECT_THROW(silhouette_samples(empty_X, empty_labels), std::invalid_argument);
  EXPECT_THROW(silhouette_score(empty_X, empty_labels), std::invalid_argument);

  Eigen::MatrixXd one_cluster_X(10, 2);
  one_cluster_X.setZero();
  Eigen::VectorXi one_cluster_labels(10);
  one_cluster_labels.setZero();
  EXPECT_THROW(silhouette_samples(one_cluster_X, one_cluster_labels),
               std::invalid_argument);
  EXPECT_THROW(silhouette_score(one_cluster_X, one_cluster_labels),
               std::invalid_argument);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
