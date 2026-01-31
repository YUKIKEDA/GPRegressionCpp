/**
 * @file kmeans_test.cpp
 * @brief K-means クラスタリングの GoogleTest
 *
 * JSON テストデータ（scikit-learn 期待値）を読み込み、fit / predict / transform の結果を比較します。
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <string>

#include "clustering/k-means/kmeans.hpp"

using json = nlohmann::json;
using namespace gprcpp::clustering;
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

  VectorXi jsonToVectorXi(const json &j)
  {
    VectorXi v(static_cast<Eigen::Index>(j.size()));
    for (size_t i = 0; i < j.size(); ++i)
    {
      v(static_cast<Eigen::Index>(i)) = j[i].get<int>();
    }
    return v;
  }

  const double kTol = 1e-10;
}

class KMeansTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    std::filesystem::path p(__FILE__);
    std::filesystem::path data_dir = p.parent_path() / "data";
    std::filesystem::path json_file = data_dir / "kmeans_test_data.json";
    test_data_ = loadTestData(json_file.string());
    test_cases_ = test_data_["test_cases"];
  }

  json test_data_;
  json test_cases_;
};

TEST_F(KMeansTest, AllCasesMatchExpected)
{
  for (auto &tc : test_cases_)
  {
    const std::string name = tc["name"].get<std::string>();
    const json &config = tc["config"];
    const int n_clusters = config["n_clusters"].get<int>();
    const int max_iter = config["max_iter"].get<int>();
    const double tol = config["tol"].get<double>();
    const unsigned random_state = config["random_state"].get<unsigned>();

    const MatrixXd X = jsonToMatrix(tc["data"]["X"]);
    const json &expected = tc["expected"];

    KMeans kmeans =
        config.contains("init_centers")
            ? KMeans(n_clusters, max_iter, tol, random_state,
                     jsonToMatrix(config["init_centers"]))
            : KMeans(n_clusters, max_iter, tol, random_state);
    kmeans.fit(X);

    const MatrixXd exp_centers = jsonToMatrix(expected["cluster_centers"]);
    const VectorXi exp_labels = jsonToVectorXi(expected["labels"]);
    const double exp_inertia = expected["inertia"].get<double>();
    const int exp_n_iter = expected["n_iter"].get<int>();
    const MatrixXd exp_X_transformed = jsonToMatrix(expected["X_transformed"]);
    const VectorXi exp_labels_pred = jsonToVectorXi(expected["labels_pred"]);

    EXPECT_EQ(kmeans.cluster_centers().rows(), exp_centers.rows()) << "case: " << name;
    EXPECT_EQ(kmeans.cluster_centers().cols(), exp_centers.cols()) << "case: " << name;
    for (Eigen::Index i = 0; i < kmeans.cluster_centers().rows(); ++i)
    {
      for (Eigen::Index j = 0; j < kmeans.cluster_centers().cols(); ++j)
      {
        EXPECT_NEAR(kmeans.cluster_centers()(i, j), exp_centers(i, j), kTol)
            << "case: " << name << " cluster_centers[" << i << "," << j << "]";
      }
    }

    EXPECT_EQ(kmeans.labels().size(), exp_labels.size()) << "case: " << name;
    for (Eigen::Index i = 0; i < kmeans.labels().size(); ++i)
    {
      EXPECT_EQ(kmeans.labels()(i), exp_labels(i)) << "case: " << name << " labels[" << i << "]";
    }

    EXPECT_NEAR(kmeans.inertia(), exp_inertia, kTol) << "case: " << name;
    EXPECT_EQ(kmeans.n_iter(), exp_n_iter) << "case: " << name;

    MatrixXd X_transformed = kmeans.transform(X);
    EXPECT_EQ(X_transformed.rows(), exp_X_transformed.rows()) << "case: " << name;
    EXPECT_EQ(X_transformed.cols(), exp_X_transformed.cols()) << "case: " << name;
    for (Eigen::Index i = 0; i < X_transformed.rows(); ++i)
    {
      for (Eigen::Index j = 0; j < X_transformed.cols(); ++j)
      {
        EXPECT_NEAR(X_transformed(i, j), exp_X_transformed(i, j), kTol)
            << "case: " << name << " X_transformed[" << i << "," << j << "]";
      }
    }

    VectorXi labels_pred = kmeans.predict(X);
    EXPECT_EQ(labels_pred.size(), exp_labels_pred.size()) << "case: " << name;
    for (Eigen::Index i = 0; i < labels_pred.size(); ++i)
    {
      EXPECT_EQ(labels_pred(i), exp_labels_pred(i)) << "case: " << name << " labels_pred[" << i << "]";
    }
  }
}

TEST_F(KMeansTest, FitPredictEqualsFitThenPredict)
{
  for (auto &tc : test_cases_)
  {
    const std::string name = tc["name"].get<std::string>();
    const json &config = tc["config"];
    const int n_clusters = config["n_clusters"].get<int>();
    const int max_iter = config["max_iter"].get<int>();
    const double tol = config["tol"].get<double>();
    const unsigned random_state = config["random_state"].get<unsigned>();

    const MatrixXd X = jsonToMatrix(tc["data"]["X"]);

    KMeans kmeans1(n_clusters, max_iter, tol, random_state);
    kmeans1.fit(X);
    VectorXi labels1 = kmeans1.predict(X);

    KMeans kmeans2(n_clusters, max_iter, tol, random_state);
    VectorXi labels2 = kmeans2.fit_predict(X);

    EXPECT_EQ(labels1.size(), labels2.size()) << "case: " << name;
    for (Eigen::Index i = 0; i < labels1.size(); ++i)
    {
      EXPECT_EQ(labels1(i), labels2(i)) << "case: " << name << " [" << i << "]";
    }
  }
}

TEST_F(KMeansTest, FitTransformEqualsFitThenTransform)
{
  for (auto &tc : test_cases_)
  {
    const std::string name = tc["name"].get<std::string>();
    const json &config = tc["config"];
    const int n_clusters = config["n_clusters"].get<int>();
    const int max_iter = config["max_iter"].get<int>();
    const double tol = config["tol"].get<double>();
    const unsigned random_state = config["random_state"].get<unsigned>();

    const MatrixXd X = jsonToMatrix(tc["data"]["X"]);

    KMeans kmeans1(n_clusters, max_iter, tol, random_state);
    kmeans1.fit(X);
    MatrixXd t1 = kmeans1.transform(X);

    KMeans kmeans2(n_clusters, max_iter, tol, random_state);
    MatrixXd t2 = kmeans2.fit_transform(X);

    EXPECT_EQ(t1.rows(), t2.rows()) << "case: " << name;
    EXPECT_EQ(t1.cols(), t2.cols()) << "case: " << name;
    for (Eigen::Index i = 0; i < t1.rows(); ++i)
    {
      for (Eigen::Index j = 0; j < t1.cols(); ++j)
      {
        EXPECT_NEAR(t1(i, j), t2(i, j), kTol) << "case: " << name << " [" << i << "," << j << "]";
      }
    }
  }
}

TEST_F(KMeansTest, InvalidArgumentsThrow)
{
  MatrixXd X = MatrixXd::Random(10, 2);

  EXPECT_THROW(KMeans(0, 300, 1e-4, 42u), std::invalid_argument);
  EXPECT_THROW(KMeans(3, 0, 1e-4, 42u), std::invalid_argument);
  EXPECT_THROW(KMeans(3, 300, -1.0, 42u), std::invalid_argument);

  KMeans kmeans(3, 300, 1e-4, 42u);
  EXPECT_THROW(kmeans.fit(MatrixXd(0, 2)), std::invalid_argument);
  EXPECT_THROW(kmeans.fit(MatrixXd(2, 2)), std::invalid_argument); // n_samples < n_clusters

  kmeans.fit(X);
  EXPECT_THROW(kmeans.predict(MatrixXd(5, 3)), std::invalid_argument); // n_features mismatch
  EXPECT_THROW(kmeans.transform(MatrixXd(5, 3)), std::invalid_argument);

  KMeans kmeans2(3, 300, 1e-4, 42u);
  EXPECT_THROW(kmeans2.predict(X), std::runtime_error);
  EXPECT_THROW(kmeans2.transform(X), std::runtime_error);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
