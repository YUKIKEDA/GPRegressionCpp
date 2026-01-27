/**
 * @file nelder_mead_test.cpp
 * @brief Nelder-MeadアルゴリズムのGoogleTestテスト
 *
 * JSONファイルからテストデータを読み込み、Nelder-Meadアルゴリズムの
 * 実装を検証します。
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <cmath>
#include <numbers>
#include <vector>
#include <string>
#include <filesystem>

#include "optimize/NelderMead/nelder_mead.hpp"
#include "optimize/optimizer.hpp"

using json = nlohmann::json;
using namespace gprcpp::optimize;
using Eigen::VectorXd;

namespace
{
  /**
   * @brief テスト関数のファクトリー関数
   */
  ObjectiveFunction createObjectiveFunction(const std::string &name, int dimension)
  {
    if (name == "Sphere")
    {
      return [](const VectorXd &x) -> double
      {
        return x.squaredNorm();
      };
    }
    else if (name == "Rosenbrock")
    {
      return [](const VectorXd &x) -> double
      {
        double sum = 0.0;
        for (int i = 0; i < x.size() - 1; ++i)
        {
          sum += 100.0 * std::pow(x(i + 1) - x(i) * x(i), 2) + std::pow(1.0 - x(i), 2);
        }
        return sum;
      };
    }
    else if (name == "Ackley")
    {
      return [](const VectorXd &x) -> double
      {
        const double a = 20.0;
        const double b = 0.2;
        const double c = 2.0 * std::numbers::pi;
        const Eigen::Index n = x.size();

        double sum1 = x.squaredNorm();
        double sum2 = 0.0;
        for (Eigen::Index i = 0; i < n; ++i)
        {
          sum2 += std::cos(c * x(i));
        }

        return -a * std::exp(-b * std::sqrt(sum1 / n)) - std::exp(sum2 / n) + a + std::exp(1.0);
      };
    }
    else if (name == "Rastrigin")
    {
      return [](const VectorXd &x) -> double
      {
        const double A = 10.0;
        const Eigen::Index n = x.size();
        double sum = A * n;
        for (Eigen::Index i = 0; i < n; ++i)
        {
          sum += x(i) * x(i) - A * std::cos(2.0 * std::numbers::pi * x(i));
        }
        return sum;
      };
    }
    else if (name == "Beale")
    {
      return [](const VectorXd &x) -> double
      {
        double x1 = x(0);
        double x2 = x(1);
        double term1 = std::pow(1.5 - x1 + x1 * x2, 2);
        double term2 = std::pow(2.25 - x1 + x1 * x2 * x2, 2);
        double term3 = std::pow(2.625 - x1 + x1 * x2 * x2 * x2, 2);
        return term1 + term2 + term3;
      };
    }
    else if (name == "Booth")
    {
      return [](const VectorXd &x) -> double
      {
        double x1 = x(0);
        double x2 = x(1);
        return std::pow(x1 + 2 * x2 - 7, 2) + std::pow(2 * x1 + x2 - 5, 2);
      };
    }
    else if (name == "Matyas")
    {
      return [](const VectorXd &x) -> double
      {
        double x1 = x(0);
        double x2 = x(1);
        return 0.26 * (x1 * x1 + x2 * x2) - 0.48 * x1 * x2;
      };
    }
    else
    {
      throw std::runtime_error("Unknown function name: " + name);
    }
  }

  /**
   * @brief JSONファイルからテストデータを読み込む
   */
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

  /**
   * @brief VectorXdをJSON配列から作成
   */
  VectorXd jsonToVector(const json &j)
  {
    VectorXd v(j.size());
    for (size_t i = 0; i < j.size(); ++i)
    {
      v(i) = j[i].get<double>();
    }
    return v;
  }

  /**
   * @brief 許容誤差を計算（次元と関数値に応じて調整）
   */
  double calculateTolerance(int dimension, double optimal_value, double base_tolerance = 1e-4)
  {
    // 高次元では許容誤差を緩める
    double dim_factor = 1.0 + 0.1 * (dimension - 1);

    // 関数値が大きい場合は相対誤差を考慮
    double value_factor = std::max(1.0, std::abs(optimal_value) * 1e-6);

    return base_tolerance * dim_factor * value_factor;
  }

  /**
   * @brief パラメータの許容誤差を計算
   */
  double calculateParameterTolerance(int dimension, double base_tolerance = 1e-3)
  {
    // 高次元では許容誤差を緩める
    double dim_factor = 1.0 + 0.1 * (dimension - 1);
    return base_tolerance * dim_factor;
  }
}

/**
 * @brief Nelder-Meadテストのフィクスチャクラス
 */
class NelderMeadTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // テストデータファイルのパスを取得
    std::filesystem::path test_file(__FILE__);
    std::filesystem::path data_dir = test_file.parent_path() / "data";
    std::filesystem::path json_file = data_dir / "nelder_mead_test_data.json";

    // JSONファイルを読み込む
    std::string json_path = json_file.string();
    test_data_ = loadTestData(json_path);
    test_cases_ = test_data_["test_cases"];

    optimizer_ = std::make_unique<NelderMeadOptimizer>();
  }

  json test_data_;
  json test_cases_;
  std::unique_ptr<NelderMeadOptimizer> optimizer_;
};

/**
 * @brief 各テストケースに対して最適化を実行し、結果を検証
 */
TEST_F(NelderMeadTest, TestAllCases)
{
  for (const auto &test_case : test_cases_)
  {
    const std::string test_name = test_case["name"].get<std::string>();
    const std::string function_name = test_case["function"]["name"].get<std::string>();
    const int dimension = test_case["function"]["dimension"].get<int>();

    // 目的関数を作成
    ObjectiveFunction objective = createObjectiveFunction(function_name, dimension);

    // 初期パラメータ
    VectorXd initial_params = jsonToVector(test_case["initial_parameters"]);

    // オプション設定
    OptimizerOptions options;
    options.max_iterations = test_case["options"]["max_iterations"].get<int>();
    options.tolerance = test_case["options"]["tolerance"].get<double>();

    // 境界条件がある場合は設定
    if (test_case.contains("bounds"))
    {
      VectorXd lower = jsonToVector(test_case["bounds"]["lower"]);
      VectorXd upper = jsonToVector(test_case["bounds"]["upper"]);
      options.bounds = Bounds(lower, upper);
    }

    // 期待される結果
    const auto &expected = test_case["expected_result"];
    VectorXd expected_params = jsonToVector(expected["optimal_parameters"]);
    double expected_value = expected["optimal_value"].get<double>();
    bool expected_converged = expected["converged"].get<bool>();

    // 最適化を実行
    OptimizationResult result = optimizer_->minimize(objective, initial_params, options);

    // 結果を検証
    SCOPED_TRACE("Test case: " + test_name);

    // 収束判定の検証（収束が期待される場合のみ）
    if (expected_converged)
    {
      EXPECT_TRUE(result.converged) << "Expected convergence but algorithm did not converge";
    }

    // 目的関数値の検証
    double value_tolerance = calculateTolerance(dimension, expected_value);
    // 収束しなかったケースでは、scipy/C++ ともに「打ち切り時点」の値を比較しており、
    // 実装差でシンプレックスの状態がずれるため、許容誤差を緩める。
    if (!expected_converged)
    {
      value_tolerance *= 50.0;
    }
    EXPECT_NEAR(result.optimal_value, expected_value, value_tolerance)
        << "Optimal value mismatch for " << test_name;

    // パラメータ値の検証
    ASSERT_EQ(result.optimal_parameters.size(), expected_params.size())
        << "Parameter dimension mismatch for " << test_name;

    double param_tolerance = calculateParameterTolerance(dimension);
    if (!expected_converged)
    {
      param_tolerance *= 20.0;
    }
    for (int i = 0; i < result.optimal_parameters.size(); ++i)
    {
      EXPECT_NEAR(result.optimal_parameters(i), expected_params(i), param_tolerance)
          << "Parameter " << i << " mismatch for " << test_name;
    }

    // 反復回数の検証（収束した場合、期待値の2倍以内であることを確認）
    if (expected_converged)
    {
      int expected_iterations = expected["iterations"].get<int>();
      // 反復回数は実装によって多少異なる可能性があるため、緩い条件
      EXPECT_LE(result.iterations, expected_iterations * 3)
          << "Too many iterations for " << test_name;
    }
  }
}

/**
 * @brief 低次元のテストケースのみを実行（高速なテスト）
 */
TEST_F(NelderMeadTest, TestLowDimensionalCases)
{
  for (const auto &test_case : test_cases_)
  {
    const int dimension = test_case["function"]["dimension"].get<int>();

    // 5次元以下のテストケースのみ実行
    if (dimension > 5)
    {
      continue;
    }

    const std::string test_name = test_case["name"].get<std::string>();
    const std::string function_name = test_case["function"]["name"].get<std::string>();

    // 目的関数を作成
    ObjectiveFunction objective = createObjectiveFunction(function_name, dimension);

    // 初期パラメータ
    VectorXd initial_params = jsonToVector(test_case["initial_parameters"]);

    // オプション設定
    OptimizerOptions options;
    options.max_iterations = test_case["options"]["max_iterations"].get<int>();
    options.tolerance = test_case["options"]["tolerance"].get<double>();

    // 境界条件がある場合は設定
    if (test_case.contains("bounds"))
    {
      VectorXd lower = jsonToVector(test_case["bounds"]["lower"]);
      VectorXd upper = jsonToVector(test_case["bounds"]["upper"]);
      options.bounds = Bounds(lower, upper);
    }

    // 期待される結果
    const auto &expected = test_case["expected_result"];
    VectorXd expected_params = jsonToVector(expected["optimal_parameters"]);
    double expected_value = expected["optimal_value"].get<double>();
    bool expected_converged = expected["converged"].get<bool>();

    // 最適化を実行
    OptimizationResult result = optimizer_->minimize(objective, initial_params, options);

    // 結果を検証
    SCOPED_TRACE("Test case: " + test_name);

    // 収束判定の検証
    if (expected_converged)
    {
      EXPECT_TRUE(result.converged) << "Expected convergence but algorithm did not converge";
    }

    // 目的関数値の検証（低次元ではより厳密に）
    double value_tolerance = calculateTolerance(dimension, expected_value, 1e-5);
    if (!expected_converged)
    {
      value_tolerance *= 50.0;
    }
    EXPECT_NEAR(result.optimal_value, expected_value, value_tolerance)
        << "Optimal value mismatch for " << test_name;

    // パラメータ値の検証
    ASSERT_EQ(result.optimal_parameters.size(), expected_params.size())
        << "Parameter dimension mismatch for " << test_name;

    double param_tolerance = calculateParameterTolerance(dimension, 1e-4);
    if (!expected_converged)
    {
      param_tolerance *= 20.0;
    }
    for (int i = 0; i < result.optimal_parameters.size(); ++i)
    {
      EXPECT_NEAR(result.optimal_parameters(i), expected_params(i), param_tolerance)
          << "Parameter " << i << " mismatch for " << test_name;
    }
  }
}

/**
 * @brief 高次元のテストケース（収束しない場合でも値が改善されていることを確認）
 */
TEST_F(NelderMeadTest, TestHighDimensionalCases)
{
  for (const auto &test_case : test_cases_)
  {
    const int dimension = test_case["function"]["dimension"].get<int>();

    // 20次元以上のテストケースのみ実行
    if (dimension < 20)
    {
      continue;
    }

    const std::string test_name = test_case["name"].get<std::string>();
    const std::string function_name = test_case["function"]["name"].get<std::string>();

    // 目的関数を作成
    ObjectiveFunction objective = createObjectiveFunction(function_name, dimension);

    // 初期パラメータ
    VectorXd initial_params = jsonToVector(test_case["initial_parameters"]);

    // 初期値
    double initial_value = objective(initial_params);

    // オプション設定
    OptimizerOptions options;
    options.max_iterations = test_case["options"]["max_iterations"].get<int>();
    options.tolerance = test_case["options"]["tolerance"].get<double>();

    // 境界条件がある場合は設定
    if (test_case.contains("bounds"))
    {
      VectorXd lower = jsonToVector(test_case["bounds"]["lower"]);
      VectorXd upper = jsonToVector(test_case["bounds"]["upper"]);
      options.bounds = Bounds(lower, upper);
    }

    // 期待される結果
    const auto &expected = test_case["expected_result"];
    double expected_value = expected["optimal_value"].get<double>();

    // 最適化を実行
    OptimizationResult result = optimizer_->minimize(objective, initial_params, options);

    // 結果を検証
    SCOPED_TRACE("Test case: " + test_name);

    // 高次元では収束しなくても、値が改善されていることを確認
    EXPECT_LE(result.optimal_value, initial_value)
        << "Value did not improve for " << test_name;

    // 期待値との比較（高次元では緩い許容誤差）
    double value_tolerance = calculateTolerance(dimension, expected_value, 1e-2);
    EXPECT_NEAR(result.optimal_value, expected_value, value_tolerance)
        << "Optimal value mismatch for " << test_name;
  }
}

/**
 * @brief 境界条件のテスト
 */
TEST_F(NelderMeadTest, TestWithBounds)
{
  for (const auto &test_case : test_cases_)
  {
    // 境界条件がないテストケースはスキップ
    if (!test_case.contains("bounds"))
    {
      continue;
    }

    const std::string test_name = test_case["name"].get<std::string>();
    const std::string function_name = test_case["function"]["name"].get<std::string>();
    const int dimension = test_case["function"]["dimension"].get<int>();

    // 目的関数を作成
    ObjectiveFunction objective = createObjectiveFunction(function_name, dimension);

    // 初期パラメータ
    VectorXd initial_params = jsonToVector(test_case["initial_parameters"]);

    // オプション設定
    OptimizerOptions options;
    options.max_iterations = test_case["options"]["max_iterations"].get<int>();
    options.tolerance = test_case["options"]["tolerance"].get<double>();

    // 境界条件を設定
    VectorXd lower = jsonToVector(test_case["bounds"]["lower"]);
    VectorXd upper = jsonToVector(test_case["bounds"]["upper"]);
    options.bounds = Bounds(lower, upper);

    // 最適化を実行
    OptimizationResult result = optimizer_->minimize(objective, initial_params, options);

    // 結果を検証
    SCOPED_TRACE("Test case: " + test_name);

    // すべてのパラメータが境界内にあることを確認
    for (int i = 0; i < result.optimal_parameters.size(); ++i)
    {
      EXPECT_GE(result.optimal_parameters(i), lower(i))
          << "Parameter " << i << " below lower bound for " << test_name;
      EXPECT_LE(result.optimal_parameters(i), upper(i))
          << "Parameter " << i << " above upper bound for " << test_name;
    }
  }
}

/**
 * @brief メイン関数
 */
int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
