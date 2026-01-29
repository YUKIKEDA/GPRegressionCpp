/**
 * @file dual_annealing_test.cpp
 * @brief Dual AnnealingアルゴリズムのGoogleTestテスト
 *
 * JSONファイルからテストデータを読み込み、Dual Annealingアルゴリズムの
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
#include <iostream>
#include <iomanip>

#include "optimize/DualAnnealing/dual_annealing.hpp"
#include "optimize/optimizer.hpp"

using json = nlohmann::json;
using namespace gprcpp::optimize;
using Eigen::VectorXd;

// TODO: Pythonの結果との比較は参考情報にとどめて、理論値との比較でテストは検証する

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
    else if (name == "Griewank")
    {
      return [](const VectorXd &x) -> double
      {
        const Eigen::Index n = x.size();
        double sum_term = x.squaredNorm() / 4000.0;
        double prod_term = 1.0;
        for (Eigen::Index i = 0; i < n; ++i)
        {
          prod_term *= std::cos(x(i) / std::sqrt(static_cast<double>(i + 1)));
        }
        return sum_term - prod_term + 1.0;
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
    else if (name == "Easom")
    {
      return [](const VectorXd &x) -> double
      {
        double x1 = x(0);
        double x2 = x(1);
        return -std::cos(x1) * std::cos(x2) * std::exp(-((x1 - std::numbers::pi) * (x1 - std::numbers::pi) + (x2 - std::numbers::pi) * (x2 - std::numbers::pi)));
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
   * @brief 結果が理論的最適値に近いかを判定する
   *
   * @param test_name テストケース名
   * @param actual 実際の計算結果
   * @param theoretical 理論上の最適値
   * @param tolerance 許容誤差
   * @return true 合格
   * @return false 不合格
   */
  bool checkOptimality(const std::string &test_name, double actual, double theoretical, double tolerance)
  {
    // 1. 理論値より良い（小さい）場合は無条件でOK（浮動小数点誤差などで起こり得る）
    if (actual <= theoretical + 1e-15)
    {
      return true;
    }

    // 2. 誤差の計算
    double diff = std::abs(actual - theoretical);

    // 3. 許容範囲内かチェック
    if (diff > tolerance)
    {
      std::cerr << "\n[FAILURE] " << test_name << ": Failed to converge to global minimum.\n"
                << "  Theoretical: " << std::fixed << std::setprecision(9) << theoretical << "\n"
                << "  Actual:      " << std::fixed << std::setprecision(9) << actual << "\n"
                << "  Difference:  " << std::scientific << diff << "\n"
                << "  Tolerance:   " << std::scientific << tolerance << "\n"
                << std::endl;
      return false;
    }
    return true;
  }

  /**
   * @brief 次元数に応じた許容誤差を決定する
   */
  double determineTolerance(int dimension)
  {
    if (dimension <= 2)
      return 1e-4; // 低次元は高精度を期待
    if (dimension <= 5)
      return 1e-2; // 5次元程度なら中程度の精度
    if (dimension <= 10)
      return 0.5; // 10次元を超えると厳密な収束は難しい場合がある
    if (dimension <= 20)
      return 2.0; // 20次元
    if (dimension <= 50)
      return 15.0; // 50次元（非常に困難）
    return 50.0;   // 100次元以上（収束していればOK程度）
  }
}

/**
 * @brief Dual Annealingテストのフィクスチャクラス
 */
class DualAnnealingTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // テストデータファイルのパスを取得
    std::filesystem::path test_file(__FILE__);
    std::filesystem::path data_dir = test_file.parent_path() / "data";
    std::filesystem::path json_file = data_dir / "dual_annealing_test_data.json";

    // JSONファイルを読み込む
    std::string json_path = json_file.string();
    test_data_ = loadTestData(json_path);
    test_cases_ = test_data_["test_cases"];

    optimizer_ = std::make_unique<DualAnnealingOptimizer>();
  }

  json test_data_;
  json test_cases_;
  std::unique_ptr<DualAnnealingOptimizer> optimizer_;
};

/**
 * @brief 各テストケースに対して最適化を実行し、結果を検証
 */
TEST_F(DualAnnealingTest, TestAllCases)
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

    // 初期値
    double initial_value = objective(initial_params);

    // オプション設定
    DualAnnealingOptions options;
    options.max_iterations = test_case["options"]["max_iterations"].get<int>();

    options.initial_params = initial_params;

    if (test_case.contains("bounds"))
    {
      // 境界条件がある場合は設定
      VectorXd lower = jsonToVector(test_case["bounds"]["lower"]);
      VectorXd upper = jsonToVector(test_case["bounds"]["upper"]);
      options.bounds = Bounds(lower, upper);
    }
    else
    {
      // 境界がない場合、初期点を中心に広い範囲の境界を設定
      // 初期点の各次元に対して、±50の範囲を設定
      VectorXd lower(dimension);
      VectorXd upper(dimension);
      for (int i = 0; i < dimension; ++i)
      {
        lower(i) = initial_params(i) - 50.0;
        upper(i) = initial_params(i) + 50.0;
      }
      options.bounds = Bounds(lower, upper);
    }

    // Dual Annealing専用オプション
    const auto &da_opts = test_case["da_options"];
    options.initial_temp = da_opts["initial_temp"].get<double>();
    options.visit_param = da_opts["visit_param"].get<double>();
    options.accept_param = da_opts["accept_param"].get<double>();
    options.restart_temp_ratio = da_opts["restart_temp_ratio"].get<double>();
    options.max_function_evaluations = da_opts["max_function_evaluations"].get<int>();

    // シード値がある場合は設定
    if (test_case.contains("seed") && !test_case["seed"].is_null())
    {
      options.seed = test_case["seed"].get<unsigned int>();
    }

    // 期待値 (Pythonの結果)
    bool expected_converged = test_case["expected_result"]["converged"].get<bool>();
    double expected_value = test_case["expected_result"]["optimal_value"].get<double>();
    VectorXd expected_params = jsonToVector(test_case["expected_result"]["optimal_parameters"]);

    // 理論上の最適値を取得
    double theoretical_value = test_case["function"]["optimal_value"].get<double>();
    VectorXd theoretical_params = jsonToVector(test_case["function"]["optimal_point"]);

    // 最適化を実行
    OptimizationResult result = optimizer_->minimize(objective, options);

    // 結果を検証
    SCOPED_TRACE("Test case: " + test_name);

    // 結果の抽出
    auto actual_converged = result.converged;
    auto actual_iterations = result.iterations;
    auto actual_optimal_value = result.optimal_value;
    auto actual_optimal_parameters = result.optimal_parameters;
    auto actual_message = result.message;

    // 収束した場合
    if (expected_converged)
    {
      // 理論値との比較
      // Dual Annealingはグローバル最適化アルゴリズムなので、理論値との比較

      // 収束判定の比較
      if (actual_converged != expected_converged)
      {
        // 目的関数値の比較
        EXPECT_NEAR(actual_optimal_value, theoretical_value, 1e-3)
            << "Optimal value mismatch for " << test_name;

        // パラメータ値の比較
        for (int i = 0; i < actual_optimal_parameters.size(); ++i)
        {
          EXPECT_NEAR(actual_optimal_parameters(i), theoretical_params(i), 1e-1)
              << "Parameter " << i << " mismatch for " << test_name;
        }

        std::cout << "  =================================" << std::endl;
        std::cout << "  [Warning] Convergence mismatch for " << test_name << std::endl;
        std::cout << std::endl;
      }
      else
      {
        // 目的関数値の比較
        EXPECT_NEAR(actual_optimal_value, theoretical_value, 1e-3)
            << "Optimal value mismatch for " << test_name;

        // パラメータ値の比較
        for (int i = 0; i < actual_optimal_parameters.size(); ++i)
        {
          EXPECT_NEAR(actual_optimal_parameters(i), theoretical_params(i), 1e-1)
              << "Parameter " << i << " mismatch for " << test_name;
        }
      }
    }
    // 収束しなかった場合
    else
    {
      // 値が改善されていることを確認
      EXPECT_LE(result.optimal_value, initial_value)
          << "Value did not improve for " << test_name;

      // 期待値との比較
      // EXPECT_NEAR(actual_optimal_value, expected_value, 1e-1)
      //     << "Optimal value mismatch for " << test_name;

      // パラメータ値の比較
      // for (int i = 0; i < actual_optimal_parameters.size(); ++i)
      // {
      //   EXPECT_NEAR(actual_optimal_parameters(i), expected_params(i), 1e-1)
      //       << "Parameter " << i << " mismatch for " << test_name;
      // }

      // 参考情報の出力（デバッグ用）
      std::cout << "  =================================" << std::endl;
      std::cout << "  [Ref] Test Case: " << test_name << std::endl;
      std::cout << "  [Ref] Theoretical Value: " << theoretical_value << std::endl;
      std::cout << "  [Ref] Python Result: " << expected_value << std::endl;
      std::cout << "  [Ref] C++ Result:    " << result.optimal_value << std::endl;
      std::cout << std::endl;
    }
  }
}

/**
 * @brief 低次元のテストケースのみを実行（高速なテスト）
 */
TEST_F(DualAnnealingTest, TestLowDimensionalCases)
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

    // 初期値
    double initial_value = objective(initial_params);

    // オプション設定
    DualAnnealingOptions options;
    options.max_iterations = test_case["options"]["max_iterations"].get<int>();

    options.initial_params = initial_params;

    // 境界条件がある場合は設定
    if (test_case.contains("bounds"))
    {
      VectorXd lower = jsonToVector(test_case["bounds"]["lower"]);
      VectorXd upper = jsonToVector(test_case["bounds"]["upper"]);
      options.bounds = Bounds(lower, upper);
    }
    else
    {
      // 境界がない場合、初期点を中心に広い範囲の境界を設定
      // 初期点の各次元に対して、±50の範囲を設定
      VectorXd lower(dimension);
      VectorXd upper(dimension);
      for (int i = 0; i < dimension; ++i)
      {
        lower(i) = initial_params(i) - 50.0;
        upper(i) = initial_params(i) + 50.0;
      }
      options.bounds = Bounds(lower, upper);
    }

    // Dual Annealing専用オプション
    const auto &da_opts = test_case["da_options"];
    options.initial_temp = da_opts["initial_temp"].get<double>();
    options.visit_param = da_opts["visit_param"].get<double>();
    options.accept_param = da_opts["accept_param"].get<double>();
    options.restart_temp_ratio = da_opts["restart_temp_ratio"].get<double>();
    options.max_function_evaluations = da_opts["max_function_evaluations"].get<int>();

    // シード値がある場合は設定
    if (test_case.contains("seed") && !test_case["seed"].is_null())
    {
      options.seed = test_case["seed"].get<unsigned int>();
    }

    // 期待値 (Pythonの結果)
    bool expected_converged = test_case["expected_result"]["converged"].get<bool>();
    double expected_value = test_case["expected_result"]["optimal_value"].get<double>();
    VectorXd expected_params = jsonToVector(test_case["expected_result"]["optimal_parameters"]);

    // 理論値
    double theoretical_value = test_case["function"]["optimal_value"].get<double>();
    VectorXd theoretical_params = jsonToVector(test_case["function"]["optimal_point"]);

    // 最適化を実行
    OptimizationResult result = optimizer_->minimize(objective, options);

    // 結果を検証
    SCOPED_TRACE("Test case: " + test_name);

    // 結果の抽出
    auto actual_converged = result.converged;
    auto actual_iterations = result.iterations;
    auto actual_optimal_value = result.optimal_value;
    auto actual_optimal_parameters = result.optimal_parameters;
    auto actual_message = result.message;

    // 収束した場合
    if (expected_converged)
    {
      // 理論値との比較
      // Dual Annealingはグローバル最適化アルゴリズムなので、理論値との比較

      // 収束判定の比較
      if (actual_converged != expected_converged)
      {
        // 目的関数値の比較
        EXPECT_NEAR(actual_optimal_value, theoretical_value, 1e-3)
            << "Optimal value mismatch for " << test_name;

        // パラメータ値の比較
        for (int i = 0; i < actual_optimal_parameters.size(); ++i)
        {
          EXPECT_NEAR(actual_optimal_parameters(i), theoretical_params(i), 1e-1)
              << "Parameter " << i << " mismatch for " << test_name;
        }

        std::cout << "  =================================" << std::endl;
        std::cout << "  [Warning] Convergence mismatch for " << test_name << std::endl;
        std::cout << std::endl;
      }
      else
      {
        // 目的関数値の比較
        EXPECT_NEAR(actual_optimal_value, theoretical_value, 1e-3)
            << "Optimal value mismatch for " << test_name;

        // パラメータ値の比較
        for (int i = 0; i < actual_optimal_parameters.size(); ++i)
        {
          EXPECT_NEAR(actual_optimal_parameters(i), theoretical_params(i), 1e-1)
              << "Parameter " << i << " mismatch for " << test_name;
        }
      }
    }
    // 収束しなかった場合
    else
    {
      // 値が改善されていることを確認
      EXPECT_LE(result.optimal_value, initial_value)
          << "Value did not improve for " << test_name;

      // 期待値との比較
      // EXPECT_NEAR(actual_optimal_value, expected_value, 1e-1)
      //     << "Optimal value mismatch for " << test_name;

      // パラメータ値の比較
      // for (int i = 0; i < actual_optimal_parameters.size(); ++i)
      // {
      //   EXPECT_NEAR(actual_optimal_parameters(i), expected_params(i), 1e-1)
      //       << "Parameter " << i << " mismatch for " << test_name;
      // }

      // 参考情報の出力（デバッグ用）
      std::cout << "  =================================" << std::endl;
      std::cout << "  [Ref] Test Case: " << test_name << std::endl;
      std::cout << "  [Ref] Theoretical Value: " << theoretical_value << std::endl;
      std::cout << "  [Ref] Python Result: " << expected_value << std::endl;
      std::cout << "  [Ref] C++ Result:    " << result.optimal_value << std::endl;
      std::cout << std::endl;
    }
  }
}

/**
 * @brief 高次元のテストケース（値が改善されていることを確認）
 */
TEST_F(DualAnnealingTest, TestHighDimensionalCases)
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
    DualAnnealingOptions options;
    options.max_iterations = test_case["options"]["max_iterations"].get<int>();

    options.initial_params = initial_params;

    // 境界条件がある場合は設定
    if (test_case.contains("bounds"))
    {
      VectorXd lower = jsonToVector(test_case["bounds"]["lower"]);
      VectorXd upper = jsonToVector(test_case["bounds"]["upper"]);
      options.bounds = Bounds(lower, upper);
    }
    else
    {
      // 境界がない場合、初期点を中心に広い範囲の境界を設定
      // 初期点の各次元に対して、±50の範囲を設定
      VectorXd lower(dimension);
      VectorXd upper(dimension);
      for (int i = 0; i < dimension; ++i)
      {
        lower(i) = initial_params(i) - 50.0;
        upper(i) = initial_params(i) + 50.0;
      }
      options.bounds = Bounds(lower, upper);
    }

    // Dual Annealing専用オプション
    const auto &da_opts = test_case["da_options"];
    options.initial_temp = da_opts["initial_temp"].get<double>();
    options.visit_param = da_opts["visit_param"].get<double>();
    options.accept_param = da_opts["accept_param"].get<double>();
    options.restart_temp_ratio = da_opts["restart_temp_ratio"].get<double>();
    options.max_function_evaluations = da_opts["max_function_evaluations"].get<int>();

    // シード値がある場合は設定
    if (test_case.contains("seed") && !test_case["seed"].is_null())
    {
      options.seed = test_case["seed"].get<unsigned int>();
    }

    // 期待値 (Pythonの結果)
    bool expected_converged = test_case["expected_result"]["converged"].get<bool>();
    double expected_value = test_case["expected_result"]["optimal_value"].get<double>();
    VectorXd expected_params = jsonToVector(test_case["expected_result"]["optimal_parameters"]);

    // 理論上の最適値を取得
    double theoretical_value = test_case["function"]["optimal_value"].get<double>();
    VectorXd theoretical_params = jsonToVector(test_case["function"]["optimal_point"]);

    // 最適化を実行
    OptimizationResult result = optimizer_->minimize(objective, options);

    // 結果を検証
    SCOPED_TRACE("Test case: " + test_name);

    // 結果の抽出
    auto actual_converged = result.converged;
    auto actual_iterations = result.iterations;
    auto actual_optimal_value = result.optimal_value;
    auto actual_optimal_parameters = result.optimal_parameters;
    auto actual_message = result.message;

    // 収束した場合
    if (expected_converged)
    {
      // 理論値との比較
      // Dual Annealingはグローバル最適化アルゴリズムなので、理論値との比較

      // 収束判定の比較
      if (actual_converged != expected_converged)
      {
        // 目的関数値の比較
        EXPECT_NEAR(actual_optimal_value, theoretical_value, 1e-3)
            << "Optimal value mismatch for " << test_name;

        // パラメータ値の比較
        for (int i = 0; i < actual_optimal_parameters.size(); ++i)
        {
          EXPECT_NEAR(actual_optimal_parameters(i), theoretical_params(i), 1e-1)
              << "Parameter " << i << " mismatch for " << test_name;
        }

        std::cout << "  =================================" << std::endl;
        std::cout << "  [Warning] Convergence mismatch for " << test_name << std::endl;
        std::cout << std::endl;
      }
      else
      {
        // 目的関数値の比較
        EXPECT_NEAR(actual_optimal_value, theoretical_value, 1e-3)
            << "Optimal value mismatch for " << test_name;

        // パラメータ値の比較
        for (int i = 0; i < actual_optimal_parameters.size(); ++i)
        {
          EXPECT_NEAR(actual_optimal_parameters(i), theoretical_params(i), 1e-1)
              << "Parameter " << i << " mismatch for " << test_name;
        }
      }
    }
    // 収束しなかった場合
    else
    {
      // 値が改善されていることを確認
      EXPECT_LE(result.optimal_value, initial_value)
          << "Value did not improve for " << test_name;

      // 期待値との比較
      // EXPECT_NEAR(actual_optimal_value, expected_value, 1e-1)
      //     << "Optimal value mismatch for " << test_name;

      // パラメータ値の比較
      // for (int i = 0; i < actual_optimal_parameters.size(); ++i)
      // {
      //   EXPECT_NEAR(actual_optimal_parameters(i), expected_params(i), 1e-1)
      //       << "Parameter " << i << " mismatch for " << test_name;
      // }

      // 参考情報の出力（デバッグ用）
      std::cout << "  =================================" << std::endl;
      std::cout << "  [Ref] Test Case: " << test_name << std::endl;
      std::cout << "  [Ref] Theoretical Value: " << theoretical_value << std::endl;
      std::cout << "  [Ref] Python Result: " << expected_value << std::endl;
      std::cout << "  [Ref] C++ Result:    " << result.optimal_value << std::endl;
      std::cout << std::endl;
    }
  }
}

/**
 * @brief 境界条件のテスト
 */
TEST_F(DualAnnealingTest, TestWithBounds)
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
    DualAnnealingOptions options;
    options.max_iterations = test_case["options"]["max_iterations"].get<int>();

    options.initial_params = initial_params;

    // 境界条件を設定
    VectorXd lower = jsonToVector(test_case["bounds"]["lower"]);
    VectorXd upper = jsonToVector(test_case["bounds"]["upper"]);
    options.bounds = Bounds(lower, upper);

    // Dual Annealing専用オプション
    const auto &da_opts = test_case["da_options"];
    options.initial_temp = da_opts["initial_temp"].get<double>();
    options.visit_param = da_opts["visit_param"].get<double>();
    options.accept_param = da_opts["accept_param"].get<double>();
    options.restart_temp_ratio = da_opts["restart_temp_ratio"].get<double>();
    options.max_function_evaluations = da_opts["max_function_evaluations"].get<int>();

    // シード値がある場合は設定
    if (test_case.contains("seed") && !test_case["seed"].is_null())
    {
      options.seed = test_case["seed"].get<unsigned int>();
    }

    // 最適化を実行
    OptimizationResult result = optimizer_->minimize(objective, options);

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
