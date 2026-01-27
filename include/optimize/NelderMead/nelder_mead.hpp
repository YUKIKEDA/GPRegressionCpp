/**
 * @file nelder_mead.hpp
 * @brief Nelder-Mead最適化アルゴリズムの実装
 *
 * このファイルは、Nelder-Mead法（シンプレックス法、Downhill Simplex法）の
 * 実装を提供します。導関数を必要としない直接探索型の最適化アルゴリズムで、
 * 目的関数の勾配情報が利用できない場合や、目的関数が非滑らかである場合に
 * 特に有効です。
 *
 * @author GPRegressionCpp
 * @date 2026-01-27
 */

#pragma once

#include "optimize/optimizer.hpp"
#include <vector>
#include <algorithm>
#include <cmath>

namespace gprcpp
{
  namespace optimize
  {
    /**
     * @struct NelderMeadOptions
     * @brief Nelder-Mead法専用のオプション設定
     *
     * Nelder-Mead法のハイパーパラメータを設定するための構造体です。
     * 標準的な値（Nelder & Mead, 1965）がデフォルトとして設定されています。
     */
    struct NelderMeadOptions
    {
      /**
       * @brief 反射係数（α）
       *
       * 最悪点を重心の反対側に反射する際の係数です。
       * 標準値は1.0で、反射点は以下の式で計算されます：
       * x_r = x_c + α * (x_c - x_w)
       */
      double alpha = 1.0;

      /**
       * @brief 拡張係数（γ）
       *
       * 反射が成功した場合に、さらに遠くまで探索する際の係数です。
       * 標準値は2.0で、拡張点は以下の式で計算されます：
       * x_e = x_c + γ * (x_r - x_c)
       */
      double gamma = 2.0;

      /**
       * @brief 収縮係数（ρ）
       *
       * 外側収縮または内側収縮を行う際の係数です。
       * 標準値は0.5で、以下の式で計算されます：
       * - 外側収縮: x_oc = x_c + ρ * (x_r - x_c)
       * - 内側収縮: x_ic = x_c - ρ * (x_c - x_w)
       */
      double rho = 0.5;

      /**
       * @brief 縮小係数（σ）
       *
       * シンプレックス全体を最良点に向かって縮小する際の係数です。
       * 標準値は0.5で、縮小後の点は以下の式で計算されます：
       * x_i = x_b + σ * (x_i - x_b)
       */
      double sigma = 0.5;
    };

    /**
     * @class NelderMeadOptimizer
     * @brief Nelder-Mead最適化アルゴリズムの実装クラス
     *
     * Nelder-Mead法は、1965年にJohn NelderとRoger Meadによって提案された、
     * 導関数を必要としない直接探索型の最適化アルゴリズムです。
     * シンプレックス（単体）を操作して最適解を探索します。
     *
     * @see Optimizer
     * @see OptimizationResult
     * @see OptimizerOptions
     * @see NelderMeadOptions
     */
    class NelderMeadOptimizer : public Optimizer
    {
    public:
      /**
       * @brief 目的関数を最小化する
       *
       * Nelder-Mead法を使用して、指定された目的関数を最小化します。
       * シンプレックスを操作しながら最適解を探索し、収束条件を満たすか
       * 最大反復回数に達するまで反復処理を続けます。
       *
       * @param objective 最小化する目的関数。パラメータベクトルを受け取り、
       *                  目的関数の値を返す関数オブジェクト
       * @param initial_params 初期パラメータ値。探索の開始点として使用されます。
       *                       各次元の値に基づいて初期シンプレックスが生成されます。
       * @param options 最適化オプション。以下の設定が可能です：
       *                - max_iterations: 最大反復回数（デフォルト: 1000）
       *                - tolerance: 収束判定の許容誤差（デフォルト: 1e-6）
       *                - bounds: パラメータの境界条件（オプション）
       *                - verbose: 詳細な出力を行うかどうか（デフォルト: false）
       *
       * @return OptimizationResult 最適化結果。以下の情報を含みます：
       *         - optimal_parameters: 最適化されたパラメータ値
       *         - optimal_value: 最適化された目的関数の値（最小値）
       *         - converged: 収束したかどうか
       *         - iterations: 実行された反復回数
       *         - message: 最適化処理の状態メッセージ
       *
       * @note 次元が0の場合、エラーメッセージと共に結果を返します。
       * @note 初期ステップサイズは、各次元の5%または0.1の大きい方として自動設定されます。
       * @note 境界制約が指定されている場合、生成された点は自動的に境界内にクリップされます。
       * @note Nelder-Mead法の係数（α, γ, ρ, σ）は標準値が使用されます。
       *       カスタム係数を使用する場合は、オーバーロード版を使用してください。
       *
       * @see OptimizerOptions
       * @see OptimizationResult
       */
      OptimizationResult minimize(
          const ObjectiveFunction &objective,
          const Eigen::VectorXd &initial_params,
          const OptimizerOptions &options = OptimizerOptions{}) override
      {
        return minimize(objective, initial_params, options, NelderMeadOptions{});
      }

      /**
       * @brief 目的関数を最小化する（Nelder-Mead専用オプション付き）
       *
       * Nelder-Mead法を使用して、指定された目的関数を最小化します。
       * Nelder-Mead法のハイパーパラメータ（α, γ, ρ, σ）をカスタマイズできます。
       *
       * @param objective 最小化する目的関数。パラメータベクトルを受け取り、
       *                  目的関数の値を返す関数オブジェクト
       * @param initial_params 初期パラメータ値。探索の開始点として使用されます。
       *                       各次元の値に基づいて初期シンプレックスが生成されます。
       * @param options 最適化オプション。以下の設定が可能です：
       *                - max_iterations: 最大反復回数（デフォルト: 1000）
       *                - tolerance: 収束判定の許容誤差（デフォルト: 1e-6）
       *                - bounds: パラメータの境界条件（オプション）
       *                - verbose: 詳細な出力を行うかどうか（デフォルト: false）
       * @param nm_options Nelder-Mead法専用のオプション。以下の係数を設定できます：
       *                   - alpha: 反射係数（デフォルト: 1.0）
       *                   - gamma: 拡張係数（デフォルト: 2.0）
       *                   - rho: 収縮係数（デフォルト: 0.5）
       *                   - sigma: 縮小係数（デフォルト: 0.5）
       *
       * @return OptimizationResult 最適化結果。以下の情報を含みます：
       *         - optimal_parameters: 最適化されたパラメータ値
       *         - optimal_value: 最適化された目的関数の値（最小値）
       *         - converged: 収束したかどうか
       *         - iterations: 実行された反復回数
       *         - message: 最適化処理の状態メッセージ
       *
       * @note 次元が0の場合、エラーメッセージと共に結果を返します。
       * @note 初期ステップサイズは、各次元の5%または0.1の大きい方として自動設定されます。
       * @note 境界制約が指定されている場合、生成された点は自動的に境界内にクリップされます。
       *
       * @see OptimizerOptions
       * @see NelderMeadOptions
       * @see OptimizationResult
       */
      OptimizationResult minimize(
          const ObjectiveFunction &objective,
          const Eigen::VectorXd &initial_params,
          const OptimizerOptions &options,
          const NelderMeadOptions &nm_options)
      {
        Eigen::Index n = initial_params.size();
        if (n == 0)
        {
          OptimizationResult result;
          result.optimal_parameters = initial_params;
          result.optimal_value = objective(initial_params);
          result.converged = false;
          result.iterations = 0;
          result.message = "Invalid dimension: dimension must be > 0";
          return result;
        }

        // 初期シンプレックスを生成（scipyの実装に合わせる）
        std::vector<Vertex> simplex = initializeSimplex(initial_params);

        // 各頂点での関数値を評価
        for (auto &vertex : simplex)
        {
          vertex.value = objective(vertex.point);
        }

        // 反復処理
        for (int iteration = 0; iteration < options.max_iterations; ++iteration)
        {
          // 頂点を並び替え（最良、次点、最悪を特定）
          sortVertices(simplex);

          // 収束判定
          if (isConverged(simplex, options.tolerance))
          {
            OptimizationResult result;
            result.optimal_parameters = simplex[0].point;
            result.optimal_value = simplex[0].value;
            result.converged = true;
            result.iterations = iteration + 1;
            result.message = "Converged";
            return result;
          }

          // 重心を計算（最悪点を除く）
          Eigen::VectorXd centroid = computeCentroid(simplex);
          const Eigen::VectorXd &worst_point = simplex.back().point;
          double worst_value = simplex.back().value;
          double best_value = simplex[0].value;
          double second_value = simplex[simplex.size() - 2].value;

          // 境界制約がある場合、重心をクリップ
          if (options.bounds.has_value())
          {
            centroid = clipToBounds(centroid, options.bounds.value());
          }

          // 反射（scipyの実装に合わせる: xr = (1 + rho) * xbar - rho * worst）
          Eigen::VectorXd reflected = (1.0 + nm_options.alpha) * centroid - nm_options.alpha * worst_point;
          if (options.bounds.has_value())
          {
            reflected = clipToBounds(reflected, options.bounds.value());
          }
          double reflected_value = objective(reflected);

          if (reflected_value < best_value)
          {
            // 拡張（scipyの実装に合わせる: xe = (1 + rho * chi) * xbar - rho * chi * worst）
            Eigen::VectorXd expanded = (1.0 + nm_options.alpha * nm_options.gamma) * centroid - nm_options.alpha * nm_options.gamma * worst_point;
            if (options.bounds.has_value())
            {
              expanded = clipToBounds(expanded, options.bounds.value());
            }
            double expanded_value = objective(expanded);

            if (expanded_value < reflected_value)
            {
              simplex.back().point = expanded;
              simplex.back().value = expanded_value;
            }
            else
            {
              simplex.back().point = reflected;
              simplex.back().value = reflected_value;
            }
          }
          else if (reflected_value < second_value)
          {
            // 反射点で置換
            simplex.back().point = reflected;
            simplex.back().value = reflected_value;
          }
          else if (reflected_value < worst_value)
          {
            // 外側収縮（scipyの実装に合わせる: xc = (1 + psi * rho) * xbar - psi * rho * worst）
            Eigen::VectorXd outside_contracted = (1.0 + nm_options.rho * nm_options.alpha) * centroid - nm_options.rho * nm_options.alpha * worst_point;
            if (options.bounds.has_value())
            {
              outside_contracted = clipToBounds(outside_contracted, options.bounds.value());
            }
            double oc_value = objective(outside_contracted);

            if (oc_value <= reflected_value)
            {
              simplex.back().point = outside_contracted;
              simplex.back().value = oc_value;
            }
            else
            {
              // 縮小
              shrinkSimplex(simplex, objective, nm_options.sigma);
            }
          }
          else
          {
            // 内側収縮（scipyの実装に合わせる: xcc = (1 - psi) * xbar + psi * worst）
            Eigen::VectorXd inside_contracted = (1.0 - nm_options.rho) * centroid + nm_options.rho * worst_point;
            if (options.bounds.has_value())
            {
              inside_contracted = clipToBounds(inside_contracted, options.bounds.value());
            }
            double ic_value = objective(inside_contracted);

            if (ic_value < worst_value)
            {
              simplex.back().point = inside_contracted;
              simplex.back().value = ic_value;
            }
            else
            {
              // 縮小
              shrinkSimplex(simplex, objective, nm_options.sigma);
            }
          }
        }

        // 最大反復回数に達した
        sortVertices(simplex);
        OptimizationResult result;
        result.optimal_parameters = simplex[0].point;
        result.optimal_value = simplex[0].value;
        result.converged = false;
        result.iterations = options.max_iterations;
        result.message = "Maximum iterations reached";
        return result;
      }

    private:
      /**
       * @struct Vertex
       * @brief シンプレックスの頂点とその関数値を保持する構造体
       *
       * Nelder-Mead法では、n次元空間におけるn+1個の頂点からなる
       * シンプレックスを操作します。各頂点は位置（point）と
       * その位置での目的関数値（value）を持ちます。
       */
      struct Vertex
      {
        Eigen::VectorXd point; ///< 頂点の位置（パラメータベクトル）
        double value;          ///< 頂点での目的関数値

        /**
         * @brief 頂点を構築
         * @param p 頂点の位置
         * @param v 頂点での目的関数値
         */
        Vertex(const Eigen::VectorXd &p, double v) : point(p), value(v) {}
      };

      /**
       * @brief 初期シンプレックスを生成
       *
       * 初期点からn+1個の頂点からなるシンプレックスを生成します。
       * scipyの実装に合わせて、各次元に対して異なる方法でシンプレックスを生成します。
       *
       * 生成されるシンプレックス:
       * - 頂点0: initial
       * - 頂点1: initial の0次元目を 1.05 * initial[0] に変更（initial[0] != 0の場合）
       *          または 0.00025 に変更（initial[0] == 0の場合）
       * - 頂点2: initial の1次元目を 1.05 * initial[1] に変更（initial[1] != 0の場合）
       *          または 0.00025 に変更（initial[1] == 0の場合）
       * - ...
       * - 頂点n: initial の(n-1)次元目を 1.05 * initial[n-1] に変更（initial[n-1] != 0の場合）
       *          または 0.00025 に変更（initial[n-1] == 0の場合）
       *
       * @param initial 初期点。シンプレックスの最初の頂点として使用されます。
       *
       * @return std::vector<Vertex> 生成されたシンプレックスの頂点リスト。
       *         サイズはn+1（nは初期点の次元数）です。
       *
       * @note 生成された頂点の関数値は初期化時に0.0に設定されますが、
       *       実際の関数値は後で評価されます。
       * @note この実装はscipy.optimize.minimizeのNelder-Mead実装に基づいています。
       */
      std::vector<Vertex> initializeSimplex(
          const Eigen::VectorXd &initial) const
      {
        Eigen::Index n = initial.size();
        std::vector<Vertex> simplex;
        simplex.reserve(n + 1);

        // scipyの実装に合わせた定数
        const double nonzdelt = 0.05; // 非ゼロ値に対する増分率（5%）
        const double zdelt = 0.00025; // ゼロ値に対する絶対値

        // 最初の頂点は初期点
        simplex.emplace_back(initial, 0.0);

        // 残りのn個の頂点を生成
        for (Eigen::Index i = 0; i < n; ++i)
        {
          Eigen::VectorXd point = initial;
          if (std::abs(point(i)) > 1e-10) // 実質的に0でない場合
          {
            // 初期点の値の5%増加（1.05倍）
            point(i) = (1.0 + nonzdelt) * point(i);
          }
          else
          {
            // 初期点の値が0の場合、小さな絶対値を使用
            point(i) = zdelt;
          }
          simplex.emplace_back(point, 0.0);
        }

        return simplex;
      }

      /**
       * @brief 頂点を関数値で並び替え（最良→最悪の順）
       *
       * シンプレックスの頂点を目的関数値の昇順で並び替えます。
       * 並び替え後、以下のように分類されます：
       * - 最良点（simplex[0]）: 最小の目的関数値を持つ頂点
       * - 次点（simplex[n-1]）: 2番目に小さい目的関数値を持つ頂点
       * - 最悪点（simplex[n]）: 最大の目的関数値を持つ頂点
       *
       * @param simplex 並び替えるシンプレックスの頂点リスト（参照渡し）。
       *                この関数呼び出し後、頂点は目的関数値の昇順で並び替えられます。
       *
       * @note この操作は各反復の最初に実行され、最良点、次点、最悪点を
       *       特定するために使用されます。
       */
      void sortVertices(std::vector<Vertex> &simplex) const
      {
        std::sort(simplex.begin(), simplex.end(),
                  [](const Vertex &a, const Vertex &b)
                  { return a.value < b.value; });
      }

      /**
       * @brief 最悪点を除く全頂点の重心を計算
       *
       * シンプレックスの最悪点（目的関数値が最大の頂点）を除いた
       * 全頂点の重心を計算します。この重心は反射、拡張、収縮などの
       * 操作の基準点として使用されます。
       *
       * 重心は以下の式で計算されます：
       * x_c = (1/n) * Σ(i≠w) x_i
       *
       * ここで、nは最悪点を除く頂点の数、wは最悪点のインデックスです。
       *
       * @param simplex シンプレックスの頂点リスト。この関数はリストを変更しません。
       *                頂点は目的関数値の昇順で並び替えられていることを前提とします。
       *
       * @return Eigen::VectorXd 最悪点を除く全頂点の重心。
       *         次元数は各頂点の次元数と同じです。
       *
       * @note この関数は、simplexが空でないこと、および各頂点の次元数が
       *       一致していることを前提としています。
       */
      Eigen::VectorXd computeCentroid(const std::vector<Vertex> &simplex) const
      {
        Eigen::Index n = simplex[0].point.size();
        Eigen::VectorXd centroid = Eigen::VectorXd::Zero(n);

        // 最悪点（最後の要素）を除く全頂点の平均
        for (size_t i = 0; i < simplex.size() - 1; ++i)
        {
          centroid += simplex[i].point;
        }
        centroid /= static_cast<double>(simplex.size() - 1);

        return centroid;
      }

      /**
       * @brief 収束判定
       *
       * シンプレックスが収束したかどうかを判定します。
       * scipyの実装に合わせて、以下の2つの条件の両方が満たされた場合、収束と判定されます：
       *
       * 1. **パラメータベースの収束判定**:
       *    シンプレックスの各頂点（最良点を除く）と最良点の差の最大値が許容誤差以下
       *    max_i ||x_i - x_best|| <= tolerance
       *
       * 2. **関数値ベースの収束判定**:
       *    最良点と他の点の目的関数値の差の最大値が許容誤差以下
       *    max_i |f_i - f_best| <= tolerance
       *
       * @param simplex シンプレックスの頂点リスト。この関数はリストを変更しません。
       *                頂点は目的関数値の昇順で並び替えられていることを前提とします。
       * @param tolerance 収束判定の許容誤差。OptimizerOptionsで指定された値が使用されます。
       *
       * @return bool 収束した場合true、そうでない場合false
       *
       * @note この実装はscipy.optimize.minimizeのNelder-Mead実装に基づいています。
       */
      bool isConverged(const std::vector<Vertex> &simplex, double tolerance) const
      {
        if (simplex.size() < 2)
        {
          return true;
        }

        const Eigen::VectorXd &best_point = simplex[0].point;
        double best_value = simplex[0].value;

        // パラメータベースの収束判定: 各頂点と最良点の差の最大値
        double max_param_diff = 0.0;
        for (size_t i = 1; i < simplex.size(); ++i)
        {
          double diff = (simplex[i].point - best_point).norm();
          max_param_diff = std::max(max_param_diff, diff);
        }

        // 関数値ベースの収束判定: 最良点と他の点の関数値の差の最大値
        double max_value_diff = 0.0;
        for (size_t i = 1; i < simplex.size(); ++i)
        {
          double diff = std::abs(simplex[i].value - best_value);
          max_value_diff = std::max(max_value_diff, diff);
        }

        // 両方の条件を満たす必要がある
        return (max_param_diff <= tolerance) && (max_value_diff <= tolerance);
      }

      /**
       * @brief シンプレックスを縮小
       *
       * 反射、拡張、収縮のすべての操作が失敗した場合に、
       * シンプレックス全体を最良点に向かって縮小します。
       * これにより、探索領域を狭め、より細かい探索が可能になります。
       *
       * 縮小後の各頂点（最良点を除く）は以下の式で計算されます：
       * x_i = x_best + σ * (x_i - x_best)
       *
       * ここで、σは縮小係数、x_bestは最良点です。
       *
       * @param simplex 縮小するシンプレックスの頂点リスト（参照渡し）。
       *                この関数呼び出し後、最良点を除く全頂点が縮小され、
       *                各頂点での目的関数値が再評価されます。
       * @param objective 目的関数。縮小後の各頂点での関数値を評価するために使用されます。
       * @param sigma 縮小係数。通常は0.5が使用されます。
       *
       * @note 最良点（simplex[0]）は変更されません。
       * @note 縮小後、各頂点での目的関数値が自動的に再評価されます。
       */
      void shrinkSimplex(std::vector<Vertex> &simplex,
                         const ObjectiveFunction &objective,
                         double sigma) const
      {
        const Eigen::VectorXd &best_point = simplex[0].point;
        for (size_t i = 1; i < simplex.size(); ++i)
        {
          simplex[i].point = best_point + sigma * (simplex[i].point - best_point);
          simplex[i].value = objective(simplex[i].point);
        }
      }
    };

  }
}