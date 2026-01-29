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
     * @struct AlgorithmParameters
     * @brief アルゴリズムパラメータ
     *
     * Nelder-Mead法のアルゴリズムパラメータを保持する構造体です。
     * 標準的な値（Nelder & Mead, 1965）がデフォルトとして設定されています。
     */
    struct AlgorithmParameters
    {
      double rho = 1.0;   ///< 収縮係数
      double chi = 2.0;   ///< 反射係数
      double psi = 0.5;   ///< 拡張係数
      double sigma = 0.5; ///< 縮小係数
    };

    /**
     * @struct NelderMeadOptions
     * @brief Nelder-Mead法専用のオプション設定
     *
     * Nelder-Mead法のハイパーパラメータを設定するための構造体です。
     * 標準的な値（Nelder & Mead, 1965）がデフォルトとして設定されています。
     */
    struct NelderMeadOptions
    {
      std::optional<int> max_iterations;    ///< 最大反復回数
      std::optional<Bounds> bounds;         ///< 境界条件（オプション、未指定の場合は制約なし）
      Eigen::VectorXd initial_params;       ///< 初期パラメータ値
      AlgorithmParameters algorithm_params; ///< アルゴリズムパラメータ
      double xatol = 1e-4;                  ///< パラメータの収束判定の許容誤差
      double fatol = 1e-4;                  ///< 目的関数の収束判定の許容誤差
      std::optional<int> max_fun_evals;     ///< 最大関数評価回数
      bool is_adaptive = false;             ///< 適応的なシンプレックス生成を使用するかどうか (これをtrueにすると、反射、拡張、収縮、縮小のステップサイズを適応的に調整します)
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
    class NelderMeadOptimizer : public Optimizer<NelderMeadOptions>
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
          const NelderMeadOptions &options = NelderMeadOptions{}) override
      {
        int dim = static_cast<int>(options.initial_params.size());
        if (dim == 0)
        {
          OptimizationResult result;
          result.optimal_parameters = options.initial_params;
          result.optimal_value = objective(options.initial_params);
          result.converged = false;
          result.iterations = 0;
          result.message = "Invalid dimension: dimension must be > 0";
          return result;
        }

        // ------------------------------------------------------------
        // 設定とパラメータの準備
        // ------------------------------------------------------------

        // 適応的なアルゴリズムパラメータを生成
        AlgorithmParameters algo_params = options.algorithm_params;
        if (options.is_adaptive)
        {
          algo_params = generateAdaptiveParams(dim);
        }

        // 最大反復回数と最大関数評価回数の設定 (SciPyのデフォルト準拠: N * 200)
        int max_iter = options.max_iterations.has_value() ? options.max_iterations.value() : (dim * 200);
        int max_fun = options.max_fun_evals.has_value() ? options.max_fun_evals.value() : (dim * 200);

        // 関数評価回数カウンタ付きのラッパー
        int func_evals = 0;
        auto wrapped_objective = [&](const Eigen::VectorXd &x) -> double
        {
          func_evals++;
          return objective(x);
        };

        // ------------------------------------------------------------
        // 初期化処理
        // ------------------------------------------------------------

        // 初期パラメータ
        Eigen::VectorXd x0 = options.initial_params;

        // 境界条件のチェック
        if (options.bounds.has_value())
        {
          // 境界条件が下限が上限より大きいかチェック
          if ((options.bounds.value().lower.array() > options.bounds.value().upper.array()).any())
          {
            throw std::invalid_argument("One of the lower bounds is greater than an upper bound.");
          }
          // 初期パラメータが境界条件内にあるかチェック
          // if (!checkBounds(x0, options.bounds.value()))
          // {
          //   throw std::invalid_argument("Initial parameters are not within the bounds.");
          // }
          // 初期パラメータを境界条件にクリップ
          x0 = clipToBounds(x0, options.bounds.value());
        }

        // 初期シンプレックスを生成
        std::vector<Vertex> simplex = initializeSimplex(x0, options.bounds);

        // 初期シンプレックスの各頂点での関数値を評価
        for (auto &vertex : simplex)
        {
          vertex.value = wrapped_objective(vertex.point);

          if (func_evals >= max_fun)
          {
            break;
          }
        }

        // 初期ソート
        sortVertices(simplex);

        // ------------------------------------------------------------
        // メインループ
        // ------------------------------------------------------------
        int iterations = 1;
        bool converged = false;
        std::string message = "";

        while (func_evals < max_fun && iterations <= max_iter)
        {
          // 収束判定
          if (isConverged(simplex, options.xatol, options.fatol))
          {
            converged = true;
            message = "Optimization terminated successfully.";
            break;
          }

          // 重心を計算（最悪点を除く）
          Eigen::VectorXd xbar = computeCentroid(simplex);

          // 参照用の最悪点、次点、最良点を取得
          // simplexはソート済みなので、
          // 最良点はsimplex[0]、次点はsimplex[dim - 2]、最悪点はsimplex[dim]で取得できる
          const double f_best = simplex[0].value;
          const double f_second_worst = simplex[simplex.size() - 2].value;
          const double f_worst = simplex.back().value;

          const Eigen::VectorXd &x_worst = simplex.back().point;

          bool do_shrink = false;

          // 反射 (Reflection)
          // xr = (1 + rho) * xbar - rho * x_worst
          Eigen::VectorXd xr = (1.0 + algo_params.rho) * xbar - algo_params.rho * x_worst;

          if (options.bounds.has_value())
          {
            xr = clipToBounds(xr, options.bounds.value());
          }

          double f_xr = wrapped_objective(xr);
          if (func_evals >= max_fun)
          {
            break;
          }

          if (f_xr < f_best)
          {
            // 拡張 (Expansion)
            // xe = (1 + rho * chi) * xbar - rho * chi * x_worst
            Eigen::VectorXd xe = (1.0 + algo_params.rho * algo_params.chi) * xbar - algo_params.rho * algo_params.chi * x_worst;

            if (options.bounds.has_value())
            {
              xe = clipToBounds(xe, options.bounds.value());
            }

            double f_xe = wrapped_objective(xe);
            if (func_evals >= max_fun)
            {
              break;
            }

            if (f_xe < f_xr)
            {
              simplex.back() = Vertex(xe, f_xe);
            }
            else
            {
              simplex.back() = Vertex(xr, f_xr);
            }
          }
          else // f_xr >= f_best
          {
            if (f_xr < f_second_worst)
            {
              // 反射を採用
              simplex.back() = Vertex(xr, f_xr);
            }
            else // f_xr >= f_second_worst
            {
              // 収縮 (Contraction)
              if (f_xr < f_worst)
              {
                // 外部収縮 (Outside Contraction)
                // xc = (1 + psi * rho) * xbar - psi * rho * x_worst
                Eigen::VectorXd xc = (1.0 + algo_params.rho * algo_params.psi) * xbar - algo_params.rho * algo_params.psi * x_worst;

                if (options.bounds.has_value())
                {
                  xc = clipToBounds(xc, options.bounds.value());
                }

                double f_xc = wrapped_objective(xc);
                if (func_evals >= max_fun)
                {
                  break;
                }

                if (f_xc <= f_xr)
                {
                  simplex.back() = Vertex(xc, f_xc);
                }
                else
                {
                  do_shrink = true;
                }
              }
              else
              {
                // 内部収縮 (Inside Contraction)
                // xcc = (1 - psi) * xbar + psi * x_worst
                Eigen::VectorXd xcc = (1.0 - algo_params.psi) * xbar + algo_params.psi * x_worst;

                if (options.bounds.has_value())
                {
                  xcc = clipToBounds(xcc, options.bounds.value());
                }

                double f_xcc = wrapped_objective(xcc);
                if (func_evals >= max_fun)
                {
                  break;
                }

                if (f_xcc < f_worst)
                {
                  simplex.back() = Vertex(xcc, f_xcc);
                }
                else
                {
                  do_shrink = true;
                }
              }

              // 縮小 (Shrink)
              if (do_shrink)
              {
                const Eigen::VectorXd x0_shrink = simplex[0].point;
                for (size_t i = 1; i < simplex.size(); ++i)
                {
                  simplex[i].point = x0_shrink + algo_params.sigma * (simplex[i].point - x0_shrink);

                  if (options.bounds.has_value())
                  {
                    simplex[i].point = clipToBounds(simplex[i].point, options.bounds.value());
                  }

                  simplex[i].value = wrapped_objective(simplex[i].point);
                  if (func_evals >= max_fun)
                  {
                    break;
                  }
                }
              }
            }
          }

          iterations++;

          // 次の反復のためにシンプレックスをソート
          sortVertices(simplex);
        }

        // ------------------------------------------------------------
        // 結果の返却
        // ------------------------------------------------------------
        if (!converged)
        {
          if (func_evals >= max_fun)
          {
            message = "Maximum number of function evaluations has been exceeded.";
          }
          else if (iterations >= max_iter)
          {
            message = "Maximum number of iterations has been exceeded.";
          }
        }

        sortVertices(simplex);

        OptimizationResult result;
        result.optimal_parameters = simplex[0].point;
        result.optimal_value = simplex[0].value;
        result.converged = converged;
        result.iterations = iterations;
        result.message = message;

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
       * @brief 適応的なアルゴリズムパラメータを生成
       *
       * 次元に応じて適応的なアルゴリズムパラメータを生成します。
       *
       * @param dimension 次元数
       *
       * @return AlgorithmParameters 適応的なアルゴリズムパラメータ
       */
      AlgorithmParameters generateAdaptiveParams(int dimension) const
      {
        double dim = static_cast<double>(dimension);

        AlgorithmParameters params;
        params.rho = 1.0;
        params.chi = 1.0 + 2 / dimension;
        params.psi = 0.75 - 1 / (2 * dimension);
        params.sigma = 1 - 1 / dimension;
        return params;
      }

      /**
       * @brief 初期シンプレックスを生成
       *
       * 初期点からn+1個の頂点からなるシンプレックスを生成します。
       *
       * @param x0_clipped 初期点。シンプレックスの最初の頂点として使用されます。
       * @param bounds 境界条件。シンプレックスの頂点がこの範囲内に収まるようにクリップされます。
       *
       * @return std::vector<Vertex> 生成されたシンプレックスの頂点リスト。
       *         サイズはn+1（nは初期点の次元数）です。
       *
       * @note 生成された頂点の関数値は初期化時に0.0に設定されますが、
       *       実際の関数値は後で評価されます。
       */
      std::vector<Vertex> initializeSimplex(
          const Eigen::VectorXd &x0_clipped,
          const std::optional<Bounds> &bounds) const
      {
        Eigen::Index dim = x0_clipped.size();
        std::vector<Vertex> simplex;
        simplex.reserve(dim + 1);

        // 最初の頂点
        simplex.emplace_back(x0_clipped, 0.0); // 値は後で評価

        // scipyの実装に合わせた定数
        const double nonzdelt = 0.05; // 非ゼロ値に対する増分率（5%）
        const double zdelt = 0.00025; // ゼロ値に対する絶対値

        for (Eigen::Index k = 0; k < dim; ++k)
        {
          Eigen::VectorXd y = x0_clipped;
          if (y(k) != 0.0)
          {
            y(k) = (1.0 + nonzdelt) * y(k);
          }
          else
          {
            y(k) = zdelt;
          }

          // SciPyの境界処理ロジック (Bounds adjustment)
          // 1. 生成された点が上限を超えている場合、領域内に反射させる
          // 2. その後、上下限でクリップする
          if (bounds.has_value())
          {
            const auto &b = bounds.value();
            if (y(k) > b.upper(k))
            {
              y(k) = 2.0 * b.upper(k) - y(k);
            }
            // 最終的なクリップ
            y = clipToBounds(y, b);
          }

          simplex.emplace_back(y, 0.0);
        }

        return simplex;
      }

      /**
       * @brief 頂点を関数値で並び替え（最良→最悪の順）
       *
       * シンプレックスの頂点を目的関数値の昇順で並び替えます。
       * 並び替え後、以下のように分類されます：
       * - 最良点（simplex[0]）: 最小の目的関数値を持つ頂点
       * - 次点（simplex[dim-1]）: 2番目に小さい目的関数値を持つ頂点
       * - 最悪点（simplex[dim]）: 最大の目的関数値を持つ頂点
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
       * x_c = (1/dim) * Σ(i≠w) x_i
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
        Eigen::Index dim = simplex[0].point.size();
        Eigen::VectorXd centroid = Eigen::VectorXd::Zero(dim);

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
       *
       * @param simplex シンプレックスの頂点リスト。この関数はリストを変更しません。
       *                頂点は目的関数値の昇順で並び替えられていることを前提とします。
       * @param xatol パラメータの収束判定の許容誤差。OptimizerOptionsで指定された値が使用されます。
       * @param fatol 目的関数の収束判定の許容誤差。OptimizerOptionsで指定された値が使用されます。
       *
       * @return bool 収束した場合true、そうでない場合false
       */
      bool isConverged(const std::vector<Vertex> &simplex, double xatol, double fatol) const
      {
        const Eigen::VectorXd &best_point = simplex[0].point;
        double best_value = simplex[0].value;

        // パラメータベースの収束判定 (L-infinity norm)
        double max_param_diff = 0.0;
        for (size_t i = 1; i < simplex.size(); ++i)
        {
          // 各次元ごとの絶対差分の最大値を計算
          double diff = (simplex[i].point - best_point).cwiseAbs().maxCoeff();

          if (diff > max_param_diff)
          {
            max_param_diff = diff;
          }
        }

        // 関数値ベースの収束判定
        double max_value_diff = 0.0;
        for (size_t i = 1; i < simplex.size(); ++i)
        {
          double diff = std::abs(simplex[i].value - best_value);
          if (diff > max_value_diff)
          {
            max_value_diff = diff;
          }
        }

        // 両方の条件を満たす必要がある
        return (max_param_diff <= xatol) && (max_value_diff <= fatol);
      }
    };
  }
}