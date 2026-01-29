/**
 * @file optimizer.hpp
 * @brief 最適化処理のインターフェースクラスと関連構造体の定義
 */

#pragma once

#include <functional>
#include <vector>
#include <optional>
#include <Eigen/Dense>

namespace gprcpp
{
  namespace optimize
  {
    /**
     * @struct OptimizationResult
     * @brief 最適化処理の結果を保持する構造体
     */
    struct OptimizationResult
    {
      Eigen::VectorXd optimal_parameters; ///< 最適化されたパラメータ値
      double optimal_value;               ///< 最適化された目的関数の値（最小値）
      bool converged;                     ///< 収束したかどうか
      int iterations;                     ///< 実行された反復回数
      std::string message;                ///< 最適化処理の状態メッセージ
    };

    /**
     * @struct Bounds
     * @brief パラメータの境界条件を保持する構造体
     *
     * 各パラメータの下限と上限を定義します。
     * 制約付き最適化問題で使用されます。
     */
    struct Bounds
    {
      Eigen::VectorXd lower; ///< 各パラメータの下限値
      Eigen::VectorXd upper; ///< 各パラメータの上限値

      /**
       * @brief 境界条件を構築する
       * @param lower 各パラメータの下限値
       * @param upper 各パラメータの上限値
       */
      Bounds(const Eigen::VectorXd &lower, const Eigen::VectorXd &upper)
          : lower(lower), upper(upper)
      {
      }
    };

    /**
     * @typedef ObjectiveFunction
     * @brief 目的関数の型エイリアス
     *
     * パラメータベクトルを受け取り、目的関数の値を返す関数オブジェクトの型です。
     * 最適化アルゴリズムはこの関数を最小化します。
     */
    using ObjectiveFunction = std::function<double(const Eigen::VectorXd &)>;

    /**
     * @class Optimizer
     * @brief 最適化処理のインターフェースクラス
     *
     * @tparam T 最適化オプションの型。派生クラスで指定する。
     *
     * 各種最適化アルゴリズム（Nelder-Mead、Dual Annealingなど）の
     * 基底クラスとして使用されます。
     *
     * @note 派生クラスは minimize() メソッドを実装する必要があります。
     */
    template <typename T>
    class Optimizer
    {
    public:
      /**
       * @brief 仮想デストラクタ
       */
      virtual ~Optimizer() = default;

      /**
       * @brief 目的関数を最小化する（純粋仮想関数）
       *
       * 指定された目的関数を最小化し、最適なパラメータ値を探索します。
       *
       * @param objective 最小化する目的関数
       * @param options 最適化オプション（最大反復回数、許容誤差など）
       * @return OptimizationResult 最適化結果
       */
      virtual OptimizationResult minimize(
          const ObjectiveFunction &objective,
          const T &options) = 0;

    protected:
      /**
       * @brief パラメータが境界条件を満たしているかチェックする
       *
       * 派生クラスで境界条件の検証に使用できます。
       *
       * @param params チェックするパラメータベクトル
       * @param bounds 境界条件
       * @return true すべてのパラメータが境界内にある場合
       * @return false 境界外のパラメータが存在する場合、またはサイズが一致しない場合
       */
      bool checkBounds(const Eigen::VectorXd &params, const Bounds &bounds) const
      {
        if (params.size() != bounds.lower.size() || params.size() != bounds.upper.size())
        {
          return false;
        }
        for (int i = 0; i < params.size(); ++i)
        {
          if (params(i) < bounds.lower(i) || params(i) > bounds.upper(i))
          {
            return false;
          }
        }
        return true;
      }

      /**
       * @brief パラメータを境界内に制限する
       *
       * 境界外のパラメータ値を境界内にクリップします。
       * 派生クラスで境界制約の処理に使用できます。
       *
       * @param params クリップするパラメータベクトル
       * @param bounds 境界条件
       * @return Eigen::VectorXd 境界内に制限されたパラメータベクトル
       */
      Eigen::VectorXd clipToBounds(const Eigen::VectorXd &params, const Bounds &bounds) const
      {
        Eigen::VectorXd clipped = params;
        for (int i = 0; i < params.size(); ++i)
        {
          clipped(i) = std::max(bounds.lower(i), std::min(params(i), bounds.upper(i)));
        }
        return clipped;
      }
    };

  }
}