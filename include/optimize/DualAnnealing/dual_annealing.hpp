/**
 * @file dual_annealing.hpp
 * @brief Dual Annealing最適化アルゴリズムの実装
 *
 * このファイルは、Dual Annealing（二重アニーリング法）の実装を提供します。
 *
 * @author GPRegressionCpp
 * @date 2026-01-28
 */

#pragma once

#include "optimize/optimizer.hpp"
#include "optimize/NelderMead/nelder_mead.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <optional>
#include <string>
#include <numbers>

namespace gprcpp
{
  namespace optimize
  {
    /**
     * @class ObjectiveFunctionWrapper
     * @brief 目的関数をラップするクラス
     *
     * 目的関数をラップし、評価回数を管理します。
     */
    class ObjectiveFunctionWrapper
    {
    public:
      /**
       * @brief コンストラクタ
       * @param func 目的関数
       * @param maxfun 最大評価回数
       */
      ObjectiveFunctionWrapper(const ObjectiveFunction &func, int maxfun)
          : func_(func), maxfun_(maxfun), num_func_eval_(0) {}

      /**
       * @brief 目的関数を評価
       * @param x 入力ベクトル
       * @return 目的関数の値
       */
      double fun(const Eigen::VectorXd &x)
      {
        ++num_func_eval_;
        return func_(x);
      }

      /**
       * @brief 評価回数を取得
       * @return 評価回数
       */
      int num_func_eval() const { return num_func_eval_; }

      /**
       * @brief 最大評価回数を取得
       * @return 最大評価回数
       */
      int maxfun() const { return maxfun_; }

    private:
      ObjectiveFunction func_; ///< 目的関数
      int maxfun_;             ///< 最大評価回数
      int num_func_eval_;      ///< 評価回数
    };

    /**
     * @class LocalSearchWrapper
     * @brief ローカルサーチ用のラッパークラス
     *
     * ローカルサーチをラップし、評価回数を管理します。
     * SciPy dual_annealing の LocalSearchWrapper に相当し、
     * デフォルトで Nelder-Mead を用いた局所探索を行います。
     */
    class LocalSearchWrapper
    {
      static constexpr int LS_MAXITER_MIN = 100;
      static constexpr int LS_MAXITER_MAX = 1000;
      static constexpr int LS_MAXITER_RATIO = 6;

    public:
      /**
       * @brief コンストラクタ
       * @param search_bounds 探索境界条件
       * @param objective_func 目的関数
       */
      LocalSearchWrapper(const Bounds &search_bounds, ObjectiveFunctionWrapper &objective_func)
          : search_bounds_(search_bounds), objective_func_(objective_func) {}

      /**
       * @brief ローカルサーチを実行
       *
       * 与えられた点 x を初期値として局所最小化を行い、有効かつ
       * エネルギーが改善された場合のみ新しい点を返す。
       *
       * @param x 初期パラメータベクトル
       * @param energy 現在の目的関数の値
       * @return ペア (最適値, 最適パラメータ)。無効または改善なしの場合は (energy, x)
       */
      std::pair<double, Eigen::VectorXd> local_search(const Eigen::VectorXd &x, double energy)
      {
        NelderMeadOptions options;
        options.bounds = search_bounds_;

        // ローカルサーチの最大反復回数を計算
        int n = static_cast<int>(x.size());
        int max_iter = std::max(LS_MAXITER_MIN, n * LS_MAXITER_RATIO);
        max_iter = std::min(max_iter, LS_MAXITER_MAX);

        options.max_iterations = max_iter;

        options.initial_params = x;

        ObjectiveFunction objective_func_unwrapped = [this](const Eigen::VectorXd &x)
        {
          return objective_func_.fun(x);
        };

        auto result = minimizer_.minimize(objective_func_unwrapped, options);

        bool is_finite = std::isfinite(result.optimal_value) && result.optimal_parameters.allFinite();
        bool is_in_bounds = (search_bounds_.lower.array() <= result.optimal_parameters.array()).all() &&
                            (result.optimal_parameters.array() <= search_bounds_.upper.array()).all();
        bool is_valid = is_finite && is_in_bounds;

        if (is_valid && result.optimal_value < energy)
        {
          return std::make_pair(result.optimal_value, result.optimal_parameters);
        }
        else
        {
          return std::make_pair(energy, x);
        }
      }

    private:
      ObjectiveFunctionWrapper &objective_func_; ///< 目的関数
      Bounds search_bounds_;                     ///< 境界条件
      NelderMeadOptimizer minimizer_;            ///< ローカルサーチのオプティマイザー
    };

    /**
     * @class EnergyState
     * @brief エネルギー状態を管理するクラス
     *
     * エネルギー状態を管理し、最適化結果を保存します。
     */
    class EnergyState
    {
      static constexpr int MAX_REINIT_COUNT = 1000;

    public:
      /**
       * @brief コンストラクタ
       * @param lower 下限
       * @param upper 上限
       */
      EnergyState(const Bounds &bounds)
          : bounds_(bounds),
            has_best_(false) {}

      /**
       * @brief エネルギー状態をリセット
       * @param objective_func 目的関数のラッパー
       * @param rng 乱数生成器
       * @param x0 初期パラメータ
       */
      void reset(ObjectiveFunctionWrapper &objective_func, std::mt19937 &rng,
                 const std::optional<Eigen::VectorXd> &x0)
      {
        // TODO : scipyの実装と同等か確認する
        // 一様分布を生成
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

        if (x0.has_value())
        {
          current_location_ = x0.value();
        }
        else
        {
          // ランダムな初期値を生成
          current_location_.resize(bounds_.lower.size());
          for (int i = 0; i < bounds_.lower.size(); ++i)
          {
            current_location_[i] = bounds_.lower[i] + (bounds_.upper[i] - bounds_.lower[i]) * uniform_dist(rng);
          }
        }

        bool init_error = true;
        int reinit_counter = 0;
        while (init_error)
        {
          current_energy_ = objective_func.fun(current_location_);

          if (std::isfinite(current_energy_))
          {
            init_error = false;
          }
          else
          {
            if (reinit_counter >= MAX_REINIT_COUNT)
            {
              init_error = false;
              std::string message = "Stopping algorithm because function create NaN or (+/-) infinity values even with trying new random parameters";
              throw std::invalid_argument(message);
            }

            // ランダムな初期値を再生成
            for (int i = 0; i < current_location_.size(); ++i)
            {
              current_location_[i] = bounds_.lower[i] + (bounds_.upper[i] - bounds_.lower[i]) * uniform_dist(rng);
            }
            reinit_counter++;
          }

          // 初回リセットまたは、最適解が未設定の場合
          if (!has_best_)
          {
            ebest_ = current_energy_;
            xbest_ = current_location_;
            has_best_ = true;
          }
        }
      }

      /**
       * @brief 最適解を更新
       * @param energy 目的関数の値
       * @param x パラメータベクトル
       * @return 最適解が更新されたかどうか (※コールバックなどで停止要求があればtrueを返す予定（今はfalseを返す）)
       */
      bool update_best(double energy, const Eigen::VectorXd &x)
      {
        ebest_ = energy;
        xbest_ = x;
        return false;
      }

      /**
       * @brief 現在のエネルギー状態を更新
       * @param energy 目的関数の値
       * @param x パラメータベクトル
       */
      void update_current(double energy, const Eigen::VectorXd &x)
      {
        current_energy_ = energy;
        current_location_ = x;
      }

      /**
       * @brief 最適解を取得
       * @return 最適解
       */
      double ebest() const { return ebest_; }

      /**
       * @brief 最適解を取得
       * @return 最適解
       */
      const Eigen::VectorXd &xbest() const { return xbest_; }

      /**
       * @brief 現在のエネルギー状態を取得
       * @return 現在のエネルギー状態
       */
      double current_energy() const { return current_energy_; }

      /**
       * @brief 現在のエネルギー状態を取得
       * @return 現在の位置
       */
      const Eigen::VectorXd &current_location() const { return current_location_; }

    private:
      Bounds bounds_;                    ///< 境界条件
      bool has_best_ = false;            ///< 最適解が存在するかどうか
      double ebest_ = 0;                 ///< 最適解のエネルギー
      Eigen::VectorXd xbest_;            ///< 最適解のパラメータ
      double current_energy_ = 0;        ///< 現在のエネルギー
      Eigen::VectorXd current_location_; ///< 現在の位置
    };

    /**
     * @class VisitingDistribution
     * @brief 訪問分布を管理するクラス
     *
     * 訪問分布を管理し、新しい位置を生成します。
     */
    class VisitingDistribution
    {
      static constexpr double TAIL_LIMIT = 1.e8;
      static constexpr double MIN_VISIT_BOUND = 1.e-10;

    public:
      /**
       * @brief コンストラクタ
       * @param lb 下限
       * @param ub 上限
       * @param visiting_param 訪問パラメータ
       * @param rng 乱数生成器
       */
      VisitingDistribution(const Bounds &bounds,
                           double visiting_param,
                           std::mt19937 &rng)
          : bounds_(bounds),
            visiting_param_(visiting_param),
            rng_(rng)
      {
        bound_range_ = bounds_.upper - bounds_.lower;

        factor2_ = std::exp((4.0 - visiting_param_) * std::log(visiting_param_ - 1.0));
        factor3_ = std::exp((2.0 - visiting_param_) * std::log(2.0) / (visiting_param_ - 1.0));
        factor4_p_ = std::sqrt(std::numbers::pi) * factor2_ / (factor3_ * (3.0 - visiting_param_));
        factor5_ = 1.0 / (visiting_param_ - 1.0) - 0.5;
        d1_ = 2.0 - factor5_;
        factor6_ = std::numbers::pi * (1.0 - factor5_) / std::sin(std::numbers::pi * (1.0 - factor5_)) / std::exp(std::lgamma(d1_));
      }

      /**
       * @brief 訪問分布を生成
       * @param x 現在の位置
       * @param step ステップ
       * @param temperature 温度
       * @return 新しい位置
       */
      Eigen::VectorXd visiting(const Eigen::VectorXd &x, int step, double temperature)
      {
        int dim = static_cast<int>(x.size());
        Eigen::VectorXd x_visit = x; // コピーしておく

        std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);

        if (step < dim)
        {
          // 全ての座標を同時に変更
          Eigen::VectorXd visits = visit_fn(temperature, dim);
          double upper_sample = dist_uniform(rng_);
          double lower_sample = dist_uniform(rng_);

          // TAIL_LIMIT のクリッピング処理
          for (int i = 0; i < dim; ++i)
          {
            if (visits[i] > TAIL_LIMIT)
            {
              visits[i] = TAIL_LIMIT * upper_sample;
            }
            else if (visits[i] < -TAIL_LIMIT)
            {
              visits[i] = -TAIL_LIMIT * lower_sample;
            }
          }

          x_visit = visits + x;

          // 境界処理 (周期境界的な処理)
          // a = x_visit - lower
          // b = fmod(a, range) + range
          // x_visit = fmod(b, range) + lower
          // Eigenのarray().fmod()等は負の値に対してプラットフォーム依存の挙動をする場合があるが、
          // SciPyのロジックを再現する。
          for (int i = 0; i < dim; ++i)
          {
            double a = x_visit[i] - bounds_.lower[i];
            double b = std::fmod(a, bound_range_[i]) + bound_range_[i];
            x_visit[i] = std::fmod(b, bound_range_[i]) + bounds_.lower[i];

            if (std::abs(x_visit[i] - bounds_.lower[i]) < MIN_VISIT_BOUND)
            {
              x_visit[i] += MIN_VISIT_BOUND;
            }
          }
        }
        else
        {
          // 1つの座標を変更

          // visit_fn は size=1 のベクトルを返す
          double visit = visit_fn(temperature, 1)[0];

          if (visit > TAIL_LIMIT)
          {
            visit = TAIL_LIMIT * dist_uniform(rng_);
          }
          else if (visit < -TAIL_LIMIT)
          {
            visit = -TAIL_LIMIT * dist_uniform(rng_);
          }

          // 変更するインデックス
          int index = step - dim;

          x_visit[index] = visit + x[index];

          // 境界処理 (単一要素)
          double a = x_visit[index] - bounds_.lower[index];
          double b = std::fmod(a, bound_range_[index]) + bound_range_[index];
          x_visit[index] = std::fmod(b, bound_range_[index]) + bounds_.lower[index];

          if (std::abs(x_visit[index] - bounds_.lower[index]) < MIN_VISIT_BOUND)
          {
            x_visit[index] += MIN_VISIT_BOUND;
          }
        }
        return x_visit;
      }

    private:
      /**
       * @brief 訪問分布を生成
       * @param temperature 温度
       * @param dim 次元
       * @return 訪問分布
       */
      Eigen::VectorXd visit_fn(double temperature, int dim)
      {
        std::normal_distribution<double> dist_normal(0.0, 1.0);

        // 正規分布を生成
        // x, y = normal(size=(dim, 2))
        Eigen::VectorXd x(dim);
        Eigen::VectorXd y(dim);
        for (int i = 0; i < dim; ++i)
        {
          x[i] = dist_normal(rng_);
          y[i] = dist_normal(rng_);
        }

        double factor1 = std::exp(std::log(temperature) / (visiting_param_ - 1.0));
        double factor4 = factor4_p_ * factor1;

        // sigmax calculation
        double term = std::exp(-(visiting_param_ - 1.0) * std::log(factor6_ / factor4) / (3.0 - visiting_param_));
        x *= term; // Vector scalar multiplication

        // den calculation
        Eigen::VectorXd den(dim);
        for (int i = 0; i < dim; ++i)
        {
          den[i] = std::exp((visiting_param_ - 1.0) * std::log(std::abs(y[i])) / (3.0 - visiting_param_));
        }

        // Element-wise division
        return x.array() / den.array();
      }

      double visiting_param_;       ///< 訪問パラメータ
      std::mt19937 &rng_;           ///< 乱数生成器
      Bounds bounds_;               ///< 境界条件
      Eigen::VectorXd bound_range_; ///< 境界条件の範囲

      // キャッシュ用定数
      double factor2_;   ///< 訪問分布のパラメータ
      double factor3_;   ///< 訪問分布のパラメータ
      double factor4_p_; ///< 訪問分布のパラメータ
      double factor5_;   ///< 訪問分布のパラメータ
      double d1_;        ///< 訪問分布のパラメータ
      double factor6_;   ///< 訪問分布のパラメータ
    };

    /**
     * @class StrategyChain
     * @brief 戦略チェーンを管理するクラス
     *
     * 戦略チェーンを管理し、新しい位置を生成します。
     */
    class StrategyChain
    {
    public:
      /**
       * @brief コンストラクタ
       * @param acceptance_param 受容パラメータ
       * @param visit_dist 訪問分布
       * @param objective_func 目的関数
       * @param minimizer_wrapper ローカルサーチのラッパー
       * @param rng 乱数生成器
       * @param energy_state エネルギー状態
       */
      StrategyChain(double acceptance_param,
                    VisitingDistribution &visit_dist,
                    ObjectiveFunctionWrapper &objective_func,
                    LocalSearchWrapper &minimizer_wrapper,
                    std::mt19937 &rng,
                    EnergyState &energy_state)
          : acceptance_param_(acceptance_param),
            visit_dist_(visit_dist),
            objective_func_(objective_func),
            minimizer_wrapper_(minimizer_wrapper),
            rng_(rng),
            energy_state_(energy_state)
      {
        xmin_ = energy_state_.current_location();
        emin_ = energy_state_.current_energy();
        // SciPy: self.K = 100 * len(energy_state.current_location)
        K_ = 100 * static_cast<int>(energy_state_.current_location().size());
      }

      /**
       * @brief 戦略チェーンを実行
       * @param step ステップ
       * @param temperature 温度
       * @return 戦略チェーンの結果
       */
      std::optional<std::string> run(int step, double temperature)
      {
        temperature_step_ = temperature / static_cast<double>(step + 1);
        not_improved_idx_++;

        int dim = static_cast<int>(energy_state_.current_location().size());

        for (int j = 0; j < dim * 2; ++j)
        {
          if (j == 0)
          {
            if (step == 0)
            {
              energy_state_improved_ = true;
              not_improved_idx_ = 0;
            }
            else
            {
              energy_state_improved_ = false;
            }
          }

          Eigen::VectorXd x_visit = visit_dist_.visiting(energy_state_.current_location(), j, temperature);

          double energy = objective_func_.fun(x_visit);

          if (energy < energy_state_.current_energy())
          {
            // より良いエネルギーが得られた場合
            energy_state_.update_current(energy, x_visit);
            if (energy < energy_state_.ebest())
            {
              energy_state_.update_best(energy, x_visit);
              energy_state_improved_ = true;
              not_improved_idx_ = 0;
            }
          }
          else
          {
            // 改善しなかったが、確率的に受容するかどうかを判断
            accept_reject(j, energy, x_visit);
          }

          if (objective_func_.num_func_eval() >= objective_func_.maxfun())
          {
            return "Maximum number of function call reached during annealing";
          }
        }
        return std::nullopt;
      }

      /**
       * @brief ローカルサーチを実行
       * @return ローカルサーチの結果
       */
      std::optional<std::string> local_search()
      {
        bool do_ls = false;

        // エネルギーが改善された場合、ローカルサーチを行う
        if (energy_state_improved_)
        {
          auto [energy, x] = minimizer_wrapper_.local_search(energy_state_.xbest(), energy_state_.ebest());
          if (energy < energy_state_.ebest())
          {
            not_improved_idx_ = 0;
            energy_state_.update_best(energy, x);
            energy_state_.update_current(energy, x);
          }
          if (objective_func_.num_func_eval() >= objective_func_.maxfun())
          {
            return std::string("Maximum number of function call reached during local search");
          }
        }

        // 改善がなくても書き率的にローカルサーチを行う判定（Metlopolis like check）
        int dim = static_cast<int>(energy_state_.current_location().size());
        if (K_ < 90 * dim)
        {
          double pls = std::exp(K_ * (energy_state_.ebest() - energy_state_.current_energy()) / temperature_step_);
          std::uniform_real_distribution<double> dist(0.0, 1.0);
          if (pls >= dist(rng_))
          {
            do_ls = true;
          }
        }

        // 長期間改善がない場合は強制的にローカルサーチを行う
        if (not_improved_idx_ >= not_improved_max_idx_)
        {
          do_ls = true;
        }

        if (do_ls)
        {
          auto [energy, x] = minimizer_wrapper_.local_search(xmin_, emin_);
          xmin_ = x;
          emin_ = energy;
          not_improved_idx_ = 0;
          not_improved_max_idx_ = dim; // reset threshold

          if (energy < energy_state_.ebest())
          {
            energy_state_.update_best(emin_, xmin_);
            energy_state_.update_current(energy, x);
          }
          if (objective_func_.num_func_eval() >= objective_func_.maxfun())
          {
            return std::string("Maximum number of function call reached during dual annealing LS");
          }
        }
        return std::nullopt;
      }

    private:
      /**
       * @brief 受容拒否を実行
       * @param j ステップ
       * @param e 目的関数の値
       * @param x_visit 新しい位置
       */
      void accept_reject(int j, double energy, const Eigen::VectorXd &x_visit)
      {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        double r = dist(rng_);

        double pqv_temp = 1.0 - ((1.0 - acceptance_param_) * (energy - energy_state_.current_energy()) / temperature_step_);
        double pqv = 0.0;

        if (pqv_temp > 0.0)
        {
          pqv = std::exp(std::log(pqv_temp) / (1.0 - acceptance_param_));
        }

        if (r <= pqv)
        {
          // 確率的に悪い解を受け入れる
          energy_state_.update_current(energy, x_visit);
          xmin_ = energy_state_.current_location();
        }

        // 長期間改善がない場合の処理
        if (not_improved_idx_ >= not_improved_max_idx_)
        {
          if (j == 0 || energy_state_.current_energy() < emin_)
          {
            emin_ = energy_state_.current_energy();
            xmin_ = energy_state_.current_location();
          }
        }
      }

      double acceptance_param_;                  ///< 受容パラメータ
      VisitingDistribution &visit_dist_;         ///< 訪問分布
      ObjectiveFunctionWrapper &objective_func_; ///< 目的関数
      LocalSearchWrapper &minimizer_wrapper_;    ///< ローカルサーチのラッパー
      std::mt19937 &rng_;                        ///< 乱数生成器
      EnergyState &energy_state_;                ///< エネルギー状態

      Eigen::VectorXd xmin_;               ///< 最適解
      double emin_;                        ///< 最適解のエネルギー
      int not_improved_idx_ = 0;           ///< 改善しない回数
      int not_improved_max_idx_ = 1000;    ///< 改善しない最大回数
      double temperature_step_ = 0;        ///< 温度ステップ
      int K_;                              ///< 最大ステップ数
      bool energy_state_improved_ = false; ///< エネルギー状態が改善したかどうか
    };

    /**
     * @struct DualAnnealingOptions
     * @brief Dual Annealing専用のオプション設定
     *
     * Scipy dual_annealing のパラメータに対応します。
     */
    struct DualAnnealingOptions
    {
      int max_iterations = 1000;                 ///< 最大反復回数
      std::optional<Bounds> bounds;              ///< 境界条件（オプション、未指定の場合は制約なし）
      Eigen::VectorXd initial_params;            ///< 初期パラメータ値
      double initial_temp = 5230.0;              ///< 初期温度 (0.01, 5e4]
      double visit_param = 2.62;                 ///< 訪問パラメータ qv (1, 3]
      double accept_param = -5.0;                ///< 受容パラメータ qa (-1e4, -5]
      double restart_temp_ratio = 2.e-5;         ///< 再アニーリング比率 (0, 1)
      int max_function_evaluations = 10'000'000; ///< 目的関数評価のソフトリミット (10^7)
      std::optional<unsigned> seed;              ///< 乱数シード（再現用）
      bool no_local_search = false;              ///< true で局所探索を行わない
    };

    /**
     * @class DualAnnealingOptimizer
     * @brief Dual Annealing最適化アルゴリズムの実装クラス
     *
     * Dual Annealing法は、Xiang et al. (1997)によって提案された
     * Generalized Simulated Annealing (GSA)に基づく大域的最適化アルゴリズムです。
     *
     * @see Optimizer
     * @see OptimizationResult
     * @see OptimizerOptions
     * @see DualAnnealingOptions
     */
    class DualAnnealingOptimizer : public Optimizer<DualAnnealingOptions>
    {
    public:
      /**
       * @brief 目的関数を最小化する
       *
       * @param objective_func 最小化する目的関数
       * @param initial_params 初期パラメータ値
       * @param options 最適化オプション
       * @return OptimizationResult 最適化結果
       */
      OptimizationResult minimize(
          const ObjectiveFunction &objective,
          const DualAnnealingOptions &options = DualAnnealingOptions{}) override
      {
        if (!options.bounds.has_value())
        {
          throw std::invalid_argument("Dual Annealing requires bounds.");
        }

        const Bounds &bounds = options.bounds.value();
        // 境界条件の次元が一致しているかチェック
        if (bounds.lower.size() != bounds.upper.size())
        {
          throw std::invalid_argument("Bounds do not have the same dimensions.");
        }
        // 境界条件と初期パラメータのサイズが一致しているかチェック
        if (bounds.upper.size() != options.initial_params.size())
        {
          throw std::invalid_argument("Bounds and initial_params size mismatch.");
        }
        // 境界条件が下限が上限より大きいかチェック
        if ((bounds.lower.array() >= bounds.upper.array()).any())
        {
          throw std::invalid_argument("Bounds: lower < upper required.");
        }
        // 再アニーリング比率が0から1の範囲内かチェック
        if (options.restart_temp_ratio <= 0.0 || options.restart_temp_ratio >= 1.0)
        {
          throw std::invalid_argument("restart_temp_ratio must be in (0, 1).");
        }

        // 目的関数のラッパーを作成
        ObjectiveFunctionWrapper objective_func_wrapper(objective, options.max_function_evaluations);

        // ローカルサーチ用のラッパーを作成
        LocalSearchWrapper minimizer_wrapper(bounds, objective_func_wrapper);

        // 乱数生成器を作成
        std::mt19937 rng(options.seed.has_value() ? *options.seed : std::random_device{}());

        // エネルギー状態を作成
        EnergyState energy_state(bounds);
        // エネルギー状態をリセット
        energy_state.reset(objective_func_wrapper, rng, options.initial_params);

        // 再アニーリング温度を計算
        const double temperature_restart = options.initial_temp * options.restart_temp_ratio;

        // 訪問分布を作成
        VisitingDistribution visit_dist(bounds, options.visit_param, rng);

        // 戦略チェーンを作成
        StrategyChain strategy_chain(options.accept_param, visit_dist,
                                     objective_func_wrapper, minimizer_wrapper, rng, energy_state);

        bool need_to_stop = false;
        int iteration = 0;
        std::vector<std::string> messages;

        bool success = true;

        // 訪問分布の温度を計算
        const double t1 = std::exp((options.visit_param - 1.0) * std::log(2.0)) - 1.0;

        while (!need_to_stop)
        {
          for (int i = 0; i < options.max_iterations; ++i)
          {
            // 温度を計算
            const double s = static_cast<double>(i) + 2.0;
            const double t2 = std::exp((options.visit_param - 1.0) * std::log(s)) - 1.0;
            const double temperature = options.initial_temp * t1 / t2;

            if (iteration >= options.max_iterations)
            {
              messages.push_back("Maximum number of iteration reached");
              need_to_stop = true;
              success = false;
              break;
            }
            // 再アニーリングが必要かチェック
            if (temperature < temperature_restart)
            {
              energy_state.reset(objective_func_wrapper, rng, std::nullopt);
              break;
            }
            // 戦略チェーンを実行
            auto val = strategy_chain.run(i, temperature);
            if (val.has_value())
            {
              messages.push_back(*val);
              need_to_stop = true;
              success = false; // 関数評価回数上限などでの停止は必ずしも失敗ではないが、ループ中断要因として処理する
              break;
            }
            // ローカルサーチを実行
            if (!options.no_local_search)
            {
              val = strategy_chain.local_search();
              if (val.has_value())
              {
                messages.push_back(*val);
                need_to_stop = true;
                success = false;
                break;
              }
            }
            ++iteration;
          }
        }

        // 最適化結果を返す
        OptimizationResult result;
        result.converged = success;
        result.optimal_parameters = energy_state.xbest();
        result.optimal_value = energy_state.ebest();
        result.iterations = iteration;

        if (!messages.empty())
        {
          result.message = messages[0];
          for (size_t k = 1; k < messages.size(); ++k)
          {
            result.message += "; " + messages[k];
          }
        }

        return result;
      }
    };
  }
}
