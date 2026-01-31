/**
 * @file gp_regression.hpp
 * @brief ガウス過程回帰 (Gaussian Process Regression)
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernel.hpp"
#include "optimize/optimizer.hpp"
#include "optimize/DualAnnealing/dual_annealing.hpp"

namespace gprcpp
{
  namespace regressor
  {
    /**
     * @brief 予測結果を保持する構造体
     *
     * mean (n_query, n_targets), std (n_query, n_targets)。
     * cov はターゲットごとの共分散行列。cov[j] は (n_query, n_query) で、normalize_y 時はターゲット j のスケールで復元済み。
     */
    struct PredictResult
    {
      Eigen::MatrixXd mean;             ///< 予測平均 (n_query, n_targets)
      Eigen::MatrixXd std;              ///< 予測標準偏差 (n_query, n_targets)
      std::vector<Eigen::MatrixXd> cov; ///< ターゲットごとの予測共分散。cov[j] は (n_query, n_query)
    };

    /**
     * @brief ガウス過程回帰 (GPR)
     *
     * 学習データに基づいてガウス過程の事後分布を計算し、
     * 任意の入力点における予測平均・分散（および共分散）を返す。
     *
     * @tparam OptimizerOptions ハイパーパラメータ最適化に使うオプション型
     */
    template <typename OptimizerOptions = optimize::DualAnnealingOptions>
    class GaussianProcessRegressor
    {
    public:
      /**
       * @brief コンストラクタ（スカラー alpha）
       * @param kernel カーネル
       * @param optimizer ハイパーパラメータ最適化に使うオプティマイザ（必須）
       * @param alpha カーネル行列の対角に加える値（数値安定・観測ノイズ）。全サンプル共通
       * @param n_restarts_optimizer ハイパーパラメータ最適化の再起動回数（0 で 1 回のみ）
       * @param normalize_y ターゲットを平均0・分散1に正規化してから学習するか
       * @param random_state 乱数シード（n_restarts や sample_y 用）。std::nullopt で非決定的
       */
      GaussianProcessRegressor(
          std::shared_ptr<kernels::Kernel> kernel,
          std::shared_ptr<optimize::Optimizer<OptimizerOptions>> optimizer,
          double alpha = 1e-10,
          int n_restarts_optimizer = 0,
          bool normalize_y = false,
          std::optional<int> random_state = std::nullopt)
          : kernel_(std::move(kernel)),
            alpha_(Eigen::VectorXd::Constant(1, alpha)),
            n_restarts_optimizer_(n_restarts_optimizer),
            normalize_y_(normalize_y),
            random_state_(random_state),
            optimizer_(std::move(optimizer))
      {
      }

      /**
       * @brief コンストラクタ（ベクトル alpha：サンプルごとのノイズ）
       * @param kernel カーネル
       * @param optimizer ハイパーパラメータ最適化に使うオプティマイザ（必須）
       * @param alpha 対角に加える値。(size()==1) のときは全サンプル共通、(size()==n_samples) のときはサンプルごと
       * @param n_restarts_optimizer ハイパーパラメータ最適化の再起動回数（0 で 1 回のみ）
       * @param normalize_y ターゲットを平均0・分散1に正規化してから学習するか
       * @param random_state 乱数シード（n_restarts や sample_y 用）。std::nullopt で非決定的
       */
      GaussianProcessRegressor(
          std::shared_ptr<kernels::Kernel> kernel,
          std::shared_ptr<optimize::Optimizer<OptimizerOptions>> optimizer,
          Eigen::VectorXd alpha,
          int n_restarts_optimizer = 0,
          bool normalize_y = false,
          std::optional<int> random_state = std::nullopt)
          : kernel_(std::move(kernel)),
            alpha_(std::move(alpha)),
            n_restarts_optimizer_(n_restarts_optimizer),
            normalize_y_(normalize_y),
            random_state_(random_state),
            optimizer_(std::move(optimizer))
      {
        if (alpha_.size() == 0)
        {
          alpha_.resize(1);
          alpha_(0) = 1e-10;
        }
      }

      /**
       * @brief ガウス過程回帰モデルを学習する（スカラー出力用オーバーロード）
       * @param X 入力 (n_samples, n_features)
       * @param y ターゲット (n_samples,)
       * @return *this
       */
      GaussianProcessRegressor &fit(const Eigen::MatrixXd &X, const Eigen::VectorXd &y)
      {
        return fit(X, Eigen::MatrixXd(y));
      }

      /**
       * @brief ガウス過程回帰モデルを学習する（ハイパーパラメータはコンストラクタで渡したオプティマイザで最適化する）
       * @param X 入力 (n_samples, n_features)
       * @param y ターゲット (n_samples, n_targets)。ベクトル出力対応
       * @return *this
       */
      GaussianProcessRegressor &fit(const Eigen::MatrixXd &X, const Eigen::MatrixXd &y)
      {
        if (X.rows() != y.rows())
        {
          throw std::invalid_argument("X.rows() must equal y.rows().");
        }
        if (y.cols() < 1)
        {
          throw std::invalid_argument("y must have at least one column.");
        }

        kernel_ = kernel_->clone();

        // --- ターゲットの正規化（オプション）---
        const Eigen::Index n_targets = y.cols();
        y_train_mean_.resize(n_targets);
        y_train_std_.resize(n_targets);
        Eigen::MatrixXd y_col = y;

        if (normalize_y_)
        {
          for (Eigen::Index j = 0; j < n_targets; ++j)
          {
            y_train_mean_(j) = y_col.col(j).mean();
            double var = (y_col.col(j).array() - y_train_mean_(j)).square().sum() / y_col.rows();
            double std_val = std::sqrt(var);
            if (std_val < 1e-10)
            {
              std_val = 1.0; // 分散がほぼ 0 の列は正規化しない（除算を避ける）
            }
            y_train_std_(j) = std_val;
            y_col.col(j) = (y_col.col(j).array() - y_train_mean_(j)) / y_train_std_(j);
          }
        }
        else
        {
          y_train_mean_.setZero();
          y_train_std_.setOnes();
        }

        X_train_ = X;
        y_train_ = std::move(y_col);

        const Eigen::Index n_samples = X.rows();
        // alpha はスカラー（全サンプル共通）または n_samples 長（サンプルごと）のみ許可
        if (alpha_.size() != 1 && alpha_.size() != n_samples)
        {
          throw std::invalid_argument(
              "alpha must have size 1 (scalar) or n_samples. (alpha.size() = " +
              std::to_string(alpha_.size()) + ", n_samples = " + std::to_string(static_cast<long long>(n_samples)) + ")");
        }

        // --- ハイパーパラメータ最適化（カーネルにハイパーパラメータがある場合）---
        int n_params = kernel_->num_hyperparameters();
        if (n_params > 0)
        {
          auto [bounds_lower, bounds_upper] = kernel_->get_hyperparameter_bounds();
          optimize::Bounds kernel_bounds(bounds_lower, bounds_upper);

          if (n_restarts_optimizer_ > 0)
          {
            if (!bounds_lower.allFinite() || !bounds_upper.allFinite())
            {
              throw std::invalid_argument(
                  "Multiple optimizer restarts (n_restarts_optimizer>0) "
                  "requires that all bounds are finite.");
            }
          }

          // 目的関数: 対数周辺尤度の符号を反転（最小化で尤度最大化）
          optimize::ObjectiveFunction obj = [this](const Eigen::VectorXd &theta) -> double
          {
            return -log_marginal_likelihood_impl(theta, false);
          };

          OptimizerOptions opts;
          opts.initial_params = kernel_->get_hyperparameters();
          opts.bounds = kernel_bounds;

          auto res = optimizer_->minimize(obj, opts);
          Eigen::VectorXd best_theta = res.optimal_parameters;
          double best_neg_lml = res.optimal_value;

          // 複数初期値から再起動して最良解を採用
          if (n_restarts_optimizer_ > 0)
          {
            std::mt19937 rng(random_state_.has_value() ? static_cast<unsigned>(*random_state_) : std::random_device{}());
            for (int r = 0; r < n_restarts_optimizer_; ++r)
            {
              opts.initial_params.resize(n_params);
              for (int i = 0; i < n_params; ++i)
              {
                std::uniform_real_distribution<double> u(bounds_lower(i), bounds_upper(i));
                opts.initial_params(i) = u(rng);
              }
              auto rest = optimizer_->minimize(obj, opts);
              if (rest.optimal_value < best_neg_lml)
              {
                best_neg_lml = rest.optimal_value;
                best_theta = rest.optimal_parameters;
              }
            }
          }
          kernel_->set_hyperparameters(best_theta);
          log_marginal_likelihood_value_ = -best_neg_lml;
        }
        else
        {
          log_marginal_likelihood_value_ = log_marginal_likelihood_impl(kernel_->get_hyperparameters(), false);
        }

        // --- カーネル行列の構築と Cholesky 分解 ---
        // K = k(X_train, X_train) + diag(alpha)。事後の平均・分散計算に K^{-1} y と L を使用
        Eigen::MatrixXd K = (*kernel_)(X_train_);
        if (alpha_.size() == 1)
        {
          K.diagonal().array() += alpha_(0);
        }
        else
        {
          K.diagonal() += alpha_;
        }
        // K = L L^T となる下三角 L を計算。正定値でない場合は alpha 増加を促す
        Eigen::LLT<Eigen::MatrixXd> llt(K);
        if (llt.info() != Eigen::Success)
        {
          throw std::runtime_error(
              "The kernel matrix is not positive definite. Try increasing 'alpha'.");
        }
        L_ = llt.matrixL();
        alpha_dual_ = llt.solve(y_train_); // K^{-1} y（双対係数）。予測平均は K(X*,X_train) * alpha_dual_

        return *this;
      }

      /**
       * @brief 予測結果を返す（平均・標準偏差・共分散行列）
       * @param X 入力 (n_samples, n_features)
       */
      PredictResult predict(const Eigen::MatrixXd &X) const
      {
        PredictResult out;
        const Eigen::Index n_targets = is_fitted() ? y_train_.cols() : 1;
        const Eigen::Index n_query = X.rows();

        // 未学習時は事前分布の平均 0・分散 k(x,x) で返す
        if (!is_fitted())
        {
          out.mean = Eigen::MatrixXd::Zero(n_query, n_targets);
          Eigen::VectorXd diag_sqrt = kernel_->diag(X).array().sqrt();
          out.std.resize(n_query, n_targets);
          for (Eigen::Index j = 0; j < n_targets; ++j)
          {
            out.std.col(j) = diag_sqrt;
          }
          out.cov.resize(1);
          out.cov[0] = (*kernel_)(X);
          return out;
        }

        // 事後平均: μ* = K(X*, X_train) @ alpha_dual_（正規化時は後でスケール・シフトを復元）
        Eigen::MatrixXd K_trans = (*kernel_)(X, X_train_);
        out.mean = K_trans * alpha_dual_;
        for (Eigen::Index j = 0; j < n_targets; ++j)
        {
          out.mean.col(j) = y_train_std_(j) * out.mean.col(j).array() + y_train_mean_(j);
        }

        // 事後共分散: Σ* = K(X*,X*) - K(X*,X_train) @ K^{-1} @ K(X_train,X*)
        // V = L^{-1} K(X_train,X*)^T とすると、第2項 = V^T V。kernel_part が正規化空間での共分散
        Eigen::MatrixXd V = L_.triangularView<Eigen::Lower>().solve(K_trans.transpose());
        Eigen::MatrixXd kernel_part = (*kernel_)(X)-V.transpose() * V;

        out.cov.resize(n_targets);
        for (Eigen::Index j = 0; j < n_targets; ++j)
        {
          out.cov[j] = (y_train_std_(j) * y_train_std_(j)) * kernel_part;
        }

        // 対角分散（予測標準偏差用）: diag(Σ*) = diag(K*) - ||v_i||^2
        // sklearn 同様 Alg 2.1 line 5-6。colwise().squaredNorm() は行ベクトルなので transpose で列ベクトルに
        Eigen::VectorXd var =
            kernel_->diag(X) - V.colwise().squaredNorm().transpose();
        for (Eigen::Index i = 0; i < var.size(); ++i)
        {
          if (var(i) < 0.0)
          {
            var(i) = 0.0; // 数値誤差で負になった場合は 0 にクリップ
          }
        }
        Eigen::VectorXd std_vals = var.array().sqrt();
        out.std.resize(n_query, n_targets);
        for (Eigen::Index j = 0; j < n_targets; ++j)
        {
          out.std.col(j) = y_train_std_(j) * std_vals;
        }

        return out;
      }

      /**
       * @brief 対数周辺尤度
       * @param theta カーネルハイパーパラメータ（対数スケール）。省略時は kernel_.theta で計算した値を返す
       * @param clone_kernel true なら kernel を clone して theta を設定してから計算（元の kernel_ は変更しない）
       */
      double log_marginal_likelihood(
          const std::optional<Eigen::VectorXd> &theta = std::nullopt,
          bool clone_kernel = true) const
      {
        if (!theta.has_value())
        {
          return log_marginal_likelihood_value_;
        }
        return log_marginal_likelihood_impl(*theta, clone_kernel);
      }

      /**
       * @brief 事後から X 上でサンプルを生成
       * @param X 入力 (n_samples, n_features)
       * @param n_samples サンプル数
       * @param random_state 乱数シード（省略時はコンストラクタの random_state または非決定的）
       * @return (n_query, n_targets * n_samples)。列は [draw0_t0, draw0_t1, ..., draw0_tN, draw1_t0, ...]。スカラー出力時は (n_query, n_samples)
       */
      Eigen::MatrixXd sample_y(
          const Eigen::MatrixXd &X,
          int n_samples = 1,
          std::optional<int> random_state = std::nullopt) const
      {
        PredictResult pr = predict(X);
        const Eigen::Index n_query = X.rows();
        const Eigen::Index n_targets = pr.mean.cols();

        // 事後共分散の Cholesky: Σ* = L_cov L_cov^T。サンプルは mean + L_cov @ z, z~N(0,I)
        std::vector<Eigen::MatrixXd> L_per_target(static_cast<std::size_t>(n_targets));
        for (Eigen::Index j = 0; j < n_targets; ++j)
        {
          Eigen::MatrixXd cov_j = pr.cov[j];
          Eigen::LLT<Eigen::MatrixXd> llt(cov_j);
          if (llt.info() != Eigen::Success)
          {
            cov_j.diagonal().array() += 1e-10; // 数値的に正定値にする
            llt.compute(cov_j);
          }
          L_per_target[static_cast<std::size_t>(j)] = llt.matrixL();
        }

        std::mt19937 rng(
            random_state.has_value() ? static_cast<unsigned>(*random_state)
                                     : (random_state_.has_value() ? static_cast<unsigned>(*random_state_) : std::random_device{}()));
        std::normal_distribution<double> norm(0.0, 1.0);

        Eigen::MatrixXd out(n_query, n_targets * static_cast<Eigen::Index>(n_samples));
        for (int s = 0; s < n_samples; ++s)
        {
          for (Eigen::Index j = 0; j < n_targets; ++j)
          {
            Eigen::VectorXd z(n_query);
            for (Eigen::Index i = 0; i < z.size(); ++i)
            {
              z(i) = norm(rng);
            }
            // 1 サンプル: y* = μ* + L_cov @ z
            out.col(s * n_targets + j) = pr.mean.col(j) + L_per_target[static_cast<std::size_t>(j)] * z;
          }
        }
        return out;
      }

      /**
       * @brief 学習済みか
       * @return 学習済みか
       */
      bool is_fitted() const
      {
        return L_.rows() > 0;
      }

      /**
       * @brief 学習入力 (n_samples, n_features)
       * @return 学習入力
       */
      const Eigen::MatrixXd &X_train() const
      {
        return X_train_;
      }

      /**
       * @brief 学習ターゲット（正規化後）(n_samples, n_targets)
       * @return 学習ターゲット
       */
      const Eigen::MatrixXd &y_train() const
      {
        return y_train_;
      }

      /**
       * @brief 出力次元数（ターゲット数）
       * @return n_targets。未学習時は 0
       */
      Eigen::Index n_targets() const
      {
        return y_train_.cols();
      }

      /**
       * @brief 使用カーネル（学習後は最適化済み）
       * @return 使用カーネル
       */
      const std::shared_ptr<kernels::Kernel> &kernel() const
      {
        return kernel_;
      }

      /**
       * @brief Cholesky 因子 L (K + alpha*I = L*L^T)
       * @return Cholesky 因子
       */
      const Eigen::MatrixXd &L() const
      {
        return L_;
      }

      /**
       * @brief 双対係数 alpha_ = K^{-1} y (n_samples, n_targets)
       * @return 双対係数
       */
      const Eigen::MatrixXd &alpha_dual() const
      {
        return alpha_dual_;
      }

      /**
       * @brief 対数周辺尤度（最後に fit したときの値）
       * @return 対数周辺尤度
       */
      double log_marginal_likelihood_value() const
      {
        return log_marginal_likelihood_value_;
      }

      /**
       * @brief 正規化時の平均 (n_targets,)
       * @return 正規化時の平均
       */
      const Eigen::VectorXd &y_train_mean() const
      {
        return y_train_mean_;
      }

      /**
       * @brief 正規化時の標準偏差 (n_targets,)
       * @return 正規化時の標準偏差
       */
      const Eigen::VectorXd &y_train_std() const
      {
        return y_train_std_;
      }

    private:
      /**
       * @brief 対数周辺尤度を計算する
       * @param theta カーネルハイパーパラメータ（対数スケール）
       * @param clone_kernel true なら kernel を clone して theta を設定してから計算（元の kernel_ は変更しない）
       * @return 対数周辺尤度
       */
      double log_marginal_likelihood_impl(const Eigen::VectorXd &theta, bool clone_kernel) const
      {
        std::shared_ptr<kernels::Kernel> k = clone_kernel ? kernel_->clone() : kernel_;
        k->set_hyperparameters(theta);

        Eigen::MatrixXd K = (*k)(X_train_);
        if (alpha_.size() == 1)
          K.diagonal().array() += alpha_(0);
        else
          K.diagonal() += alpha_;

        Eigen::LLT<Eigen::MatrixXd> llt(K);
        if (llt.info() != Eigen::Success)
        {
          return -std::numeric_limits<double>::infinity();
        }
        Eigen::MatrixXd L = llt.matrixL();
        Eigen::MatrixXd a = llt.solve(y_train_); // a = K^{-1} y

        // log p(y|X,θ) = -0.5 y^T K^{-1} y - sum(log(diag(L))) - (n/2) log(2π)（ターゲットごとの和）
        double log_lik = 0.0;
        for (Eigen::Index j = 0; j < y_train_.cols(); ++j)
        {
          log_lik += -0.5 * (y_train_.col(j).dot(a.col(j)));
        }
        log_lik -= static_cast<double>(y_train_.cols()) * L.diagonal().array().log().sum();
        log_lik -= 0.5 * X_train_.rows() * y_train_.cols() * std::log(2.0 * (2.0 * std::acos(0.0)));
        return log_lik;
      }

      std::shared_ptr<kernels::Kernel> kernel_;                          ///< カーネル
      Eigen::VectorXd alpha_;                                            ///< 対角に加える値。size()==1 はスカラー、それ以外はサンプルごと
      int n_restarts_optimizer_;                                         ///< ハイパーパラメータ最適化の再起動回数（0 で 1 回のみ）
      bool normalize_y_;                                                 ///< ターゲットを平均0・分散1に正規化してから学習するか
      std::optional<int> random_state_;                                  ///< 乱数シード（n_restarts や sample_y 用）。std::nullopt で非決定的
      std::shared_ptr<optimize::Optimizer<OptimizerOptions>> optimizer_; ///< ハイパーパラメータ最適化に使うオプティマイザ

      Eigen::MatrixXd X_train_;                   ///< 学習入力
      Eigen::MatrixXd y_train_;                   ///< 学習ターゲット（正規化後）(n_samples, n_targets)
      Eigen::VectorXd y_train_mean_;              ///< 正規化時の平均 (n_targets,)
      Eigen::VectorXd y_train_std_;               ///< 正規化時の標準偏差 (n_targets,)
      Eigen::MatrixXd L_;                         ///< Cholesky 因子 L (K + alpha*I = L*L^T)
      Eigen::MatrixXd alpha_dual_;                ///< 双対係数 alpha_ = K^{-1} y (n_samples, n_targets)
      double log_marginal_likelihood_value_{0.0}; ///< 対数周辺尤度（最後に fit したときの値）
    };
  }
}
