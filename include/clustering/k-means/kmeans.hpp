/**
 * @file kmeans.hpp
 * @brief K-means クラスタリングアルゴリズム
 *
 * Lloyd 法による K-means クラスタリング。
 * 初期化は k-means++ を採用。
 */

#pragma once

#include "linalg/array.hpp"
#include "linalg/array_ops.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <vector>

namespace gprcpp
{
  namespace clustering
  {

    /**
     * @brief K-means クラスタリング
     *
     * 各サンプルを最も近いセントロイドに割り当て、
     * セントロイドをクラスタ内の平均位置に更新する反復を行う。
     * 収束条件: ラベルが変化しない、または中心の変化が tol 以下。
     */
    class KMeans
    {
    public:
      using MatrixXd = gprcpp::linalg::Array2D<double>;
      using VectorXd = gprcpp::linalg::Array1D<double>;
      using VectorXi = gprcpp::linalg::Array1D<int>;

      /**
       * @brief コンストラクタ
       * @param n_clusters クラスタ数
       * @param max_iter 最大反復回数
       * @param tol 収束判定の相対許容誤差（中心の変化の Frobenius ノルム）
       * @param random_state 乱数シード（非負整数）。std::nullopt で非決定的
       * @param init 初期中心 (n_clusters, n_features)。std::nullopt のとき k-means++ で初期化
       */
      explicit KMeans(
          int n_clusters = 8,
          int max_iter = 300,
          double tol = 1e-4,
          std::optional<unsigned> random_state = std::nullopt,
          std::optional<MatrixXd> init = std::nullopt)
          : n_clusters_(n_clusters),
            max_iter_(max_iter),
            tol_(tol),
            random_state_(random_state),
            init_centers_(std::move(init))
      {
        if (n_clusters_ < 1)
        {
          throw std::invalid_argument("kmeans: n_clusters must be >= 1");
        }
        if (max_iter_ < 1)
        {
          throw std::invalid_argument("kmeans: max_iter must be >= 1");
        }
        if (tol_ < 0)
        {
          throw std::invalid_argument("kmeans: tol must be >= 0");
        }
        if (init_centers_.has_value())
        {
          const MatrixXd &ic = *init_centers_;
          if (ic.rows() != static_cast<std::size_t>(n_clusters_) || ic.cols() < 1)
          {
            throw std::invalid_argument(
                "kmeans: init must have shape (n_clusters, n_features)");
          }
        }
      }

      /**
       * @brief データにフィットし、クラスタ中心を学習する
       * @param X データ行列 (n_samples, n_features)
       * @return *this
       */
      KMeans &fit(const MatrixXd &X)
      {
        const std::size_t n_samples = X.rows();
        const std::size_t n_features = X.cols();
        if (n_samples == 0 || n_features == 0)
        {
          throw std::invalid_argument("kmeans: X must not be empty");
        }
        if (n_samples < static_cast<std::size_t>(n_clusters_))
        {
          throw std::invalid_argument(
              "kmeans: n_samples must be >= n_clusters");
        }

        n_samples_ = n_samples;
        n_features_ = n_features;

        std::mt19937 rng(
            random_state_.has_value()
                ? *random_state_
                : std::random_device{}());

        MatrixXd centers;
        if (init_centers_.has_value())
        {
          const MatrixXd &ic = *init_centers_;
          if (ic.rows() != static_cast<std::size_t>(n_clusters_) ||
              ic.cols() != n_features)
          {
            throw std::invalid_argument(
                "kmeans: init must have shape (n_clusters, n_features)");
          }
          centers = ic;
        }
        else
        {
          centers = init_centroids_(X, rng);
        }

        VectorXi labels(static_cast<std::size_t>(X.rows()));
        double inertia = 0;
        int n_iter = 0;

        lloyd_iter_(X, centers, labels, inertia, n_iter, rng);

        cluster_centers_ = centers;
        labels_ = labels;
        inertia_ = inertia;
        n_iter_ = n_iter;

        return *this;
      }

      /**
       * @brief フィットし、各サンプルのクラスタラベルを返す
       * @param X データ行列 (n_samples, n_features)
       * @return ラベルベクトル (n_samples,)
       */
      VectorXi fit_predict(const MatrixXd &X)
      {
        fit(X);
        return labels_;
      }

      /**
       * @brief 各サンプルに最も近いクラスタを予測する（fit 済みであること）
       * @param X データ行列 (n_samples, n_features)
       * @return ラベルベクトル (n_samples,)
       */
      VectorXi predict(const MatrixXd &X) const
      {
        if (!is_fitted())
        {
          throw std::runtime_error("kmeans: fit has not been called");
        }
        if (X.cols() != n_features_)
        {
          throw std::invalid_argument(
              "kmeans: X.cols() must match n_features");
        }
        return assign_labels_(X, cluster_centers_);
      }

      /**
       * @brief X をクラスタ中心との距離空間に変換する
       * @param X データ行列 (n_samples, n_features)
       * @return 各サンプルから各クラスタ中心への距離 (n_samples, n_clusters)
       */
      MatrixXd transform(const MatrixXd &X) const
      {
        if (!is_fitted())
        {
          throw std::runtime_error("kmeans: fit has not been called");
        }
        if (X.cols() != n_features_)
        {
          throw std::invalid_argument(
              "kmeans: X.cols() must match n_features");
        }
        return euclidean_distances_(X, cluster_centers_);
      }

      /**
       * @brief フィットし、クラスタ距離空間に変換する
       * @param X データ行列 (n_samples, n_features)
       * @return クラスタ距離空間に変換したデータ行列 (n_samples, n_clusters)
       */
      MatrixXd fit_transform(const MatrixXd &X)
      {
        fit(X);
        return transform(X);
      }

      const MatrixXd &cluster_centers() const { return cluster_centers_; }
      const VectorXi &labels() const { return labels_; }
      double inertia() const { return inertia_; }
      int n_iter() const { return n_iter_; }
      std::size_t n_samples() const { return n_samples_; }
      std::size_t n_features() const { return n_features_; }
      int n_clusters() const { return n_clusters_; }

      bool is_fitted() const { return cluster_centers_.size() > 0; }

    private:
      int n_clusters_;
      int max_iter_;
      double tol_;
      std::optional<unsigned> random_state_;
      std::optional<MatrixXd> init_centers_;

      std::size_t n_samples_{0};
      std::size_t n_features_{0};
      MatrixXd cluster_centers_;
      VectorXi labels_;
      double inertia_{0};
      int n_iter_{0};

      /**
       * @brief データの分散に基づく許容誤差を計算
       * @param X データ行列 (n_samples, n_features)
       * @return 許容誤差
       */
      double tolerance_(const MatrixXd &X) const
      {
        if (tol_ == 0)
        {
          return 0;
        }
        const VectorXd col_mean = gprcpp::linalg::column_means(X);
        MatrixXd centered(X.rows(), X.cols());
        for (std::size_t i = 0; i < X.rows(); ++i)
          for (std::size_t j = 0; j < X.cols(); ++j)
            centered(i, j) = X(i, j) - col_mean(j);
        double var_sum = 0;
        for (std::size_t j = 0; j < X.cols(); ++j)
        {
          double s = 0;
          for (std::size_t i = 0; i < X.rows(); ++i)
            s += centered(i, j) * centered(i, j);
          var_sum += s / static_cast<double>(X.rows());
        }
        return (var_sum / static_cast<double>(X.cols())) * tol_;
      }

      /**
       * @brief 各行のノルムの二乗を計算
       * @param X データ行列 (n_samples, n_features)
       * @return 各行のノルムの二乗 (n_samples,)
       */
      VectorXd row_norms_squared_(const MatrixXd &X) const
      {
        return gprcpp::linalg::row_norms_squared(X);
      }

      /**
       * @brief 二乗ユークリッド距離行列を計算
       * @param X データ行列 (n_samples, n_features)
       * @param Y データ行列 (n_samples, n_features)
       * @param X_sq_norms 各サンプルの二乗ノルム (n_samples,)
       * @param Y_sq_norms 各サンプルの二乗ノルム (n_samples,)
       * @return 二乗ユークリッド距離行列 (n_samples, n_samples)
       *
       * D(i,j) = ||X(i,:) - Y(j,:)||^2 = ||x_i||^2 + ||y_j||^2 - 2 * x_i^T y_j
       */
      MatrixXd squared_euclidean_distances_(
          const MatrixXd &X,
          const MatrixXd &Y,
          const VectorXd &X_sq_norms,
          const VectorXd &Y_sq_norms) const
      {
        MatrixXd D = gprcpp::linalg::matmul_abt(X, Y);
        for (std::size_t i = 0; i < D.rows(); ++i)
          for (std::size_t j = 0; j < D.cols(); ++j)
            D(i, j) *= -2;
        gprcpp::linalg::add_to_columns(D, X_sq_norms);
        gprcpp::linalg::add_to_rows(D, Y_sq_norms);
        gprcpp::linalg::cwise_max_zero(D);
        return D;
      }

      /**
       * @brief ユークリッド距離行列（非二乗）
       * @param X データ行列 (n_samples, n_features)
       * @param Y データ行列 (n_samples, n_features)
       * @return ユークリッド距離行列 (n_samples, n_samples)
       */
      MatrixXd euclidean_distances_(const MatrixXd &X, const MatrixXd &Y) const
      {
        VectorXd X_sq = row_norms_squared_(X);
        VectorXd Y_sq = row_norms_squared_(Y);
        MatrixXd D_sq = squared_euclidean_distances_(X, Y, X_sq, Y_sq);
        return gprcpp::linalg::sqrt_elementwise(D_sq);
      }

      /**
       * @brief k-means++ 初期化
       * @param X データ行列 (n_samples, n_features)
       * @param x_squared_norms 各サンプルの二乗ノルム (n_samples,)
       * @param rng 乱数エンジン
       * @return クラスタ中心行列 (n_clusters, n_features)
       */
      MatrixXd init_kmeans_plusplus_(
          const MatrixXd &X,
          const VectorXd &x_squared_norms,
          std::mt19937 &rng) const
      {
        const std::size_t n_samples = X.rows();
        const std::size_t n_features = X.cols();
        MatrixXd centers(static_cast<std::size_t>(n_clusters_), n_features);

        // 最初の中心をランダムに選択
        std::uniform_int_distribution<std::size_t> dist(0, n_samples - 1);
        std::size_t first = dist(rng);
        for (std::size_t j = 0; j < n_features; ++j)
          centers(0, j) = X(first, j);

        MatrixXd centers1(1, n_features);
        for (std::size_t j = 0; j < n_features; ++j)
          centers1(0, j) = centers(0, j);
        VectorXd Y_sq = row_norms_squared_(centers1);
        MatrixXd closest_dist_sq = squared_euclidean_distances_(
            X, centers1, x_squared_norms, Y_sq);
        VectorXd closest_col(n_samples);
        for (std::size_t i = 0; i < n_samples; ++i)
          closest_col(i) = closest_dist_sq.at(i, static_cast<std::size_t>(0));

        for (int c = 1; c < n_clusters_; ++c)
        {
          double current_pot = 0;
          for (std::size_t i = 0; i < n_samples; ++i)
            current_pot += closest_col(i);
          if (current_pot <= 0)
          {
            break;
          }

          // 累積和で重み付きサンプリング
          VectorXd cumsum(n_samples);
          cumsum(0) = closest_col(0);
          for (std::size_t i = 1; i < n_samples; ++i)
            cumsum(i) = cumsum(i - 1) + closest_col(i);

          std::uniform_real_distribution<double> u01(0, 1);
          double r = u01(rng) * current_pot;
          std::size_t idx = 0;
          for (std::size_t i = 0; i < n_samples; ++i)
          {
            if (cumsum(i) >= r)
            {
              idx = i;
              break;
            }
          }
          if (idx >= n_samples)
            idx = n_samples - 1;

          for (std::size_t j = 0; j < n_features; ++j)
            centers(static_cast<std::size_t>(c), j) = X(idx, j);

          // 新しい中心との距離を計算し、最近傍距離を更新
          MatrixXd center_row(1, n_features);
          for (std::size_t j = 0; j < n_features; ++j)
            center_row(0, j) = centers(static_cast<std::size_t>(c), j);
          Y_sq = row_norms_squared_(center_row);
          MatrixXd dist_mat = squared_euclidean_distances_(
              X, center_row, x_squared_norms, Y_sq);
          for (std::size_t i = 0; i < n_samples; ++i)
          {
            double d = dist_mat.at(i, static_cast<std::size_t>(0));
            if (d < closest_col(i))
              closest_col(i) = d;
          }
        }

        return centers;
      }

      /**
       * @brief k-means++ 初期化
       * @param X データ行列 (n_samples, n_features)
       * @param rng 乱数エンジン
       * @return クラスタ中心行列 (n_clusters, n_features)
       */
      MatrixXd init_centroids_(const MatrixXd &X, std::mt19937 &rng) const
      {
        VectorXd x_sq = row_norms_squared_(X);
        return init_kmeans_plusplus_(X, x_sq, rng);
      }

      /**
       * @brief 各サンプルを最も近い中心に割り当て
       * @param X データ行列 (n_samples, n_features)
       * @param centers クラスタ中心行列 (n_clusters, n_features)
       * @return ラベルベクトル (n_samples,)
       */
      VectorXi assign_labels_(const MatrixXd &X, const MatrixXd &centers) const
      {
        VectorXd X_sq = row_norms_squared_(X);
        VectorXd C_sq = row_norms_squared_(centers);
        MatrixXd D_sq = squared_euclidean_distances_(X, centers, X_sq, C_sq);

        VectorXi labels(X.rows());
        for (std::size_t i = 0; i < X.rows(); ++i)
          labels(i) = static_cast<int>(gprcpp::linalg::row_argmin(D_sq, i));
        return labels;
      }

      /**
       * @brief 慣性（各サンプルから割当クラスタ中心までの二乗距離の和）を計算
       * @param X データ行列 (n_samples, n_features)
       * @param centers クラスタ中心行列 (n_clusters, n_features)
       * @param labels ラベルベクトル (n_samples,)
       * @return 慣性
       */
      double compute_inertia_(
          const MatrixXd &X,
          const MatrixXd &centers,
          const VectorXi &labels) const
      {
        double sum_sq = 0;
        for (std::size_t i = 0; i < X.rows(); ++i)
        {
          int k = labels(i);
          sum_sq += gprcpp::linalg::row_squared_distance(
              X, i, centers, static_cast<std::size_t>(k));
        }
        return sum_sq;
      }

      /**
       * @brief Lloyd 反復
       * @param X データ行列 (n_samples, n_features)
       * @param centers クラスタ中心行列 (n_clusters, n_features)
       * @param labels ラベルベクトル (n_samples,)
       * @param inertia 慣性
       * @param n_iter 反復回数
       * @param rng 乱数エンジン
       */
      void lloyd_iter_(
          const MatrixXd &X,
          MatrixXd &centers,
          VectorXi &labels,
          double &inertia,
          int &n_iter,
          std::mt19937 &rng) const
      {
        const double tol_val = tolerance_(X);
        MatrixXd centers_new = centers;
        VectorXi labels_old(X.rows(), -1);
        VectorXd weight_in_clusters(static_cast<std::size_t>(n_clusters_), 0.0);

        for (n_iter = 0; n_iter < max_iter_; ++n_iter)
        {
          // E-step: 割り当て
          labels = assign_labels_(X, centers);

          // M-step: 中心の更新
          centers_new.fill(0);
          weight_in_clusters.fill(0);

          for (std::size_t i = 0; i < X.rows(); ++i)
          {
            int k = labels(i);
            for (std::size_t j = 0; j < X.cols(); ++j)
              centers_new(static_cast<std::size_t>(k), j) += X(i, j);
            weight_in_clusters(static_cast<std::size_t>(k)) += 1;
          }

          // 空クラスタの処理: ランダムなサンプルで置換
          std::uniform_int_distribution<std::size_t> dist(0, X.rows() - 1);
          for (int k = 0; k < n_clusters_; ++k)
          {
            const std::size_t kk = static_cast<std::size_t>(k);
            if (weight_in_clusters(kk) <= 0)
            {
              std::size_t row = dist(rng);
              for (std::size_t j = 0; j < X.cols(); ++j)
                centers_new(kk, j) = X(row, j);
            }
            else
            {
              double w = weight_in_clusters(kk);
              for (std::size_t j = 0; j < X.cols(); ++j)
                centers_new(kk, j) /= w;
            }
          }

          // 収束判定: ラベルが変化しない
          bool labels_unchanged = true;
          for (std::size_t i = 0; i < labels.size(); ++i)
          {
            if (labels(i) != labels_old(i))
            {
              labels_unchanged = false;
              break;
            }
          }
          if (labels_unchanged)
          {
            ++n_iter;
            centers = centers_new;
            inertia = compute_inertia_(X, centers, labels);
            return;
          }

          // 収束判定: 中心の変化が tol 以下
          double center_shift_sq = gprcpp::linalg::frobenius_squared_diff(centers_new, centers);
          if (center_shift_sq <= tol_val)
          {
            ++n_iter;
            centers = centers_new;
            inertia = compute_inertia_(X, centers, labels);
            return;
          }

          centers = centers_new;
          for (std::size_t i = 0; i < labels.size(); ++i)
            labels_old(i) = labels(i);
        }

        inertia = compute_inertia_(X, centers, labels);
      }
    };
  }
}
