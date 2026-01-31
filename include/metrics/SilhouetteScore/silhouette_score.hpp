/**
 * @file silhouette_score.hpp
 * @brief シルエット係数（Silhouette Coefficient）メトリクス
 *
 * クラスタリングの評価指標。各サンプルについてクラスタ内平均距離 a と
 * 最近傍他クラスタ平均距離 b を用い、s = (b - a) / max(a, b) を計算する。
 * 最良は 1、最悪は -1。0 付近はクラスタの重なりを示す。
 *
 * @see Rousseeuw (1987), "Silhouettes: a Graphical Aid to the Interpretation
 *      and Validation of Cluster Analysis", Computational and Applied Mathematics 20.
 */

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace gprcpp
{
  namespace metrics
  {

    namespace detail
    {
      /**
       * @brief ラベル数が有効範囲かチェック（2 <= n_labels <= n_samples - 1）
       * シルエット係数は「同一クラスタ内」と「他クラスタ」の両方が必要なため、
       * クラスタ数は 2 以上かつ全サンプルが同一クラスタでない（n_samples - 1 以下）必要がある。
       * @param n_labels ラベル数（クラスタ数）
       * @param n_samples サンプル数
       */
      inline void check_number_of_labels(int n_labels, Eigen::Index n_samples)
      {
        if (n_labels < 2 || n_labels >= static_cast<int>(n_samples))
        {
          throw std::invalid_argument(
              "silhouette: number of labels must be in [2, n_samples - 1]");
        }
      }

      /**
       * @brief ラベルを 0..K-1 にエンコードし、各クラスタの頻度を返す
       * @param labels ラベルベクトル
       * @param encoded エンコードされたラベルベクトル
       * @param n_clusters クラスタ数
       * @return 各クラスタの頻度
       */
      inline std::vector<int>
      encode_labels(const Eigen::Ref<const Eigen::VectorXi> &labels,
                    Eigen::VectorXi &encoded,
                    int &n_clusters)
      {
        const Eigen::Index n_samples = labels.size();
        std::map<int, int> label_to_idx;
        encoded.resize(n_samples);

        // 第1パス: 出現順に 0..K-1 へ写像
        for (Eigen::Index i = 0; i < n_samples; ++i)
        {
          int lab = labels(i);
          auto it = label_to_idx.find(lab);
          if (it == label_to_idx.end())
          {
            int k = static_cast<int>(label_to_idx.size());
            label_to_idx[lab] = k;
            encoded(i) = k;
          }
          else
          {
            encoded(i) = it->second;
          }
        }

        n_clusters = static_cast<int>(label_to_idx.size());
        std::vector<int> label_freqs(static_cast<std::size_t>(n_clusters), 0);
        // 第2パス: 各クラスタのサンプル数を集計
        for (Eigen::Index i = 0; i < n_samples; ++i)
        {
          int k = encoded(i);
          label_freqs[static_cast<std::size_t>(k)]++;
        }
        return label_freqs;
      }

      /**
       * @brief 特徴行列からペアワイズユークリッド距離行列を計算
       * @param X 特徴行列
       * @return ペアワイズユークリッド距離行列
       */
      inline Eigen::MatrixXd
      pairwise_euclidean_distances(const Eigen::Ref<const Eigen::MatrixXd> &X)
      {
        const Eigen::Index n = X.rows();
        Eigen::VectorXd sq_norms = X.rowwise().squaredNorm();
        // ||x_i - x_j||^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i'x_j で二乗距離を一括計算
        Eigen::MatrixXd D = -2 * X * X.transpose();
        D.colwise() += sq_norms;
        D.rowwise() += sq_norms.transpose();
        D = D.cwiseMax(0).array().sqrt().matrix(); // 数値誤差で負になった要素を 0 にし、平方根で距離に

        return D;
      }
    } // namespace detail

    /**
     * @brief 各サンプルのシルエット係数を計算する
     *
     * シルエット係数は (b - a) / max(a, b)。a は同一クラスタ内の平均距離、
     * b はサンプルが属さないクラスタのうち最も近いクラスタへの平均距離。
     * クラスタサイズ 1 のサンプルは 0 とする。
     *
     * @param X データ行列 (n_samples, n_features)
     * @param labels 各サンプルのクラスタラベル (n_samples,)
     * @return 各サンプルのシルエット係数 (n_samples,)
     */
    inline Eigen::VectorXd
    silhouette_samples(const Eigen::Ref<const Eigen::MatrixXd> &X,
                       const Eigen::Ref<const Eigen::VectorXi> &labels)
    {
      const Eigen::Index n_samples = X.rows();
      const Eigen::Index n_features = X.cols();

      // 入力チェック
      if (n_samples == 0 || n_features == 0)
      {
        throw std::invalid_argument("silhouette: X must not be empty");
      }
      if (labels.size() != n_samples)
      {
        throw std::invalid_argument(
            "silhouette: labels size must equal number of samples");
      }

      // ラベルを 0..K-1 に正規化し、各クラスタのサンプル数を取得
      Eigen::VectorXi encoded;
      int n_clusters = 0;
      std::vector<int> label_freqs =
          detail::encode_labels(labels, encoded, n_clusters);
      detail::check_number_of_labels(n_clusters, n_samples);

      // ペアワイズ距離 D(i,j) を計算
      Eigen::MatrixXd D = detail::pairwise_euclidean_distances(X);

      // サンプル i からクラスタ k 内の全点への距離の和を累積（後で a_i, b_i の計算に使用）
      Eigen::MatrixXd cluster_distances =
          Eigen::MatrixXd::Zero(n_samples, n_clusters);
      for (Eigen::Index i = 0; i < n_samples; ++i)
      {
        for (Eigen::Index j = 0; j < n_samples; ++j)
        {
          int k = encoded(j);
          cluster_distances(i, k) += D(i, j);
        }
      }

      Eigen::VectorXd sil_samples(n_samples);
      const double inf = std::numeric_limits<double>::infinity();

      for (Eigen::Index i = 0; i < n_samples; ++i)
      {
        int li = encoded(i);
        int freq_li = label_freqs[static_cast<std::size_t>(li)];

        // a_i: 同一クラスタ内の他点との平均距離（自分は除くので分母は freq_li - 1）
        double a_i;
        if (freq_li <= 1)
        {
          a_i = 0.0; // クラスタサイズ 1 のときは定義に従い 0
        }
        else
        {
          double intra_sum = cluster_distances(i, li);
          a_i = intra_sum / (freq_li - 1);
        }

        // b_i: サンプル i が属さないクラスタのうち、最も近いクラスタへの平均距離
        double b_i = inf;
        for (int k = 0; k < n_clusters; ++k)
        {
          if (k == li)
          {
            continue;
          }
          int fk = label_freqs[static_cast<std::size_t>(k)];
          if (fk == 0)
          {
            continue;
          }
          double mean_k = cluster_distances(i, k) / fk;
          if (mean_k < b_i)
          {
            b_i = mean_k;
          }
        }
        if (b_i == inf)
        {
          b_i = 0.0; // 他クラスタが存在しない場合（通常は check で弾かれる）
        }

        // s_i = (b_i - a_i) / max(a_i, b_i)。分母 0 や非有限は 0 とする（singleton 等）
        double denom = (a_i > b_i) ? a_i : b_i;
        if (denom <= 0.0 || !std::isfinite(denom))
        {
          sil_samples(i) = 0.0;
        }
        else
        {
          sil_samples(i) = (b_i - a_i) / denom;
        }
      }

      return sil_samples;
    }

    /**
     * @brief 全サンプルのシルエット係数の平均を返す
     *
     * @param X データ行列 (n_samples, n_features)
     * @param labels 各サンプルのクラスタラベル (n_samples,)
     * @return 平均シルエット係数（スカラー）
     */
    inline double
    silhouette_score(const Eigen::Ref<const Eigen::MatrixXd> &X,
                     const Eigen::Ref<const Eigen::VectorXi> &labels)
    {
      Eigen::VectorXd samples = silhouette_samples(X, labels);

      return samples.mean();
    }

    /**
     * @brief 事前計算された距離行列から各サンプルのシルエット係数を計算する
     *
     * D は (n_samples, n_samples) の距離行列で、対角は 0 であること。
     *
     * @param D ペアワイズ距離行列 (n_samples, n_samples)
     * @param labels 各サンプルのクラスタラベル (n_samples,)
     * @return 各サンプルのシルエット係数 (n_samples,)
     */
    inline Eigen::VectorXd
    silhouette_samples_precomputed(const Eigen::Ref<const Eigen::MatrixXd> &D,
                                   const Eigen::Ref<const Eigen::VectorXi> &labels)
    {
      const Eigen::Index n_samples = D.rows();

      // 入力チェック（D は正方かつ対角 0 を想定）
      if (D.cols() != n_samples)
      {
        throw std::invalid_argument(
            "silhouette: precomputed distance matrix must be square");
      }
      if (labels.size() != n_samples)
      {
        throw std::invalid_argument(
            "silhouette: labels size must equal number of samples");
      }

      Eigen::VectorXi encoded;
      int n_clusters = 0;
      std::vector<int> label_freqs =
          detail::encode_labels(labels, encoded, n_clusters);
      detail::check_number_of_labels(n_clusters, n_samples);

      // サンプル i からクラスタ k 内の全点への距離の和を累積
      Eigen::MatrixXd cluster_distances =
          Eigen::MatrixXd::Zero(n_samples, n_clusters);
      for (Eigen::Index i = 0; i < n_samples; ++i)
      {
        for (Eigen::Index j = 0; j < n_samples; ++j)
        {
          int k = encoded(j);
          cluster_distances(i, k) += D(i, j);
        }
      }

      Eigen::VectorXd sil_samples(n_samples);
      const double inf = std::numeric_limits<double>::infinity();

      for (Eigen::Index i = 0; i < n_samples; ++i)
      {
        int li = encoded(i);
        int freq_li = label_freqs[static_cast<std::size_t>(li)];

        // a_i: 同一クラスタ内の他点との平均距離（自分を除く）
        double a_i;
        if (freq_li <= 1)
        {
          a_i = 0.0;
        }
        else
        {
          double intra_sum = cluster_distances(i, li);
          a_i = intra_sum / (freq_li - 1);
        }

        // b_i: 最も近い他クラスタへの平均距離
        double b_i = inf;
        for (int k = 0; k < n_clusters; ++k)
        {
          if (k == li)
          {
            continue;
          }
          int fk = label_freqs[static_cast<std::size_t>(k)];
          if (fk == 0)
          {
            continue;
          }
          double mean_k = cluster_distances(i, k) / fk;
          if (mean_k < b_i)
          {
            b_i = mean_k;
          }
        }
        if (b_i == inf)
        {
          b_i = 0.0;
        }

        // s_i = (b_i - a_i) / max(a_i, b_i)
        double denom = (a_i > b_i) ? a_i : b_i;
        if (denom <= 0.0 || !std::isfinite(denom))
        {
          sil_samples(i) = 0.0;
        }
        else
        {
          sil_samples(i) = (b_i - a_i) / denom;
        }
      }

      return sil_samples;
    }

    /**
     * @brief 事前計算された距離行列から平均シルエット係数を返す
     * @param D ペアワイズ距離行列 (n_samples, n_samples)
     * @param labels 各サンプルのクラスタラベル (n_samples,)
     * @return 平均シルエット係数（スカラー）
     */
    inline double
    silhouette_score_precomputed(const Eigen::Ref<const Eigen::MatrixXd> &D,
                                 const Eigen::Ref<const Eigen::VectorXi> &labels)
    {
      Eigen::VectorXd samples = silhouette_samples_precomputed(D, labels);
      return samples.mean();
    }

  } // namespace metrics
} // namespace gprcpp
