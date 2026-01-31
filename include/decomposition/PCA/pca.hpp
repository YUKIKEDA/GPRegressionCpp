/**
 * @file pca.hpp
 * @brief 特異値分解（SVD）を用いた主成分分析（PCA）
 *
 * データを中心化した上で SVD を適用し、主成分軸・分散・変換結果を保持する。
 * scikit-learn の PCA(svd_solver='full') と同様の挙動を目指す。
 */

#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace gprcpp
{
  namespace decomposition
  {

    /**
     * @brief 特異値分解（SVD）を用いた主成分分析（PCA）
     *
     * 入力データは中心化されるが、各特徴量のスケールは変更しない。
     * n_components は保持する主成分数（1 以上 min(n_samples, n_features) 以下）。
     */
    class PCA
    {
    public:
      using MatrixXd = Eigen::MatrixXd;
      using VectorXd = Eigen::VectorXd;

      /**
       * @brief コンストラクタ
       * @param n_components 保持する主成分数。省略時は min(n_samples, n_features)
       * @param whiten True の場合、変換後の各成分の分散を 1 にする（白化）
       */
      explicit PCA(int n_components = -1, bool whiten = false)
          : n_components_(n_components), whiten_(whiten)
      {
        if (n_components_ < -1 || n_components_ == 0)
        {
          throw std::invalid_argument("pca: n_components must be -1 (auto) or >= 1");
        }
      }

      /**
       * @brief データにフィットし、主成分を計算する
       * @param X データ行列 (n_samples, n_features)。行がサンプル、列が特徴量
       * @return *this
       */
      PCA &fit(const Eigen::Ref<const MatrixXd> &X)
      {
        const Eigen::Index n_samples = X.rows();
        const Eigen::Index n_features = X.cols();
        if (n_samples == 0 || n_features == 0)
        {
          throw std::invalid_argument("pca: X must not be empty");
        }

        // n_components が -1 のときは全主成分を保持
        const Eigen::Index max_components = std::min(n_samples, n_features);
        int k = n_components_;
        if (k < 0)
        {
          k = static_cast<int>(max_components);
        }
        if (k < 1 || k > max_components)
        {
          throw std::invalid_argument(
              "pca: n_components must be between 1 and min(n_samples, n_features)");
        }

        n_samples_ = n_samples;
        n_features_ = n_features;
        n_components_ = k;

        // 列ごとの平均を取り、中心化（SVD は中心化データに対して行う）
        mean_ = X.colwise().mean();
        MatrixXd X_centered = X.rowwise() - mean_.transpose();

        // Thin SVD: X_centered ≈ U * S * V^T（U, V は正規直交、S は特異値）
        Eigen::JacobiSVD<MatrixXd> svd(
            X_centered,
            Eigen::ComputeThinU | Eigen::ComputeThinV);

        VectorXd S = svd.singularValues();
        MatrixXd U = svd.matrixU();
        MatrixXd V = svd.matrixV();

        // 主成分の符号を一意にする（scikit-learn 互換）
        svd_flip_(U, V, static_cast<Eigen::Index>(k));

        // 主成分軸は V の左 k 列の転置、特異値は上から k 個
        components_ = V.leftCols(k).transpose();
        singular_values_ = S.head(k);

        // 説明分散 = 特異値^2 / (n-1)（不偏分散の自由度）
        const double ddof = static_cast<double>(n_samples - 1);
        explained_variance_ = (singular_values_.array().square() / ddof).matrix();

        // 説明分散比 = 各成分の分散 / 全分散
        const Eigen::Index total_size = S.size();
        const double total_var =
            (S.array().square() / ddof).sum();
        if (total_var > 0)
        {
          explained_variance_ratio_ = (explained_variance_.array() / total_var).matrix();
        }
        else
        {
          explained_variance_ratio_ = VectorXd::Ones(k);
        }

        // 捨てた成分の分散の平均をノイズ分散として推定（k < rank のときのみ）
        if (k < total_size)
        {
          const Eigen::Index rest = total_size - k;
          double sum_rest = 0;
          for (Eigen::Index i = k; i < total_size; ++i)
          {
            sum_rest += S(i) * S(i) / ddof;
          }
          noise_variance_ = sum_rest / static_cast<double>(rest);
        }
        else
        {
          noise_variance_ = 0.0;
        }

        return *this;
      }

      /**
       * @brief フィットし、同じデータを低次元に変換する
       * @param X データ行列 (n_samples, n_features)
       * @return 変換後の行列 (n_samples, n_components)
       */
      MatrixXd fit_transform(const Eigen::Ref<const MatrixXd> &X)
      {
        fit(X);
        return transform(X);
      }

      /**
       * @brief 学習済み主成分でデータを変換する（fit 済みであること）
       * @param X データ行列 (n_samples, n_features)
       * @return 変換後の行列 (n_samples, n_components)
       */
      MatrixXd transform(const Eigen::Ref<const MatrixXd> &X) const
      {
        if (mean_.size() == 0)
        {
          throw std::runtime_error("pca: fit has not been called");
        }
        if (X.cols() != n_features_)
        {
          throw std::invalid_argument("pca: X.cols() must match n_features");
        }

        // fit 時の平均で中心化し、主成分軸への射影: X_centered * components_^T
        const Eigen::Index n = X.rows();
        MatrixXd X_centered = X.rowwise() - mean_.transpose();
        MatrixXd proj = X_centered * components_.transpose();

        // 白化時は各成分を sqrt(n-1)/特異値 でスケールし、分散を 1 にする
        if (whiten_)
        {
          const double scale = std::sqrt(static_cast<double>(n_samples_ - 1));
          for (Eigen::Index j = 0; j < n_components_; ++j)
          {
            proj.col(j) *= (scale / singular_values_(j));
          }
        }
        return proj;
      }

      /**
       * @brief 平均を返す
       * @return 平均
       */
      const VectorXd &mean() const { return mean_; }
      /**
       * @brief 主成分軸を返す
       * @return 主成分軸
       */
      const MatrixXd &components() const { return components_; }
      /**
       * @brief 特異値を返す
       * @return 特異値
       */
      const VectorXd &singular_values() const { return singular_values_; }
      /**
       * @brief 説明分散を返す
       * @return 説明分散
       */
      const VectorXd &explained_variance() const { return explained_variance_; }
      /**
       * @brief 説明分散比を返す
       * @return 説明分散比
       */
      const VectorXd &explained_variance_ratio() const { return explained_variance_ratio_; }
      /**
       * @brief ノイズ分散を返す
       * @return ノイズ分散
       */
      double noise_variance() const { return noise_variance_; }
      /**
       * @brief サンプル数を返す
       * @return サンプル数
       */
      Eigen::Index n_samples() const { return n_samples_; }
      /**
       * @brief 特徴量数を返す
       * @return 特徴量数
       */
      Eigen::Index n_features() const { return n_features_; }
      /**
       * @brief 保持する主成分数を返す
       * @return 保持する主成分数
       */
      int n_components() const { return n_components_; }

    private:
      int n_components_;                  ///< 保持する主成分数
      bool whiten_;                       ///< 白化を行うかどうか
      Eigen::Index n_samples_{0};         ///< サンプル数
      Eigen::Index n_features_{0};        ///< 特徴量数
      VectorXd mean_;                     ///< 平均
      MatrixXd components_;               ///< 主成分軸
      VectorXd singular_values_;          ///< 特異値
      VectorXd explained_variance_;       ///< 説明分散
      VectorXd explained_variance_ratio_; ///< 説明分散比
      double noise_variance_{0};          ///< ノイズ分散

      /**
       * @brief SVD の結果を flip する
       * @param U SVD の結果の U 行列
       * @param V SVD の結果の V 行列
       * @param k 保持する主成分数
       */
      void svd_flip_(MatrixXd &U, MatrixXd &V, Eigen::Index k)
      {
        // 各主成分について、V の列で絶対値最大の要素が正になるように U, V の符号を揃える
        for (Eigen::Index j = 0; j < k; ++j)
        {
          Eigen::Index i_max = 0;
          double max_abs = std::abs(V(0, j));
          for (Eigen::Index i = 1; i < V.rows(); ++i)
          {
            double a = std::abs(V(i, j));
            if (a > max_abs)
            {
              max_abs = a;
              i_max = i;
            }
          }
          double sign = (max_abs > 0 && V(i_max, j) < 0) ? -1.0 : 1.0;
          U.col(j) *= sign;
          V.col(j) *= sign;
        }
      }
    };
  }
}
