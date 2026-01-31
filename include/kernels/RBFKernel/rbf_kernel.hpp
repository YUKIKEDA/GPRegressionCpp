#pragma once
#include "kernels/kernel.hpp"

namespace gprcpp
{
  namespace kernels
  {
    /**
     * @brief RBFカーネル（等方・異方両対応）
     * length_scale がスカラーなら等方、ベクトルなら次元ごとの長さスケール（ARD）。
     */
    class RBF : public Kernel
    {
    private:
      Eigen::VectorXd length_scale_; ///< 長さスケール（size==1: 等方, size==n_features: 異方）
      double lower_bound_;           ///< ハイパーパラメータの下限
      double upper_bound_;           ///< ハイパーパラメータの上限

      /**
       * @brief 入力を行ごとに length_scale でスケール（等方なら全次元同じ、異方なら次元別）
       * @param X 入力ベクトル
       * @return スケール済み入力ベクトル
       */
      Eigen::MatrixXd scale_input(const Eigen::MatrixXd &X) const
      {
        if (length_scale_.size() == 1)
        {
          return X / length_scale_(0);
        }
        return X.array().rowwise() / length_scale_.array().transpose();
      }

      /**
       * @brief ユークリッド距離の二乗行列を計算するヘルパー（スケール済み入力用）
       * @param X 入力ベクトル1
       * @param Y 入力ベクトル2
       * @return ユークリッド距離の二乗行列
       */
      static Eigen::MatrixXd sq_dist_impl(const Eigen::MatrixXd &X, const Eigen::MatrixXd &Y)
      {
        Eigen::VectorXd a = X.rowwise().squaredNorm();
        Eigen::VectorXd b = Y.rowwise().squaredNorm();
        return a.replicate(1, Y.rows()) + b.replicate(1, X.rows()).transpose() - 2.0 * (X * Y.transpose());
      }

    public:
      /**
       * @brief コンストラクタ（等方: 全次元で同じ長さスケール）
       * @param l 長さスケール（スカラー）
       * @param lower ハイパーパラメータの下限
       * @param upper ハイパーパラメータの上限
       */
      RBF(double l = 1.0, double lower = 1e-5, double upper = 1e5)
          : length_scale_(Eigen::VectorXd::Constant(1, l)), lower_bound_(lower), upper_bound_(upper) {}

      /**
       * @brief コンストラクタ（異方: 次元ごとの長さスケール、ARD）
       * @param l 長さスケール（n_features 次元ベクトル）
       * @param lower ハイパーパラメータの下限
       * @param upper ハイパーパラメータの上限
       */
      RBF(const Eigen::VectorXd &l, double lower = 1e-5, double upper = 1e5)
          : length_scale_(l), lower_bound_(lower), upper_bound_(upper) {}

      /**
       * @brief RBFカーネルのカーネル行列を計算
       * @param X 入力ベクトル1
       * @param Y 入力ベクトル2
       * @return カーネル行列
       */
      Eigen::MatrixXd operator()(
          const Eigen::MatrixXd &X,
          const Eigen::MatrixXd &Y = Eigen::MatrixXd()) const override
      {
        const Eigen::MatrixXd &targetY = (Y.size() == 0) ? X : Y;
        Eigen::MatrixXd Xs = scale_input(X);
        Eigen::MatrixXd Ys = scale_input(targetY);
        Eigen::MatrixXd D = sq_dist_impl(Xs, Ys);
        return (-0.5 * D).array().exp();
      }

      /**
       * @brief RBFカーネルの対角要素を計算
       * @param X 入力ベクトル
       * @return 対角要素
       */
      Eigen::VectorXd diag(const Eigen::MatrixXd &X) const override
      {
        return Eigen::VectorXd::Ones(X.rows()); // exp(0) = 1
      }

      /**
       * @brief RBFカーネルが定常かどうか
       * @return RBFカーネルが定常かどうか
       */
      bool is_stationary() const override
      {
        return true;
      }

      /**
       * @brief RBFカーネルのハイパーパラメータの数を取得
       * @return 等方なら 1、異方なら n_features
       */
      int num_hyperparameters() const override
      {
        return static_cast<int>(length_scale_.size());
      }

      /**
       * @brief ハイパーパラメータ theta の探索境界を取得（対数スケール）
       * 元の length_scale の (lower_bound_, upper_bound_) を log 変換した値を返す
       */
      std::pair<Eigen::VectorXd, Eigen::VectorXd> get_hyperparameter_bounds() const override
      {
        int n = static_cast<int>(length_scale_.size());
        Eigen::VectorXd lower = Eigen::VectorXd::Constant(n, std::log(lower_bound_));
        Eigen::VectorXd upper = Eigen::VectorXd::Constant(n, std::log(upper_bound_));
        return {lower, upper};
      }

      /**
       * @brief RBFカーネルのハイパーパラメータを取得（対数スケール）
       * @return theta = log(length_scale)
       */
      Eigen::VectorXd get_hyperparameters() const override
      {
        return length_scale_.array().log();
      }

      /**
       * @brief RBFカーネルのハイパーパラメータを設定（対数スケール）
       * @param theta theta.size() は length_scale_.size() と一致すること
       */
      void set_hyperparameters(const Eigen::VectorXd &theta) override
      {
        if (theta.size() != length_scale_.size())
        {
          return;
        }
        length_scale_ = theta.array().exp();
      }

      /**
       * @brief RBFカーネルのクローン
       * @return RBFカーネルのクローン
       */
      std::shared_ptr<Kernel> clone() const override
      {
        return std::make_shared<RBF>(*this);
      }

      /**
       * @brief RBFカーネルの文字列表現
       * @return RBFカーネルの文字列表現
       */
      std::string to_string() const override
      {
        if (length_scale_.size() == 1)
        {
          return "RBF(length_scale=" + std::to_string(length_scale_(0)) + ")";
        }
        std::string s = "RBF(length_scale=[";
        for (Eigen::Index i = 0; i < length_scale_.size(); ++i)
        {
          s += (i ? ", " : "") + std::to_string(length_scale_(i));
        }
        return s + "])";
      }

      /**
       * @brief 等方カーネルかどうか
       * @return 等方カーネルかどうか
       */
      bool is_isotropic() const
      {
        return length_scale_.size() == 1;
      }
    };
  }
}