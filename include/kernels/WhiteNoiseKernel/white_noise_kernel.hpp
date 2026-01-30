#pragma once
#include "kernels/kernel.hpp"

namespace gprcpp
{
  namespace kernels
  {
    /**
     * @brief ホワイトノイズカーネル
     */
    class WhiteKernel : public Kernel
    {
    private:
      double noise_level_; ///< ノイズレベル
      double lower_bound_; ///< ハイパーパラメータの下限
      double upper_bound_; ///< ハイパーパラメータの上限

    public:
      /**
       * @brief コンストラクタ
       * @param noise ノイズレベル
       * @param lower ハイパーパラメータの下限
       * @param upper ハイパーパラメータの上限
       */
      WhiteKernel(double noise = 1.0, double lower = 1e-5, double upper = 1e5)
          : noise_level_(noise), lower_bound_(lower), upper_bound_(upper) {}

      /**
       * @brief ホワイトノイズカーネルのカーネル行列を計算
       * @param x1 入力ベクトル1
       * @param x2 入力ベクトル2
       * @return カーネル行列
       */
      Eigen::MatrixXd operator()(
          const Eigen::MatrixXd &x1,
          const Eigen::MatrixXd &x2 = Eigen::MatrixXd()) const override
      {
        Eigen::Index nx = x1.rows();
        // Yが空、もしくはポインタが同じ場合は自己共分散（対角にノイズを乗せる）
        if (x2.size() == 0)
        {
          return Eigen::MatrixXd::Identity(nx, nx) * noise_level_;
        }
        else
        {
          // クロス共分散では通常ホワイトノイズは0 (観測位置が一致しない限り独立)
          return Eigen::MatrixXd::Zero(nx, x2.rows());
        }
      }

      /**
       * @brief ホワイトノイズカーネルの対角要素を計算
       * @param X 入力ベクトル
       * @return 対角要素
       */
      Eigen::VectorXd diag(const Eigen::MatrixXd &X) const override
      {
        return Eigen::VectorXd::Constant(X.rows(), noise_level_);
      }

      /**
       * @brief ホワイトノイズカーネルが定常かどうか
       * @return ホワイトノイズカーネルが定常かどうか
       */
      bool is_stationary() const override
      {
        return true;
      }

      /**
       * @brief ホワイトノイズカーネルのハイパーパラメータの数を取得
       * @return ホワイトノイズカーネルのハイパーパラメータの数
       */
      int num_hyperparameters() const override
      {
        return 1;
      }

      /**
       * @brief ホワイトノイズカーネルのハイパーパラメータを取得
       * @return ホワイトノイズカーネルのハイパーパラメータ
       */
      Eigen::VectorXd get_hyperparameters() const override
      {
        return Eigen::VectorXd::Constant(1, std::log(noise_level_));
      }

      /**
       * @brief ホワイトノイズカーネルのハイパーパラメータを設定
       * @param theta ホワイトノイズカーネルのハイパーパラメータ
       */
      void set_hyperparameters(const Eigen::VectorXd &theta) override
      {
        noise_level_ = std::exp(theta(0));
      }

      /**
       * @brief ホワイトノイズカーネルのクローン
       * @return ホワイトノイズカーネルのクローン
       */
      std::shared_ptr<Kernel> clone() const override
      {
        return std::make_shared<WhiteKernel>(*this);
      }

      /**
       * @brief ホワイトノイズカーネルの文字列表現
       * @return ホワイトノイズカーネルの文字列表現
       */
      std::string to_string() const override
      {
        return "WhiteKernel(" + std::to_string(noise_level_) + ")";
      }
    };
  }
}