#pragma once
#include "kernels/kernel.hpp"

namespace gprcpp
{
  namespace kernels
  {
    /**
     * @brief 定数カーネル
     */
    class ConstantKernel : public Kernel
    {
    private:
      double constant_value_; ///< 定数値
      double lower_bound_;    ///< ハイパーパラメータの下限
      double upper_bound_;    ///< ハイパーパラメータの上限

    public:
      /**
       * @brief コンストラクタ
       * @param value 定数値
       * @param lower_bound ハイパーパラメータの下限
       * @param upper_bound ハイパーパラメータの上限
       */
      ConstantKernel(double value = 1.0, double lower_bound = 1.0e-5, double upper_bound = 1.0e5)
          : constant_value_(value), lower_bound_(lower_bound), upper_bound_(upper_bound)
      {
      }

      /**
       * @brief 定数カーネルのカーネル行列を計算
       * @param x1 入力ベクトル1
       * @param x2 入力ベクトル2
       * @return カーネル行列
       */
      Eigen::MatrixXd operator()(
          const Eigen::MatrixXd &x1,
          const Eigen::MatrixXd &x2 = Eigen::MatrixXd()) const override
      {
        Eigen::Index rows = x1.rows();
        Eigen::Index cols = (x2.size() == 0) ? x1.rows() : x2.rows();

        return Eigen::MatrixXd::Constant(rows, cols, constant_value_);
      }

      /**
       * @brief 定数カーネルの対角要素を計算
       * @param X 入力ベクトル
       * @return 対角要素
       */
      Eigen::VectorXd diag(const Eigen::MatrixXd &X) const override
      {
        return Eigen::VectorXd::Constant(X.rows(), constant_value_);
      }

      /**
       * @brief 定数カーネルが定常かどうか
       * @return 定数カーネルが定常かどうか
       */
      bool is_stationary() const override
      {
        return true;
      }

      /**
       * @brief 定数カーネルのハイパーパラメータの数を取得
       * @return 定数カーネルのハイパーパラメータの数
       */
      int num_hyperparameters() const override
      {
        return 1;
      }

      /**
       * @brief ハイパーパラメータ theta の探索境界を取得（対数スケール）
       * 元の constant_value の (lower_bound_, upper_bound_) を log 変換した値を返す
       */
      std::pair<Eigen::VectorXd, Eigen::VectorXd> get_hyperparameter_bounds() const override
      {
        Eigen::VectorXd lower(1), upper(1);
        lower(0) = std::log(lower_bound_);
        upper(0) = std::log(upper_bound_);
        return {lower, upper};
      }

      /**
       * @brief 定数カーネルのハイパーパラメータを取得
       * @return 定数カーネルのハイパーパラメータ
       */
      Eigen::VectorXd get_hyperparameters() const override
      {
        return Eigen::VectorXd::Constant(1, std::log(constant_value_));
      }

      /**
       * @brief 定数カーネルのハイパーパラメータを設定
       * @param theta 定数カーネルのハイパーパラメータ
       */
      void set_hyperparameters(const Eigen::VectorXd &theta) override
      {
        constant_value_ = std::exp(theta(0));
      }

      /**
       * @brief 定数カーネルのクローン
       * @return 定数カーネルのクローン
       */
      std::shared_ptr<Kernel> clone() const override
      {
        return std::make_shared<ConstantKernel>(*this);
      }

      /**
       * @brief 定数カーネルの文字列表現
       * @return 定数カーネルの文字列表現
       */
      std::string to_string() const override
      {
        return std::to_string(constant_value_) + "^2";
      }
    };
  }
}