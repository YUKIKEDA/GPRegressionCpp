#include "kernels/kernel.hpp"
#include "kernels/ConstantKernel/constant_kernel.hpp"

namespace gprcpp
{
  namespace kernels
  {
    /**
     * @brief カーネルの演算子
     */
    class KernelOperator : public Kernel
    {
    protected:
      std::shared_ptr<Kernel> kernel1_; ///< カーネル1
      std::shared_ptr<Kernel> kernel2_; ///< カーネル2

    public:
      /**
       * @brief コンストラクタ
       * @param kernel1 カーネル1
       * @param kernel2 カーネル2
       */
      KernelOperator(std::shared_ptr<Kernel> kernel1, std::shared_ptr<Kernel> kernel2)
          : kernel1_(std::move(kernel1)), kernel2_(std::move(kernel2))
      {
      }

      /**
       * @brief カーネルが定常かどうか
       * @return カーネルが定常かどうか
       */
      bool is_stationary() const override
      {
        return kernel1_->is_stationary() && kernel2_->is_stationary();
      }

      /**
       * @brief カーネルのハイパーパラメータを取得
       * @return カーネルのハイパーパラメータ
       */
      Eigen::VectorXd get_hyperparameters() const override
      {
        Eigen::VectorXd theta1 = kernel1_->get_hyperparameters();
        Eigen::VectorXd theta2 = kernel2_->get_hyperparameters();
        Eigen::VectorXd theta = Eigen::VectorXd(theta1.size() + theta2.size());

        if (theta1.size() > 0)
        {
          theta.head(theta1.size()) = theta1;
        }
        if (theta2.size() > 0)
        {
          theta.tail(theta2.size()) = theta2;
        }
        return theta;
      }

      /**
       * @brief カーネルのハイパーパラメータを設定
       * @param theta カーネルのハイパーパラメータ
       */
      void set_hyperparameters(const Eigen::VectorXd &theta) override
      {
        int n1 = kernel1_->num_hyperparameters();
        int n2 = kernel2_->num_hyperparameters();

        if (theta.size() != n1 + n2)
        {
          std::cerr << "Error: Theta size mismatch in KernelOperator." << std::endl;
          return;
        }

        if (n1 > 0)
        {
          kernel1_->set_hyperparameters(theta.head(n1));
        }
        if (n2 > 0)
        {
          kernel2_->set_hyperparameters(theta.tail(n2));
        }
      }

      /**
       * @brief カーネルのハイパーパラメータの数を取得
       * @return カーネルのハイパーパラメータの数
       */
      int num_hyperparameters() const override
      {
        return kernel1_->num_hyperparameters() + kernel2_->num_hyperparameters();
      }
    };

    /**
     * @brief 和カーネル演算子
     */
    class SumKernelOperator : public KernelOperator
    {
    public:
      // 親クラスのコンストラクタを使用
      using KernelOperator::KernelOperator;

      /**
       * @brief 和カーネル演算子
       * @param x1 入力ベクトル1
       * @param x2 入力ベクトル2
       * @return 和カーネル演算子
       */
      Eigen::MatrixXd operator()(
          const Eigen::MatrixXd &x1,
          const Eigen::MatrixXd &x2 = Eigen::MatrixXd()) const override
      {
        return (*kernel1_)(x1, x2) + (*kernel2_)(x1, x2);
      }

      /**
       * @brief 和カーネル演算子の対角要素
       * @param x 入力ベクトル
       * @return 和カーネル演算子の対角要素
       */
      Eigen::VectorXd diag(const Eigen::MatrixXd &x) const override
      {
        return kernel1_->diag(x) + kernel2_->diag(x);
      }

      /**
       * @brief 和カーネル演算子のクローン
       * @return 和カーネル演算子のクローン
       */
      std::shared_ptr<Kernel> clone() const override
      {
        return std::make_shared<SumKernelOperator>(kernel1_->clone(), kernel2_->clone());
      }

      /**
       * @brief 和カーネル演算子の文字列表現
       * @return 和カーネル演算子の文字列表現
       */
      std::string to_string() const override
      {
        return "(" + kernel1_->to_string() + " + " + kernel2_->to_string() + ")";
      }
    };

    /**
     * @brief 積カーネル演算子
     */
    class ProductKernelOperator : public KernelOperator
    {
    public:
      // 親クラスのコンストラクタを使用
      using KernelOperator::KernelOperator;

      /**
       * @brief 積カーネル演算子
       * @param x1 入力ベクトル1
       * @param x2 入力ベクトル2
       * @return 積カーネル演算子
       */
      Eigen::MatrixXd operator()(
          const Eigen::MatrixXd &x1,
          const Eigen::MatrixXd &x2 = Eigen::MatrixXd()) const override
      {
        return (*kernel1_)(x1, x2).array() * (*kernel2_)(x1, x2).array();
      }

      /**
       * @brief 積カーネル演算子の対角要素
       * @param x 入力ベクトル
       * @return 積カーネル演算子の対角要素
       */
      Eigen::VectorXd diag(const Eigen::MatrixXd &x) const override
      {
        return kernel1_->diag(x).array() * kernel2_->diag(x).array();
      }

      /**
       * @brief 積カーネル演算子のクローン
       * @return 積カーネル演算子のクローン
       */
      std::shared_ptr<Kernel> clone() const override
      {
        return std::make_shared<ProductKernelOperator>(kernel1_->clone(), kernel2_->clone());
      }

      /**
       * @brief 積カーネル演算子の文字列表現
       * @return 積カーネル演算子の文字列表現
       */
      std::string to_string() const override
      {
        return "(" + kernel1_->to_string() + " * " + kernel2_->to_string() + ")";
      }
    };

    /**
     * @brief 和カーネル演算子の演算子オーバーロード
     * @param lhs 左辺
     * @param rhs 右辺
     * @return 和カーネル演算子
     */
    std::shared_ptr<Kernel> operator+(
        const std::shared_ptr<Kernel> &lhs,
        const std::shared_ptr<Kernel> &rhs)
    {
      return std::make_shared<SumKernelOperator>(lhs->clone(), rhs->clone());
    }

    /**
     * @brief 積カーネル演算子の演算子オーバーロード
     * @param lhs 左辺
     * @param rhs 右辺
     * @return 積カーネル演算子
     */
    std::shared_ptr<Kernel> operator*(
        const std::shared_ptr<Kernel> &lhs,
        const std::shared_ptr<Kernel> &rhs)
    {
      return std::make_shared<ProductKernelOperator>(lhs->clone(), rhs->clone());
    }

    /**
     * @brief doubleとの演算 (ConstantKernelへの自動変換)
     * @param val 値
     * @param rhs 右辺
     * @return 積カーネル演算子
     */
    std::shared_ptr<Kernel> operator*(double val, const std::shared_ptr<Kernel> &rhs)
    {
      return std::make_shared<ProductKernelOperator>(std::make_shared<ConstantKernel>(val), rhs->clone());
    }
    /**
     * @brief doubleとの演算 (ConstantKernelへの自動変換)
     * @param lhs 左辺
     * @param val 値
     * @return 積カーネル演算子
     */
    std::shared_ptr<Kernel> operator*(const std::shared_ptr<Kernel> &lhs, double val)
    {
      return std::make_shared<ProductKernelOperator>(lhs->clone(), std::make_shared<ConstantKernel>(val));
    }
    /**
     * @brief doubleとの演算 (ConstantKernelへの自動変換)
     * @param val 値
     * @param rhs 右辺
     * @return 和カーネル演算子
     */
    std::shared_ptr<Kernel> operator+(double val, const std::shared_ptr<Kernel> &rhs)
    {
      return std::make_shared<SumKernelOperator>(std::make_shared<ConstantKernel>(val), rhs->clone());
    }
    /**
     * @brief doubleとの演算 (ConstantKernelへの自動変換)
     * @param lhs 左辺
     * @param val 値
     * @return 和カーネル演算子
     */
    std::shared_ptr<Kernel> operator+(const std::shared_ptr<Kernel> &lhs, double val)
    {
      return std::make_shared<SumKernelOperator>(lhs->clone(), std::make_shared<ConstantKernel>(val));
    }
  }
}