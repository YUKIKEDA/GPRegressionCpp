#pragma once
#include <string>
#include <iostream>
#include <Eigen/Dense>

namespace gprcpp
{
  namespace kernels
  {
    /**
     * @brief カーネルの基底クラス
     */
    class Kernel
    {
    public:
      /**
       * @brief 仮想デストラクタ
       */
      virtual ~Kernel() = default;

      /**
       * カーネル行列 K(X, Y) を計算
       * @param x1 学習データ等 (n_samples_x1, n_features)
       * @param x2 ターゲットデータ (n_samples_x2, n_features)。空の場合は x1 とみなす
       * @return カーネル行列
       */
      virtual Eigen::MatrixXd operator()(
          const Eigen::MatrixXd &x1,
          const Eigen::MatrixXd &x2 = Eigen::MatrixXd()) const = 0;

      /**
       * @brief カーネル行列の対角成分のみ計算 K(x, x) (計算効率化のため)
       * @param x 入力ベクトル
       * @return カーネルの対角要素
       */
      virtual Eigen::VectorXd diag(const Eigen::MatrixXd &x) const = 0;

      /**
       * @brief カーネルが定常かどうか
       * @return カーネルが定常かどうか
       */
      virtual bool is_stationary() const = 0;

      /**
       * @brief カーネルのハイパーパラメータを取得 (thetaは対数スケール)
       *
       * ハイパーパラメータは対数スケール(theta)で扱う。
       * - 正値制約: length_scale 等は正である必要がある。theta = log(値) とすれば
       *   theta は任意の実数でよく、exp(theta) が常に正になるため制約付き最適化が不要。
       * - 探索空間: パラメータが 1e-3〜1e3 のように桁で変わるため、対数スケールの方が
       *   等しいステップが「倍率」として等しくなり探索しやすい。
       * - 最適化: 無制約の L-BFGS 等をそのまま使える。scikit-learn のカーネルも同様。
       *
       * @return カーネルのハイパーパラメータ theta（内部の正の値の log）
       */
      virtual Eigen::VectorXd get_hyperparameters() const = 0;

      /**
       * @brief カーネルのハイパーパラメータを設定 (thetaは対数スケール)
       * @param hyperparameters カーネルのハイパーパラメータ theta（内部では exp(theta) で保持）
       */
      virtual void set_hyperparameters(const Eigen::VectorXd &hyperparameters) = 0;

      /**
       * @brief カーネルのハイパーパラメータの数を取得
       * @return カーネルのハイパーパラメータの数
       */
      virtual int num_hyperparameters() const = 0;

      /**
       * @brief カーネルのクローン
       * @return カーネルのクローン
       */
      virtual std::shared_ptr<Kernel> clone() const = 0;

      /**
       * @brief カーネルの文字列表現
       * @return カーネルの文字列表現
       */
      virtual std::string to_string() const = 0;
    };
  }
}