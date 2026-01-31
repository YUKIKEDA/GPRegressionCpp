# ガウス過程回帰 (Gaussian Process Regression)

## 目次

1. [概要](#概要)
2. [数学的定式化](#数学的定式化)
3. [事後分布と予測](#事後分布と予測)
4. [対数周辺尤度とハイパーパラメータ最適化](#対数周辺尤度とハイパーパラメータ最適化)
5. [数値計算の流れ](#数値計算の流れ)
6. [実装との対応](#実装との対応)
7. [使用例](#使用例)
8. [参考文献](#参考文献)

---

## 概要

**ガウス過程回帰（GPR, Gaussian Process Regression）**は、関数 $f(\mathbf{x})$ を**ガウス過程（GP）**としてモデル化し、有限個の観測データ $(X, \mathbf{y})$ に基づいて、任意の入力 $\mathbf{x}_*$ における出力の**事後分布**（平均・分散）を求める回帰手法である。カーネル関数により関数の滑らかさや構造を指定し、観測ノイズを対角項（$\alpha$）で表現する。**ベイズ的な枠組み**のもとで、予測の不確実性（分散）も自然に得られる。本実装では、カーネル（例: RBF、定数カーネル・ホワイトノイズとの組み合わせ可）、観測ノイズ $\alpha$、およびオプションでターゲットの正規化（`normalize_y`）を備え、ハイパーパラメータは**対数周辺尤度の最大化**により最適化する（Dual Annealing や Nelder-Mead 等のオプティマイザを使用）。

### 主な特徴

- **非パラメトリック**: 関数形を固定せず、カーネルで共分散構造を指定
- **不確実性の定量化**: 予測平均に加え、予測分散（標準偏差）および共分散を出力
- **ベイズ推定**: 事前分布（GP）→ 尤度（観測ノイズ）→ 事後分布が解析的に計算可能
- **ハイパーパラメータ最適化**: 対数周辺尤度を最大化してカーネルパラメータを推定
- **多出力対応**: ターゲットが複数列（多ターゲット）の場合も同一カーネルで各ターゲットごとに独立に事後を計算

---

## 数学的定式化

### 記号

- $\mathcal{X} \subseteq \mathbb{R}^d$: 入力空間
- $X = (\mathbf{x}_1, \ldots, \mathbf{x}_n)^\top \in \mathbb{R}^{n \times d}$: 学習用入力（$n$ サンプル）
- $\mathbf{y} = (y_1, \ldots, y_n)^\top \in \mathbb{R}^n$: 学習用ターゲット（スカラー出力の場合）
- $X_* \in \mathbb{R}^{n_* \times d}$: 予測したい入力（クエリ点）
- $k(\cdot, \cdot)$: カーネル関数（正定値カーネル）
- $\alpha$: 観測ノイズに対応する対角項（スカラーまたはサンプルごと）

### 事前分布（ガウス過程）

関数 $f: \mathcal{X} \to \mathbb{R}$ を、平均関数 $m(\mathbf{x})$ とカーネル $k$ によるガウス過程として定義する：

$$
f(\mathbf{x}) \sim \mathcal{GP}\bigl(m(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}')\bigr)
$$

本実装では**平均関数は $m(\mathbf{x}) \equiv 0$** とする（オプションでターゲットを平均 0・分散 1 に正規化することで、実質的に平均を考慮したモデルになる）。したがって、任意の有限個の点 $\mathbf{x}_1, \ldots, \mathbf{x}_N$ に対する $f$ の値のベクトル $\mathbf{f} = (f(\mathbf{x}_1), \ldots, f(\mathbf{x}_N))^\top$ は、

$$
\mathbf{f} \sim \mathcal{N}(\mathbf{0},\, K)
$$

とする。ここで $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ はカーネル行列である。

### 尤度（観測モデル）

観測値 $y_i$ は、真の関数値 $f(\mathbf{x}_i)$ に**独立なガウスノイズ**を加えたものとする：

$$
y_i = f(\mathbf{x}_i) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0,\, \sigma_n^2)
$$

サンプルごとに異なるノイズ分散を許す場合は $\mathrm{Var}(\varepsilon_i) = \alpha_i$ とする。ベクトル表記では、

$$
\mathbf{y} \mid \mathbf{f} \sim \mathcal{N}(\mathbf{f},\, \Lambda)
$$

ここで $\Lambda = \mathrm{diag}(\alpha_1, \ldots, \alpha_n)$（スカラー $\alpha$ の場合は $\Lambda = \alpha I$）。$\mathbf{f}$ を積分消去すると、

$$
\mathbf{y} \mid X \sim \mathcal{N}\bigl(\mathbf{0},\, K(X,X) + \Lambda\bigr)
$$

となる。$K(X,X)$ の $(i,j)$ 要素は $k(\mathbf{x}_i, \mathbf{x}_j)$ である。

---

## 事後分布と予測

### 同時分布

学習入力 $X$ とクエリ入力 $X_*$ における関数値を $\mathbf{f}$ と $\mathbf{f}_*$ とすると、事前より

$$
\begin{pmatrix} \mathbf{f} \\ \mathbf{f}_* \end{pmatrix}
\sim \mathcal{N}\left(
  \mathbf{0},\,
  \begin{pmatrix} K(X,X) & K(X,X_*) \\ K(X_*,X) & K(X_*,X_*) \end{pmatrix}
\right)
$$

である。観測は $\mathbf{y} = \mathbf{f} + \boldsymbol{\varepsilon}$、$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \Lambda)$ なので、

$$
\begin{pmatrix} \mathbf{y} \\ \mathbf{f}_* \end{pmatrix}
\sim \mathcal{N}\left(
  \mathbf{0},\,
  \begin{pmatrix} K(X,X) + \Lambda & K(X,X_*) \\ K(X_*,X) & K(X_*,X_*) \end{pmatrix}
\right)
$$

### 事後分布（予測分布）

条件付きガウス分布の公式から、$\mathbf{f}_* \mid \mathbf{y}, X, X_*$ はガウス分布であり、その**平均**と**共分散**は、

$$
\begin{aligned}
  \boldsymbol{\mu}_* &= K(X_*, X) \bigl( K(X,X) + \Lambda \bigr)^{-1} \mathbf{y}, \\
  \Sigma_* &= K(X_*, X_*) - K(X_*, X) \bigl( K(X,X) + \Lambda \bigr)^{-1} K(X, X_*).
\end{aligned}
$$

- **予測平均** $\boldsymbol{\mu}_*$: クエリ点 $X_*$ における事後期待値。実装では $K^{-1}\mathbf{y}$ を**双対係数** $\boldsymbol{\alpha}$ として保持し、$\boldsymbol{\mu}_* = K(X_*, X) \, \boldsymbol{\alpha}$ で計算する。
- **予測共分散** $\Sigma_*$: クエリ点どうしの事後共分散。対角成分が各点の**予測分散**、その平方根が**予測標準偏差**となる。

$K + \Lambda$ を **Cholesky 分解** $K + \Lambda = L L^\top$ すると、$L$ を用いて逆行列を陽に作らずに線形方程式を解くことで、数値的に安定に $\boldsymbol{\alpha} = (K+\Lambda)^{-1}\mathbf{y}$ および $\Sigma_*$ の対角（予測分散）を計算できる。

---

## 対数周辺尤度とハイパーパラメータ最適化

### 対数周辺尤度

カーネルのハイパーパラメータを $\boldsymbol{\theta}$ とすると、周辺尤度は

$$
p(\mathbf{y} \mid X, \boldsymbol{\theta}) = \mathcal{N}\bigl(\mathbf{y}; \mathbf{0},\, K_{\boldsymbol{\theta}}(X,X) + \Lambda\bigr).
$$

その対数は（スカラー出力の場合）

$$
\log p(\mathbf{y} \mid X, \boldsymbol{\theta})
= -\frac{1}{2} \mathbf{y}^\top (K + \Lambda)^{-1} \mathbf{y}
   - \frac{1}{2} \log \det(K + \Lambda)
   - \frac{n}{2} \log(2\pi).
$$

Cholesky 分解 $K + \Lambda = L L^\top$ を用いると、

$$
\log \det(K + \Lambda) = 2 \sum_i \log L_{ii},
\quad
\mathbf{y}^\top (K+\Lambda)^{-1} \mathbf{y} = \mathbf{y}^\top L^{-\top} L^{-1} \mathbf{y} = \|L^{-1}\mathbf{y}\|^2.
$$

実装では $L^{-1}\mathbf{y}$ を線形方程式の解として求める。**多ターゲット**の場合は、各ターゲット列について上記の二次形式と $\log\det$ を足し合わせる（カーネルは共通、ノイズは共通と仮定）。

### ハイパーパラメータの推定

ハイパーパラメータ $\boldsymbol{\theta}$（例: RBF の長さスケールの対数）は、**対数周辺尤度を最大化**するように推定する：

$$
\hat{\boldsymbol{\theta}} = \arg\max_{\boldsymbol{\theta}} \log p(\mathbf{y} \mid X, \boldsymbol{\theta}).
$$

本実装では、この最大化を**最小化**に置き換え（目的関数を $-\log p(\mathbf{y}|X,\boldsymbol{\theta})$ とする）、Dual Annealing などの大域的最適化オプティマイザで最小化する。必要に応じて `n_restarts_optimizer` で複数初期値から再起動し、最も尤度の高い解を採用する。

---

## 数値計算の流れ

### 学習（fit）

1. **ターゲットの正規化**（`normalize_y == true` のとき）: 各列を平均 0・分散 1 にし、平均と標準偏差を保持。
2. **ハイパーパラメータ最適化**: 対数周辺尤度を最大化する $\boldsymbol{\theta}$ をオプティマイザで探索。複数再起動の場合はランダム初期値から実行し、最良の $\boldsymbol{\theta}$ を採用。
3. **カーネル行列の構築**: $K = k(X, X)$、対角に $\alpha$ を加えて $K + \Lambda$ を形成。
4. **Cholesky 分解**: $K + \Lambda = L L^\top$ を計算。
5. **双対係数**: $\boldsymbol{\alpha} = (K+\Lambda)^{-1} \mathbf{y}$ を $L$ を用いた線形方程式で解き、保持。

### 予測（predict）

1. **事後平均**: $\boldsymbol{\mu}_* = K(X_*, X) \, \boldsymbol{\alpha}$。正規化していた場合は、スケール・シフトを復元。
2. **事後共分散**: $V = L^{-1} K(X, X_*)^\top$ を解き、$\Sigma_* = K(X_*, X_*) - V^\top V$。正規化時はターゲットの分散でスケール。
3. **予測標準偏差**: $\mathrm{diag}(\Sigma_*)$ の平方根。数値誤差で負になった場合は 0 にクリップ。

### サンプリング（sample_y）

事後分布 $\mathbf{f}_* \mid \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_*, \Sigma_*)$ からサンプルを生成する。$\Sigma_* = L_{\mathrm{cov}} L_{\mathrm{cov}}^\top$ と Cholesky 分解し、$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$ として $\boldsymbol{\mu}_* + L_{\mathrm{cov}} \mathbf{z}$ を返す。

---

## 実装との対応

| 項目 | 数学 | 実装 |
|------|------|------|
| 事前平均 | $m(\mathbf{x}) = 0$ | 正規化時は学習データで平均 0・分散 1 に変換 |
| カーネル | $k(\mathbf{x}, \mathbf{x}')$ | `kernel_`（例: `kernels::RBF`） |
| 観測ノイズ | $\Lambda = \mathrm{diag}(\alpha)$ | `alpha_`（スカラーまたはサンプルごと） |
| カーネル行列 | $K(X,X) + \Lambda$ | `K = (*kernel_)(X_train_); K.diagonal() += alpha_` |
| Cholesky | $K + \Lambda = L L^\top$ | `L_`（`Eigen::LLT`） |
| 双対係数 | $\boldsymbol{\alpha} = (K+\Lambda)^{-1}\mathbf{y}$ | `alpha_dual_` |
| 予測平均 | $\boldsymbol{\mu}_* = K(X_*, X) \boldsymbol{\alpha}$ | `K_trans * alpha_dual_` のあと正規化復元 |
| 予測共分散 | $\Sigma_* = K(X_*,X_*) - K(X_*,X)(K+\Lambda)^{-1}K(X,X_*)$ | `V = L_.solve(K_trans'); kernel_part = K(X) - V'*V` |
| 対数周辺尤度 | $\log p(\mathbf{y}|X,\boldsymbol{\theta})$ | `log_marginal_likelihood_impl` |
| ハイパーパラメータ最適化 | $\arg\max_\theta \log p(\mathbf{y}|X,\theta)$ | `optimizer_->minimize(-log_marginal_likelihood, opts)` |

- **normalize_y**: 学習時にターゲットを平均 0・分散 1 にし、予測時に `y_train_mean_` と `y_train_std_` で元のスケールに戻す。
- **多ターゲット**: `y` が行列 $(n \times n_{\mathrm{targets}})$ の場合、同じカーネル・同じ $\alpha$ で各列ごとに双対係数と予測平均・共分散を計算する。

---

## 使用例

### 基本的な学習と予測

```cpp
#include "gp_regression.hpp"
#include "kernels/RBFKernel/rbf_kernel.hpp"
#include "optimize/DualAnnealing/dual_annealing.hpp"

// カーネルとオプティマイザを用意
auto kernel = std::make_shared<gprcpp::kernels::RBF>(1.0);
auto optimizer = std::make_shared<gprcpp::optimize::DualAnnealing>();

gprcpp::regressor::GaussianProcessRegressor<> gpr(
    kernel, optimizer,
    1e-10,   // alpha
    0,       // n_restarts_optimizer
    true,    // normalize_y
    42       // random_state
);

// 学習
Eigen::MatrixXd X_train(n_samples, n_features);
Eigen::VectorXd y_train(n_samples);
gpr.fit(X_train, y_train);

// 予測（平均・標準偏差・共分散）
Eigen::MatrixXd X_query(n_query, n_features);
auto result = gpr.predict(X_query);
// result.mean, result.std, result.cov
```

### 対数周辺尤度とハイパーパラメータ

```cpp
double lml = gpr.log_marginal_likelihood_value();
Eigen::VectorXd theta = gpr.kernel()->get_hyperparameters();
```

### 事後からのサンプル

```cpp
Eigen::MatrixXd samples = gpr.sample_y(X_query, 5, 123);
// samples: (n_query, 5) の行列
```

### 多ターゲット（ベクトル出力）

```cpp
Eigen::MatrixXd y_multi(n_samples, n_targets);
gpr.fit(X_train, y_multi);
auto result = gpr.predict(X_query);
// result.mean: (n_query, n_targets), result.cov: ターゲットごとの共分散行列
```

---

## 参考文献

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- 石川 聡彦 (2020). 『ガウス過程と機械学習』. 講談社.

---

**関連ドキュメント**

- **カーネル**
  - [RBFカーネル](kernels/RBFKernel/README.md): 等方・ARD の定義と性質
  - [定数カーネル](kernels/ConstantKernel/README.md): オフセット・バイアス項（他カーネルと加算して使用）
  - [ホワイトノイズカーネル](kernels/WhiteNoiseKernel/README.md): 観測ノイズの表現（他カーネルと加算して使用）
- **オプティマイザ**（ハイパーパラメータ最適化）
  - [Dual Annealing](optimize/DualAnnealing/README.md): 大域的最適化アルゴリズム
  - [Nelder-Mead](optimize/NelderMead/README.md): シンプレックス法（導関数不要）
