# RBFカーネル (Radial Basis Function Kernel)

## 概要

RBFカーネル（**Radial Basis Function**、別名 **Squared Exponential kernel** や **Gaussian kernel**）は、ガウス過程において最も広く使われるカーネル関数の一つである。**入力間の距離**にのみ依存し、近い点ほど高い共分散、遠い点ほど低い共分散を与える。**定常カーネル**であり、滑らかな関数をモデル化するのに適している。本実装では**等方**（全次元で同じ長さスケール）と**異方・ARD**（次元ごとの長さスケール）の両方に対応している。

---

## 数学的定義

### 等方 RBF カーネル

スカラー長さスケール $\ell > 0$ を用いた等方 RBF カーネルは次のように定義される。

$$
k_{\mathrm{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left( -\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\ell^2} \right)
$$

ここで、

- $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^d$ ：任意の入力ベクトル
- $\|\cdot\|$ ：ユークリッドノルム
- $\ell > 0$ ：ハイパーパラメータ（**長さスケール**, length scale）

**同一入力** $\mathbf{x} = \mathbf{x}'$ のとき $k = 1$、距離が大きくなるほど $k \to 0$ となる。

等価な表現として、スケール済み距離 $r = \|\mathbf{x} - \mathbf{x}'\|/\ell$ を用いると、

$$
k_{\mathrm{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left( -\frac{r^2}{2} \right), \quad r = \frac{\|\mathbf{x} - \mathbf{x}'\|}{\ell}
$$

となる。

### 異方 RBF カーネル（ARD）

**Automatic Relevance Determination (ARD)** では、次元ごとに長さスケール $\ell_i$ を設ける。$\boldsymbol{\ell} = (\ell_1, \ldots, \ell_d)^\top \in \mathbb{R}^d_+$ として、

$$
k_{\mathrm{RBF}}(\mathbf{x}, \mathbf{x}') = \exp\left( -\frac{1}{2} \sum_{i=1}^{d} \frac{(x_i - x_i')^2}{\ell_i^2} \right)
$$

となる。等方の場合は $\ell_1 = \cdots = \ell_d = \ell$ とすれば上記の等方形と一致する。異方にすることで、**重要でない次元**は $\ell_i$ が大きくなり（その次元方向の変化に鈍感）、**重要な次元**は $\ell_i$ が小さく保たれるため、特徴選択の解釈がしやすくなる。

---

## カーネル行列

### 二乗距離行列

実装では、スケール済み入力 $\tilde{\mathbf{x}} = \mathbf{x} / \boldsymbol{\ell}$（等方なら $\tilde{x}_i = x_i/\ell$、異方なら $\tilde{x}_i = x_i/\ell_i$）に対して、**二乗ユークリッド距離**の行列を計算する。

$n$ 個の入力 $X = (\mathbf{x}_1, \ldots, \mathbf{x}_n)^\top$ と $n'$ 個の入力 $Y = (\mathbf{y}_1, \ldots, \mathbf{y}_{n'})^\top$ に対し、スケール済みを $\tilde{X}, \tilde{Y}$ とすると、二乗距離行列 $D$ の $(i,j)$ 要素は、

$$
D_{ij} = \|\tilde{\mathbf{x}}_i - \tilde{\mathbf{y}}_j\|^2
$$

である。これを効率よく求める公式（展開形）は、

$$
D_{ij} = \|\tilde{\mathbf{x}}_i\|^2 + \|\tilde{\mathbf{y}}_j\|^2 - 2\,\tilde{\mathbf{x}}_i^\top \tilde{\mathbf{y}}_j
$$

であり、行列形式では $D = \mathbf{a}\mathbf{1}^\top + \mathbf{1}\mathbf{b}^\top - 2\tilde{X}\tilde{Y}^\top$（$\mathbf{a}_i = \|\tilde{\mathbf{x}}_i\|^2,\, \mathbf{b}_j = \|\tilde{\mathbf{y}}_j\|^2$）で計算できる。

### 自己共分散（同一入力 $X$）

カーネル行列 $K$ の $(i,j)$ 要素は、

$$
K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\frac{1}{2} D_{ij} \right)
$$

である。対角成分は $D_{ii} = 0$ より $K_{ii} = 1$ となる。

### クロス共分散（異なる入力 $X$ と $Y$）

$X$（$n$ 行）と $Y$（$n'$ 行）のクロス共分散行列も同様に、スケール済み二乗距離 $D$ を計算し、$K_{ij} = \exp(-D_{ij}/2)$ で与えられる。

---

## 性質

### 定常性

RBFカーネルは**定常カーネル**である。$k(\mathbf{x}, \mathbf{x}')$ が差 $\mathbf{x} - \mathbf{x}'$ のノルム（およびARDの場合は各成分の差）にのみ依存するためである。

### 正定値性

等方・異方ともに、有限個の相異なる点に対して構成したカーネル行列は**正定値**である。RBFカーネルは Bochner の定理の意味で正定値なフーリエ変換に対応するため、任意の相異なる点列で正定値となる。

### 滑らかさ

RBFカーネルは **infinitely differentiable**（無限回微分可能）な関数を誘導する。そのため、ガウス過程のサンプルパスも滑らかになり、連続的な現象のモデル化に適している。

### 長さスケール $\ell$ の意味

- **$\ell$ が大きい**：共分散が距離に対してゆるやかに減衰する。関数が「のんびり」変化し、より滑らかで大域的な傾向を捉えやすい。
- **$\ell$ が小さい**：近い点だけが強く相関し、遠い点は速く無相関になる。関数が「細かく」変化し、局所的な変動を捉えやすい。

ハイパーパラメータ $\ell$（または $\boldsymbol{\ell}$）は、通常は周辺尤度の最大化などでデータから推定する。

---

## 実装との対応

| 項目 | 数学 | 実装 |
|------|------|------|
| 等方 | $\ell$ スカラー | `RBF(double l)` |
| 異方（ARD） | $\boldsymbol{\ell}$ ベクトル | `RBF(const Eigen::VectorXd &l)` |
| カーネル値 | $\exp(-\|x-x'\|^2/(2\ell^2))$ | スケール入力 → 二乗距離 $D$ → $\exp(-0.5\,D)$ |
| 対角 | $k(\mathbf{x},\mathbf{x})=1$ | `diag(X)` → すべて 1 |
| 定常性 | 定常 | `is_stationary()` → `true` |
| ハイパーパラメータ | $\ell$ または $\boldsymbol{\ell}$ | `length_scale_`（内部） |
| 最適化用パラメータ | 対数スケール $\theta = \log \ell$ | `get_hyperparameters()` / `set_hyperparameters(theta)` |

- **scale_input**: 入力を行ごとに $\boldsymbol{\ell}$ で割り、スケール済み $\tilde{X}$ を生成する。
- **sq_dist_impl**: スケール済み入力に対して二乗ユークリッド距離行列 $D$ を、$D_{ij} = \|\tilde{\mathbf{x}}_i\|^2 + \|\tilde{\mathbf{y}}_j\|^2 - 2\tilde{\mathbf{x}}_i^\top\tilde{\mathbf{y}}_j$ の形で計算する。
- カーネル行列は `(-0.5 * D).array().exp()` により $K_{ij} = \exp(-D_{ij}/2)$ として計算している。

---

## 使用例

### 等方 RBF

```cpp
#include "kernels/RBFKernel/rbf_kernel.hpp"

gprcpp::kernels::RBF kernel(1.0);  // length_scale = 1.0
Eigen::MatrixXd K = kernel(X, Y);  // カーネル行列
Eigen::VectorXd d = kernel.diag(X); // 対角 → すべて 1
```

### 異方 RBF（ARD）

```cpp
Eigen::VectorXd length_scales(3);
length_scales << 1.0, 0.5, 2.0;
gprcpp::kernels::RBF kernel_ard(length_scales);
Eigen::MatrixXd K_ard = kernel_ard(X, Y);
```

### ハイパーパラメータの取得・設定

```cpp
Eigen::VectorXd theta = kernel.get_hyperparameters(); // log(length_scale)
kernel.set_hyperparameters(new_theta);
```

---

## 参考文献

- Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- 石川 聡彦 (2020). 『ガウス過程と機械学習』. 講談社.
