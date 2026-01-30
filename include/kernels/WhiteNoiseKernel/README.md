# ホワイトノイズカーネル (White Noise Kernel)

## 概要

ホワイトノイズカーネルは、ガウス過程において**観測ノイズ**を表現するためのカーネル関数である。**同じ入力位置**では共分散が正の定数（ノイズ分散）、**異なる入力位置**では共分散が 0 となる。つまり、異なる観測点のノイズは**互いに独立**であるとみなす。定常カーネルの一種であり、主に「観測誤差」や「ノイズ分散」をモデルに組み込むために、他のカーネル（RBF や定数カーネルなど）と**加算**して用いられる。

---

## 数学的定義

ホワイトノイズカーネルは次のように定義される。

$$
k_{\mathrm{White}}(\mathbf{x}, \mathbf{x}') = \sigma_n^2 \, \delta(\mathbf{x}, \mathbf{x}')
$$

ここで、

- $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^d$ ：任意の入力ベクトル
- $\sigma_n^2 > 0$ ：ハイパーパラメータ（ノイズ分散、ノイズレベル）
- $\delta(\mathbf{x}, \mathbf{x}')$ ：**クロネッカーのデルタ**。$\mathbf{x} = \mathbf{x}'$ のとき 1、それ以外は 0

したがって、

$$
k_{\mathrm{White}}(\mathbf{x}, \mathbf{x}') =
\begin{cases}
\sigma_n^2 & (\mathbf{x} = \mathbf{x}') \\
0 & (\mathbf{x} \neq \mathbf{x}')
\end{cases}
$$

**同じ点**では分散 $\sigma_n^2$、**異なる点**では共分散 0（無相関＝独立）を表す。

---

## カーネル行列

### 自己共分散（同一入力 $X$）

$n$ 個の入力 $X = (\mathbf{x}_1, \ldots, \mathbf{x}_n)^\top$ に対する**自己共分散**のカーネル行列 $K$ は、

$$
K = \sigma_n^2 \, I
$$

となる。$I$ は $n \times n$ の単位行列である。つまり、

$$
K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j) = \sigma_n^2 \, \delta_{ij} =
\begin{cases}
\sigma_n^2 & (i = j) \\
0 & (i \neq j)
\end{cases}
$$

**対角成分だけが $\sigma_n^2$、非対角成分は 0** の対角行列である。

### クロス共分散（異なる入力 $X$ と $X'$）

入力 $X$（$n$ 行）と別の入力 $X'$（$n'$ 行）に対するクロス共分散では、異なる位置のノイズは独立なので、

$$
K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j') = 0 \quad \forall i, j
$$

となり、**零行列**である。

---

## 性質

### 定常性

ホワイトノイズカーネルは**定常カーネル**とみなせる。$k(\mathbf{x}, \mathbf{x}')$ が $\mathbf{x} - \mathbf{x}'$ に依存する形で、$\mathbf{x} = \mathbf{x}'$ のときのみ非ゼロ（差が 0 のときのみ値を持つ）と解釈できるためである。

### 正定値性

$\sigma_n^2 > 0$ のとき、自己共分散行列 $K = \sigma_n^2 I$ は**正定値**である。単位行列の正のスカラー倍なので、固有値はすべて $\sigma_n^2 > 0$ である。

### 意味づけ

- **観測ノイズ**：各観測 $y_i = f(\mathbf{x}_i) + \varepsilon_i$ において、$\varepsilon_i \sim \mathcal{N}(0, \sigma_n^2)$ で互いに独立とするとき、$\varepsilon$ の共分散がまさに $\sigma_n^2 \delta_{ij}$ となる。
- ガウス過程回帰では、潜在関数 $f$ の共分散（RBF など）に、このノイズ項を**加算**した共分散行列を使う。すると、対角に $\sigma_n^2$ が足され、数値的にも安定しつつ「観測の不確実性」を表現できる。

---

## ハイパーパラメータ

| パラメータ | 記号 | 意味 | 制約 |
|-----------|------|------|------|
| ノイズ分散（ノイズレベル） | $\sigma_n^2$ | 観測ノイズの分散 | $\sigma_n^2 > 0$ |

### 最適化時のパラメータ化

本実装では、$\sigma_n^2 > 0$ を満たすために**対数スケール**のハイパーパラメータ $\theta$ を導入している。

$$
\theta = \ln(\sigma_n^2) \quad \Leftrightarrow \quad \sigma_n^2 = \exp(\theta)
$$

- `get_hyperparameters()` は $\theta = \ln(\sigma_n^2)$ を返す。
- `set_hyperparameters(theta)` では $\sigma_n^2 = \exp(\theta)$ で内部のノイズレベルを更新する。

これにより、勾配ベースの最適化でも $\sigma_n^2$ が常に正に保たれる。

---

## 他のカーネルとの組み合わせ

ホワイトノイズカーネルは、**加算**で他のカーネルと組み合わせて使う。

### 例：RBF + ホワイトノイズ（標準的な GP 回帰）

$$
k(\mathbf{x}, \mathbf{x}') = k_{\mathrm{RBF}}(\mathbf{x}, \mathbf{x}') + \sigma_n^2 \, \delta(\mathbf{x}, \mathbf{x}')
$$

- RBF 部分：滑らかな潜在関数 $f$ の共分散
- ホワイトノイズ部分：観測 $y = f(\mathbf{x}) + \varepsilon$ の $\varepsilon$ の分散

カーネル行列では、

$$
K = K_{\mathrm{RBF}} + \sigma_n^2 I
$$

となり、対角にノイズ分散が足される。これがガウス過程回帰で最もよく使われる形である。

### 例：RBF + 定数 + ホワイトノイズ

$$
k(\mathbf{x}, \mathbf{x}') = k_{\mathrm{RBF}}(\mathbf{x}, \mathbf{x}') + \sigma_0^2 + \sigma_n^2 \, \delta(\mathbf{x}, \mathbf{x}')
$$

- 信号の共分散（RBF + 定数）と観測ノイズ（ホワイト）を分離して表現できる。

---

## まとめ

- **定義**: $k(\mathbf{x}, \mathbf{x}') = \sigma_n^2 \, \delta(\mathbf{x}, \mathbf{x}')$（同一入力でのみ $\sigma_n^2$、それ以外は 0）。
- **役割**: 観測ノイズの分散を表し、他のカーネルと足して GP 回帰の尤度・予測に用いる。
- **性質**: 定常、$\sigma_n^2 > 0$ で正定値。自己共分散は $\sigma_n^2 I$、クロス共分散は零行列。
- **実装**: ハイパーパラメータは対数スケール $\theta = \ln(\sigma_n^2)$ で保持・最適化する。
