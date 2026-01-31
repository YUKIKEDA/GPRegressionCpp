# 主成分分析 (Principal Component Analysis, PCA)

## 目次

1. [概要](#概要)
2. [理論的背景](#理論的背景)
3. [特異値分解による計算](#特異値分解による計算)
4. [説明分散とノイズ分散](#説明分散とノイズ分散)
5. [白化（Whiten）](#白化whiten)
6. [アルゴリズムの流れ](#アルゴリズムの流れ)
7. [実装との対応](#実装との対応)
8. [使用例](#使用例)
9. [利点と制限](#利点と制限)
10. [参考文献](#参考文献)

## 概要

**主成分分析（PCA, Principal Component Analysis）**は、多変量データの**分散が最大となる方向**（主成分）を順に求め、データをそれらの軸へ射影することで**次元削減**や**可視化**を行う手法である。相関のある多数の特徴量を、互いに無相関な少数の主成分で要約し、元のデータの変動の大部分を保持する。本実装では、データを**中心化**した上で **特異値分解（SVD）** を適用し、主成分軸・説明分散・変換結果を保持する。scikit-learn の `PCA(svd_solver='full')` と同様の挙動を目指している。

### 主な特徴

- **線形変換**: データに線形変換（射影）を施すだけなので解釈が容易
- **分散最大化**: 第1主成分はデータの分散を最大にする方向、第2主成分はそれに直交する中で分散を最大にする方向、と順に定義
- **無相関化**: 主成分得点同士は無相関
- **SVDベース**: 共分散行列の固有値分解と等価だが、本実装では数値的に安定な SVD を直接用いる
- **オプションで白化**: 変換後の各成分の分散を 1 にそろえる（白化）が可能

## 理論的背景

### 問題の定式化

$n$ 個のサンプル、$d$ 個の特徴量からなるデータ行列を

$$
X = (\mathbf{x}_1, \ldots, \mathbf{x}_n)^\top \in \mathbb{R}^{n \times d}
$$

とする。行 $\mathbf{x}_i^\top \in \mathbb{R}^d$ が $i$ 番目のサンプルである。

PCA の目的は、**射影後の分散が最大**となる単位ベクトル $\mathbf{w}_1 \in \mathbb{R}^d$（第1主成分）を見つけることである。データを**中心化**した

$$
\tilde{X} = X - \mathbf{1}_n \bar{\mathbf{x}}^\top, \quad \bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i
$$

に対して、射影は $\tilde{X} \mathbf{w}_1$ であり、その分散（サンプル数で正規化）は

$$
\frac{1}{n-1} \|\tilde{X} \mathbf{w}_1\|^2 = \frac{1}{n-1} \mathbf{w}_1^\top \tilde{X}^\top \tilde{X} \, \mathbf{w}_1 = \mathbf{w}_1^\top C \, \mathbf{w}_1
$$

となる。ここで

$$
C = \frac{1}{n-1} \tilde{X}^\top \tilde{X} \in \mathbb{R}^{d \times d}
$$

は**標本共分散行列**（不偏推定、自由度 $n-1$）である。

### 第1主成分

第1主成分は、$\|\mathbf{w}_1\| = 1$ の下で $\mathbf{w}_1^\top C \, \mathbf{w}_1$ を最大化する単位ベクトルである。これは $C$ の**最大固有値** $\lambda_1$ に対する**固有ベクトル**であり、最大分散は $\lambda_1$ である。

### 第2主成分以降

第2主成分 $\mathbf{w}_2$ は、$\|\mathbf{w}_2\| = 1$ かつ $\mathbf{w}_2 \perp \mathbf{w}_1$ の下で $\mathbf{w}_2^\top C \, \mathbf{w}_2$ を最大化する。これは $C$ の**2番目に大きい固有値** $\lambda_2$ に対する固有ベクトルである。同様に、第 $k$ 主成分は $k$ 番目に大きい固有値 $\lambda_k$ の固有ベクトルとなる。

### 主成分得点

主成分軸を列に並べた行列を

$$
W = (\mathbf{w}_1, \ldots, \mathbf{w}_k) \in \mathbb{R}^{d \times k}
$$

とすると、中心化データの主成分得点（変換後のデータ）は

$$
Z = \tilde{X} W \in \mathbb{R}^{n \times k}
$$

である。$Z$ の各列は互いに無相関であり、$j$ 列目の標本分散は $\lambda_j$（$C$ の $j$ 番目に大きい固有値）である。


## 特異値分解による計算

共分散行列 $C$ の固有値分解の代わりに、 **中心化データ行列 $\tilde{X}$ の特異値分解（SVD）** を直接用いる。これにより、$C$ を陽に作らずに主成分を求めることができ、数値的にも安定である。

### Thin SVD

$\tilde{X} \in \mathbb{R}^{n \times d}$ のランクを $r$（$r \leq \min(n, d)$）とする。Thin SVD は

$$
\tilde{X} = U \, S \, V^\top
$$

とする。ここで、

- $U \in \mathbb{R}^{n \times r}$：左特異ベクトル、$U^\top U = I_r$
- $V \in \mathbb{R}^{d \times r}$：右特異ベクトル、$V^\top V = I_r$
- $S = \mathrm{diag}(\sigma_1, \ldots, \sigma_r) \in \mathbb{R}^{r \times r}$：特異値、$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$

### 共分散行列との関係

$$
C = \frac{1}{n-1} \tilde{X}^\top \tilde{X} = \frac{1}{n-1} V S^\top U^\top U S V^\top = \frac{1}{n-1} V S^2 V^\top
$$

であるから、$C$ の固有値は $\lambda_j = \sigma_j^2 / (n-1)$、対応する固有ベクトルは $V$ の列である。したがって、

- **主成分軸**（単位ベクトル）：$V$ の列 $\mathbf{v}_1, \ldots, \mathbf{v}_r$ がそのまま主成分方向
- **説明分散**（各主成分が説明する分散）：$\lambda_j = \sigma_j^2 / (n-1)$
- **主成分得点**：$\tilde{X} V = U S V^\top V = U S$ より、得点は $U S$ の左から $k$ 列（第 $k$ 主成分まで）で与えられる

本実装では、主成分軸を **components_** として $V$ の左 $k$ 列の**転置**（行が主成分）で保持し、`transform` では $\tilde{X} \cdot \mathrm{components\_}^\top = \tilde{X} \cdot V_{1:k} = (U S)_{1:k}$ に相当する計算を行う。

### 符号の一意化（svd_flip）

SVD では $U, V$ の列の符号は一意でない。scikit-learn と同様に、各主成分（$V$ の列）について「その列で絶対値が最大の要素が正」になるように $U, V$ の符号を揃える。

## 説明分散とノイズ分散

### 説明分散（explained variance）

第 $j$ 主成分が説明する分散は

$$
\lambda_j = \frac{\sigma_j^2}{n-1}
$$

である。本実装では `explained_variance_` に $\lambda_1, \ldots, \lambda_k$ を格納する。

### 説明分散比（explained variance ratio）

全主成分（SVD で得られたすべての特異値）が説明する分散の合計は

$$
\lambda_{\mathrm{total}} = \sum_{j=1}^{r} \lambda_j = \frac{1}{n-1} \sum_{j=1}^{r} \sigma_j^2
$$

である。第 $j$ 主成分の**説明分散比**は

$$
\frac{\lambda_j}{\lambda_{\mathrm{total}}}
$$

であり、`explained_variance_ratio_` に格納する。合計は 1 以下で、$k < r$ のときは 1 未満（捨てた成分の分だけ説明されない）。

### ノイズ分散（noise variance）

$k < r$ で主成分を $k$ 個だけ使う場合、捨てた $r - k$ 個の成分の分散の**平均**をノイズ分散として推定する：

$$
\sigma_{\mathrm{noise}}^2 = \frac{1}{r - k} \sum_{j=k+1}^{r} \lambda_j
$$

これは、低ランク近似の残差の分散の推定などに用いられる。本実装では `noise_variance_` に格納する。

## 白化（Whiten）

**白化**を有効にすると、変換後の各主成分の**標本分散が 1** になるようにスケールする。

### 変換式

通常の主成分得点は $Z = \tilde{X} V_{1:k}$ で、$j$ 列目の分散は $\lambda_j$ である。白化では各列を $\sqrt{\lambda_j}$ で割る（標準偏差で正規化）ので、

$$
Z_{\mathrm{whiten}}^{(j)} = \frac{Z^{(j)}}{\sqrt{\lambda_j}} = \frac{\tilde{X} \mathbf{v}_j}{\sigma_j / \sqrt{n-1}} = \sqrt{n-1} \, \frac{\tilde{X} \mathbf{v}_j}{\sigma_j}
$$

とする。実装では、得点を求めた後に各 $j$ について `proj.col(j) *= sqrt(n_samples - 1) / singular_values(j)` としている。これで $Z_{\mathrm{whiten}}$ の各列の標本分散が 1 になる。

白化は、例えば独立成分分析（ICA）の前処理や、特徴量のスケールを揃えたい場合に用いられる。

---

## アルゴリズムの流れ

### 学習（fit）

1. **入力チェック**: $X$ が空でないこと、`n_components` が 1 以上 $\min(n, d)$ 以下（または -1 で全成分）であることを確認する。
2. **中心化**: 列ごとの平均 $\bar{\mathbf{x}}$ を計算し、$\tilde{X} = X - \mathbf{1}_n \bar{\mathbf{x}}^\top$ を形成する。
3. **Thin SVD**: $\tilde{X} = U S V^\top$ を計算する（Eigen の `JacobiSVD`、`ComputeThinU | ComputeThinV`）。
4. **符号の一意化**: `svd_flip_(U, V, k)` で各主成分の符号を揃える。
5. **主成分の保持**: `components_` に $V$ の左 $k$ 列の転置、`singular_values_` に $\sigma_1, \ldots, \sigma_k$ を格納する。
6. **説明分散**: $\lambda_j = \sigma_j^2 / (n-1)$ を `explained_variance_` に格納する。
7. **説明分散比**: $\lambda_j / \lambda_{\mathrm{total}}$ を `explained_variance_ratio_` に格納する。
8. **ノイズ分散**: $k < r$ のとき、捨てた成分の分散の平均を `noise_variance_` に格納する。

### 変換（transform）

1. **事前条件**: `fit` が呼ばれていること、入力 $X$ の列数が学習時の特徴量数と一致することを確認する。
2. **中心化**: 学習時の平均 `mean_` で中心化し、$\tilde{X}$ を得る。
3. **射影**: $\tilde{X} \cdot \mathrm{components\_}^\top$ で主成分得点を計算する。
4. **白化**（`whiten_ == true` のとき）: 各列 $j$ に $\sqrt{n-1} / \sigma_j$ をかける。
5. 変換後の行列を返す。

### 疑似コード

```
function PCA.fit(X):
    n, d = X.rows, X.cols
    k = (n_components == -1) ? min(n, d) : n_components
    mean = colwise_mean(X)
    X_centered = X - mean
    U, S, V = thin_svd(X_centered)
    svd_flip(U, V, k)
    components = V[:, 1:k].T
    singular_values = S[1:k]
    ddof = n - 1
    explained_variance = singular_values.^2 / ddof
    total_var = sum(S.^2) / ddof
    explained_variance_ratio = explained_variance / total_var
    if k < rank(X_centered):
        noise_variance = mean( (S[k+1:end].^2) / ddof )
    else:
        noise_variance = 0
    return self

function PCA.transform(X):
    X_centered = X - mean
    proj = X_centered * components.T
    if whiten:
        for j in 1:k:
            proj[:, j] *= sqrt(n_samples - 1) / singular_values[j]
    return proj
```

---

## 実装との対応

| 項目 | 数学 | 実装 |
|------|------|------|
| データ行列 | $X \in \mathbb{R}^{n \times d}$ | `fit(X)` の引数、行がサンプル |
| 中心化 | $\tilde{X} = X - \mathbf{1}_n \bar{\mathbf{x}}^\top$ | `mean_ = X.colwise().mean()`, `X_centered = X.rowwise() - mean_` |
| 主成分数 | $k$ | `n_components_`（-1 のときは $\min(n, d)$） |
| 主成分軸 | $V$ の左 $k$ 列（行ベクトルとして） | `components_`（$k \times d$） |
| 特異値 | $\sigma_1, \ldots, \sigma_k$ | `singular_values_` |
| 説明分散 | $\lambda_j = \sigma_j^2/(n-1)$ | `explained_variance_` |
| 説明分散比 | $\lambda_j / \lambda_{\mathrm{total}}$ | `explained_variance_ratio_` |
| ノイズ分散 | 捨てた成分の分散の平均 | `noise_variance_` |
| 変換 | $\tilde{X} \cdot \mathrm{components\_}^\top$ | `X_centered * components_.transpose()` |
| 白化 | 各列を $\sqrt{n-1}/\sigma_j$ でスケール | `whiten_ == true` のとき `proj.col(j) *= scale / singular_values_(j)` |
| SVD | Thin SVD | `Eigen::JacobiSVD`（`ComputeThinU \| ComputeThinV`） |
| 符号の一意化 | 各 $V$ の列で最大絶対値要素が正 | `svd_flip_(U, V, k)` |

---

## 使用例

### 基本的なフィットと変換

```cpp
#include "decomposition/PCA/pca.hpp"

gprcpp::decomposition::PCA pca(3);  // 主成分を 3 個に削減
// または
gprcpp::decomposition::PCA pca(-1);  // 全主成分を保持

Eigen::MatrixXd X(n_samples, n_features);
pca.fit(X);

// 変換（次元削減）
Eigen::MatrixXd Z = pca.transform(X);  // (n_samples, n_components)

// 学習と変換を一括
Eigen::MatrixXd Z2 = pca.fit_transform(X);
```

### 白化付き

```cpp
gprcpp::decomposition::PCA pca(5, true);  // n_components=5, whiten=true
pca.fit(X);
Eigen::MatrixXd Z_whiten = pca.transform(X);  // 各列の分散が 1
```

### 説明分散・分散比の取得

```cpp
Eigen::VectorXd ev = pca.explained_variance();
Eigen::VectorXd evr = pca.explained_variance_ratio();
double noise_var = pca.noise_variance();

// 累積説明分散比で「何次元まで使うか」を決める例
double cum = 0;
int k_use = 0;
for (int j = 0; j < pca.n_components(); ++j) {
    cum += evr(j);
    if (cum >= 0.95) { k_use = j + 1; break; }
}
```

### 主成分軸と平均

```cpp
const Eigen::MatrixXd& comp = pca.components();   // (n_components, n_features)
const Eigen::VectorXd& mean = pca.mean();       // (n_features,)
const Eigen::VectorXd& sing = pca.singular_values();
```

---

## 利点と制限

### 利点

1. **解釈が容易**: 主成分は「分散が大きい方向」として明確な意味を持つ。
2. **無相関化**: 主成分得点同士は無相関で、その後の解析がしやすい。
3. **次元削減**: 説明分散比の小さい成分を捨てることで、情報をあまり失わずに次元を減らせる。
4. **SVD による安定性**: 共分散行列を陽に作らず SVD で計算するため、数値的に安定である。
5. **実装が単純**: 中心化と SVD のみで一貫したパイプライン。

### 制限

1. **線形のみ**: 非線形な構造には対応できない（カーネル PCA などは別手法）。
2. **スケールに依存**: 特徴量の単位・スケールが違うと、分散の大きい変数が主成分を支配しやすい。必要に応じて事前に標準化（各特徴量を標準偏差で割る）を行う。
3. **外れ値に敏感**: 分散を最大化するため、外れ値の影響を受けやすい。
4. **クラスラベルを無視**: 教師なしなので、分類性能を直接は考慮しない。

### 適用が適切な場面

- 高次元データの可視化（2〜3 主成分への射影）
- 回帰・分類の前処理としての次元削減
- 相関の強い特徴量の圧縮
- ノイズ低減（小さい特異値に対応する成分を捨てる）

---

## 参考文献

### 主要な文献

1. **Jolliffe, I. T. (2002)**
   - *Principal Component Analysis*, 2nd ed.
   - Springer Series in Statistics.
   - PCA の標準的な教科書。

2. **Pearson, K. (1901)**
   - "On Lines and Planes of Closest Fit to Systems of Points in Space"
   - *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 2(11), 559–572.
   - PCA の起源とされる論文。

3. **Hotelling, H. (1933)**
   - "Analysis of a complex of statistical variables into principal components"
   - *Journal of Educational Psychology*, 24(6), 417–441, 498–520.
   - 主成分の現代的な定式化。

### 実装・数値計算

4. **Golub, G. H., & Van Loan, C. F. (2013)**
   - *Matrix Computations*, 4th ed.
   - Johns Hopkins University Press.
   - SVD と固有値分解の数値計算。

5. **scikit-learn: sklearn.decomposition.PCA**
   - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
   - 本実装が挙動を合わせている API（`svd_solver='full'`）。

### 関連手法

- **Kernel PCA**: カーネル trick による非線形主成分分析
- **Incremental PCA**: データを分割して逐次更新する PCA
- **Sparse PCA**: 主成分をスパースにし解釈しやすくする
- **Factor Analysis**: 潜在変数モデルに基づく因子分析

---

**関連ドキュメント**

- [include/README.md](../../README.md): ガウス過程回帰をはじめとする本プロジェクトの概要

---

**最終更新**: 2026年1月31日
