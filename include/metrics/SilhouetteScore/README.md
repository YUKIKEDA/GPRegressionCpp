# シルエットスコア（Silhouette Score）

## 目次

1. [概要](#概要)
2. [理論的背景](#理論的背景)
3. [シルエット係数の計算](#シルエット係数の計算)
4. [距離の扱い](#距離の扱い)
5. [アルゴリズムの流れ](#アルゴリズムの流れ)
6. [実装との対応](#実装との対応)
7. [使用例](#使用例)
8. [利点と制限](#利点と制限)
9. [参考文献](#参考文献)

---

## 概要

**シルエットスコア（Silhouette Score）**は、クラスタリング結果の**品質を評価する**教師なし指標である。Rousseeuw (1987) により提案され、各サンプルについて「自分のクラスタ内での凝集度」と「最も近い他クラスタとの分離度」を比較し、**-1 から 1** の範囲の**シルエット係数**を算出する。全サンプルの係数の平均がシルエットスコアとなり、クラスタ数 $K$ の選択やクラスタリング手法の比較に用いられる。本実装では、特徴行列 $X$ からユークリッド距離で計算する方式と、**事前計算された距離行列** $D$ を用いる方式の両方を提供する。scikit-learn の `silhouette_score` / `silhouette_samples` と同様の定義に従う。

### 主な特徴

- **クラスタ品質の評価**: 各サンプルが「自分のクラスタにどれだけよく属しているか」を -1～1 で表現
- **凝集度と分離度**: クラスタ内平均距離 $a$ と最近傍他クラスタ平均距離 $b$ の比で定義
- **クラスタ数選択**: シルエットスコアが高い $K$ を選ぶことで、適切なクラスタ数の目安を得られる
- **距離行列オプション**: 特徴行列 $X$（ユークリッド）または事前計算距離行列 $D$ のどちらでも計算可能
- **サンプル単位と全体**: 各サンプルの係数（`silhouette_samples`）とその平均（`silhouette_score`）の両方を取得可能

---

## 理論的背景

### 問題の設定

$n$ 個のサンプル $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^d$ が、クラスタラベル $z_1, \ldots, z_n \in \{0, 1, \ldots, K-1\}$ により $K$ 個のクラスタに分割されているとする。シルエット係数は、**各サンプル $i$ について**「自分のクラスタへの所属の妥当性」を一つのスカラーで表す。

### 記号

- $C_k$: ラベル $k$ を持つサンプルの集合。$|C_k|$ はそのサイズ。
- $d(i, j)$: サンプル $i$ と $j$ の間の距離（本実装のデフォルトはユークリッド距離 $\|\mathbf{x}_i - \mathbf{x}_j\|$）。
- $a(i)$: サンプル $i$ の**クラスタ内平均距離**。自分以外の同一クラスタ内の点との距離の平均。
- $b(i)$: サンプル $i$ の**最近傍他クラスタ平均距離**。$i$ が属さないクラスタのうち、$i$ からそのクラスタ内の点への距離の平均が最小となるクラスタでの、その平均値。

### シルエット係数の定義

サンプル $i$ の**シルエット係数** $s(i)$ は次で定義される：

$$
s(i) = \frac{b(i) - a(i)}{\max\bigl(a(i), b(i)\bigr)}
$$

- **$a(i)$（凝集度）**: 小さいほど、$i$ は自分のクラスタ内の点に「近い」＝クラスタにうまく属している。
- **$b(i)$（分離度）**: 大きいほど、$i$ は他クラスタから「遠い」＝クラスタが分離している。
- 分子 $b(i) - a(i)$: 分離度が凝集度より大きいほど正になり、クラスタ構造が良いことを示す。
- 分母 $\max(a(i), b(i))$: スコアを **-1 から 1** に正規化する。

### 係数の解釈

| $s(i)$ の範囲 | 解釈 |
|---------------|------|
| $s(i) \approx 1$ | クラスタ内に近く、他クラスタから遠い。所属が適切。 |
| $s(i) \approx 0$ | クラスタの境界付近。どちらのクラスタとも曖昧。 |
| $s(i) \approx -1$ | 他クラスタの方が近い。誤ったクラスタに割り当てられている可能性。 |

### シルエットスコア（全体）

**シルエットスコア**は、全サンプルのシルエット係数の**平均**である：

$$
\bar{s} = \frac{1}{n} \sum_{i=1}^{n} s(i)
$$

クラスタリング全体の品質を一つのスカラーで表し、$K$ を変えて $\bar{s}$ を比較することで、適切なクラスタ数の選択に利用できる。

### 有効なラベル範囲

シルエット係数は「同一クラスタ内」と「他クラスタ」の両方が必要である。したがって、

- **クラスタ数** $K \geq 2$
- **全サンプルが同一クラスタに属さない**こと（各クラスタに少なくとも 1 点あるとして、実質 $K \leq n - 1$）

が必要である。本実装では $K \in [2, n-1]$ を要求する。

---

## シルエット係数の計算

### クラスタ内平均距離 $a(i)$

サンプル $i$ が属するクラスタを $C_{\ell}$（$\ell = z_i$）とする。$a(i)$ は、**自分を除いた**同一クラスタ内の点との距離の平均である：

$$
a(i) = \frac{1}{|C_{\ell}| - 1} \sum_{j \in C_{\ell},\, j \neq i} d(i, j)
$$

- **シングルトン**（$|C_{\ell}| = 1$）のときは、同一クラスタ内に他に点がないため、定義に従い **$a(i) = 0$** とする。
- 実装では、クラスタ $\ell$ 内の全点への距離の和を累積し、分母を $|C_{\ell}| - 1$（1 のときは 0 扱い）で割る。

### 最近傍他クラスタ平均距離 $b(i)$

$i$ が属さないクラスタ $C_k$（$k \neq \ell$）について、$i$ から $C_k$ 内の全点への距離の**平均**を

$$
\bar{d}(i, C_k) = \frac{1}{|C_k|} \sum_{j \in C_k} d(i, j)
$$

とする。**$b(i)$** は、この平均が最小となる他クラスタでの値である：

$$
b(i) = \min_{k \neq \ell} \bar{d}(i, C_k)
$$

他クラスタが一つも存在しない場合は通常起こらないが、その場合は $b(i) = 0$ とする（実装上のフォールバック）。

### 係数 $s(i)$ の式

$$
s(i) = \frac{b(i) - a(i)}{\max\bigl(a(i), b(i)\bigr)}
$$

- $\max(a(i), b(i)) = 0$ のとき（シングルトンで $a(i)=0$ かつ他クラスタなしなど）、分母を 0 にしないため **$s(i) = 0$** とする。
- 非有限値になった場合も $s(i) = 0$ とする（実装上の安定化）。

---

## 距離の扱い

### 特徴行列からユークリッド距離で計算

入力がデータ行列 $X \in \mathbb{R}^{n \times d}$ の場合、サンプル間の距離は**ユークリッド距離**で計算する：

$$
d(i, j) = \|\mathbf{x}_i - \mathbf{x}_j\| = \sqrt{\sum_{t=1}^{d} (x_{it} - x_{jt})^2}
$$

実装では、二乗距離 $\|\mathbf{x}_i - \mathbf{x}_j\|^2$ を効率的に求めるため、展開

$$
\|\mathbf{x}_i - \mathbf{x}_j\|^2 = \|\mathbf{x}_i\|^2 + \|\mathbf{x}_j\|^2 - 2 \mathbf{x}_i^\top \mathbf{x}_j
$$

を用い、行列演算で一括計算したのち、非負にクリップして平方根を取り、$d(i,j)$ を得る。

### 事前計算された距離行列

既に距離行列 $D \in \mathbb{R}^{n \times n}$（$D_{ij} = d(i,j)$、対角は 0）が得られている場合は、**事前計算版** `silhouette_samples_precomputed` / `silhouette_score_precomputed` に $D$ とラベルを渡す。この場合、距離の定義（ユークリッド、マンハッタン、カスタム距離など）は呼び出し側で自由に決められる。

---

## アルゴリズムの流れ

### 特徴行列 $X$ から計算する場合

1. **入力チェック**: $X$ が空でないこと、`labels` の長さが $n$ であることを確認する。
2. **ラベルの正規化**: ラベルを 0 から $K-1$ の連続した整数にエンコードし、各クラスタのサンプル数 `label_freqs` を求める。
3. **ラベル範囲チェック**: $K \in [2, n-1]$ であることを確認する。
4. **距離行列の計算**: $X$ からペアワイズユークリッド距離行列 $D$ を計算する。
5. **クラスタ別距離の累積**: 各サンプル $i$ について、クラスタ $k$ 内の全点への距離の和を `cluster_distances(i, k)` に格納する。
6. **各サンプルのシルエット係数**:
   - $a_i$: `cluster_distances(i, li) / (freq_li - 1)`（$|C_{\ell}| \leq 1$ のときは 0）。
   - $b_i$: $i$ が属さない $k$ について `cluster_distances(i, k) / freq_k` の最小値。
   - $s_i = (b_i - a_i) / \max(a_i, b_i)$。分母 0 や非有限のときは 0。
7. **サンプル単位**: ベクトル $(s_1, \ldots, s_n)$ を返す（`silhouette_samples`）。
8. **スコア**: その平均を返す（`silhouette_score`）。

### 事前計算距離行列 $D$ から計算する場合

1. **入力チェック**: $D$ が $n \times n$ であること、`labels` の長さが $n$ であることを確認する。
2. 上記の 2～6 と同様に、$D$ をそのまま用いて $a_i, b_i, s_i$ を計算する。
3. `silhouette_samples_precomputed` は $(s_1, \ldots, s_n)$、`silhouette_score_precomputed` はその平均を返す。

### 疑似コード

```
function silhouette_samples(X, labels):
    n = X.rows
    encoded, n_clusters, label_freqs = encode_labels(labels)
    check_number_of_labels(n_clusters, n)

    D = pairwise_euclidean_distances(X)   // (n, n)

    cluster_distances = zeros(n, n_clusters)
    for i = 1 to n:
        for j = 1 to n:
            k = encoded(j)
            cluster_distances(i, k) += D(i, j)

    sil_samples = empty(n)
    for i = 1 to n:
        li = encoded(i)
        freq_li = label_freqs[li]
        a_i = (freq_li <= 1) ? 0 : cluster_distances(i, li) / (freq_li - 1)
        b_i = min over k != li of (cluster_distances(i, k) / label_freqs[k])
        denom = max(a_i, b_i)
        sil_samples(i) = (denom <= 0 or !finite(denom)) ? 0 : (b_i - a_i) / denom

    return sil_samples

function silhouette_score(X, labels):
    return mean(silhouette_samples(X, labels))
```

---

## 実装との対応

| 項目 | 数学 | 実装 |
|------|------|------|
| サンプル数 | $n$ | `X.rows()` または `D.rows()` |
| クラスタ数 | $K$ | `n_clusters`（`encode_labels` 内で算出） |
| ラベル | $z_1, \ldots, z_n$ | `labels` → エンコード後 `encoded` |
| クラスタ内平均距離 | $a(i)$ | `intra_sum / (freq_li - 1)`（freq_li ≤ 1 のとき 0） |
| 最近傍他クラスタ平均距離 | $b(i)$ | 他クラスタに対する `cluster_distances(i,k)/fk` の最小値 |
| シルエット係数 | $s(i) = (b(i)-a(i))/\max(a(i),b(i))$ | `sil_samples(i)`（分母 0 等のとき 0） |
| シルエットスコア | $\bar{s} = \frac{1}{n}\sum_i s(i)$ | `samples.mean()` |
| 距離行列（特徴から） | $d(i,j) = \|\mathbf{x}_i - \mathbf{x}_j\|$ | `pairwise_euclidean_distances(X)` |
| 距離行列（事前計算） | $D_{ij} = d(i,j)$ | `silhouette_*_precomputed(D, labels)` |
| ラベル範囲 | $K \in [2, n-1]$ | `check_number_of_labels` |

### 二乗ユークリッド距離の計算

特徴行列 $X$ から距離を求める際、実装では二乗ノルムの展開

$$
\|\mathbf{x}_i - \mathbf{x}_j\|^2 = \|\mathbf{x}_i\|^2 + \|\mathbf{x}_j\|^2 - 2 \mathbf{x}_i^\top \mathbf{x}_j
$$

を用い、行列として $D^{(2)} = \mathbf{a}\mathbf{1}^\top + \mathbf{1}\mathbf{b}^\top - 2 X X^\top$（$\mathbf{a}_i = \|\mathbf{x}_i\|^2$, $\mathbf{b}_j = \|\mathbf{x}_j\|^2$）を計算し、非負化してから平方根を取り、ユークリッド距離行列にしている。

---

## 使用例

### 特徴行列からシルエットスコア・サンプル係数

```cpp
#include "metrics/SilhouetteScore/silhouette_score.hpp"

Eigen::MatrixXd X(n_samples, n_features);
Eigen::VectorXi labels(n_samples);  // クラスタリング結果（0, 1, ..., K-1 など）

// 全サンプルの平均シルエットスコア
double score = gprcpp::metrics::silhouette_score(X, labels);

// 各サンプルのシルエット係数
Eigen::VectorXd per_sample = gprcpp::metrics::silhouette_samples(X, labels);
```

### 事前計算された距離行列から

```cpp
Eigen::MatrixXd D(n_samples, n_samples);  // ペアワイズ距離行列、対角 0

double score_pre = gprcpp::metrics::silhouette_score_precomputed(D, labels);
Eigen::VectorXd per_sample_pre = gprcpp::metrics::silhouette_samples_precomputed(D, labels);
```

### クラスタ数 K の選択（エルボー的な利用）

```cpp
std::vector<double> scores;
for (int K = 2; K <= K_max; ++K) {
    gprcpp::clustering::KMeans kmeans(K, 300, 1e-4, 42);
    kmeans.fit(X);
    double s = gprcpp::metrics::silhouette_score(X, kmeans.labels());
    scores.push_back(s);
}
// スコアが最大となる K を候補とする
```

---

## 利点と制限

### 利点

1. **直感的な解釈**: -1～1 の範囲で「よく所属している／境界／誤割り当て」が分かりやすい。
2. **クラスタ数選択**: $K$ を変えてスコアを比較し、適切なクラスタ数の目安を得られる。
3. **手法非依存**: K-means、階層クラスタリングなど、ラベルさえあればどの手法の結果にも適用できる。
4. **サンプル単位の診断**: 各点の係数を見ることで、境界付近や外れのサンプルを把握できる。
5. **事前計算距離**: カスタム距離を使う場合は距離行列を渡すだけでよい。

### 制限

1. **計算コスト**: 距離行列が $O(n^2)$ のメモリ・計算となるため、大規模データでは負荷が大きい。
2. **凸・球状のクラスタに有利**: ユークリッド距離を用いる場合、凸でコンパクトなクラスタで高スコアが出やすく、細長い・非凸のクラスタでは低くなりがち。
3. **クラスタ数 2 以上**: 1 クラスタや「全員同一ラベル」の場合は定義できない（本実装では例外）。
4. **スケール依存**: 特徴のスケールが距離に影響するため、必要に応じて標準化を行うとよい。

### 適用が適切な場面

- K-means などでクラスタ数 $K$ を決める補助（シルエットスコアの比較）。
- クラスタリング結果の品質の定量的な比較。
- 各サンプルがどの程度「自分のクラスタにうまく入っているか」の可視化・診断。

### 適用が不適切な場面

- サンプル数が非常に大きく、$O(n^2)$ の距離行列が扱えない場合（サンプリングや近似の検討）。
- クラスタ構造が非凸・複雑な形状で、ユークリッド距離での評価が適さない場合（距離を変えるか、別指標の併用を検討）。

---

## 参考文献

### 主要な文献

1. **Rousseeuw, P. J. (1987)**
   - "Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis"
   - *Journal of Computational and Applied Mathematics*, 20, 53–65.
   - シルエット係数・スコアの原論文。定義と解釈、クラスタ検証への利用。

### 実装・API

2. **scikit-learn: sklearn.metrics.silhouette_score, silhouette_samples**
   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
   - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html
   - 本実装が定義を合わせている API（特徴行列版・事前計算距離版）。

### 関連手法

- **K-means**: クラスタ数 $K$ を決めたうえでクラスタリング；シルエットスコアで $K$ を評価可能（[K-means README](../../clustering/k-means/README.md)）。
- **Davies–Bouldin index**: クラスタ内分散とクラスタ間距離に基づく別のクラスタ評価指標。
- **Calinski–Harabasz index**: 分散比に基づくクラスタ評価指標。

---

**関連ドキュメント**

- [include/README.md](../../README.md): ガウス過程回帰をはじめとする本プロジェクトの概要
- [K-means クラスタリング](../../clustering/k-means/README.md): シルエットスコアでクラスタ数や結果を評価する典型的な対象

---

**最終更新**: 2026年1月31日
