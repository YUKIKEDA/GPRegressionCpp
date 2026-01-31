# K-meansクラスタリング

## 目次

1. [概要](#概要)
2. [理論的背景](#理論的背景)
3. [k-means++初期化](#k-means初期化)
4. [Lloydアルゴリズム](#lloydアルゴリズム)
5. [収束判定](#収束判定)
6. [実装との対応](#実装との対応)
7. [使用例](#使用例)
8. [利点と制限](#利点と制限)
9. [参考文献](#参考文献)

---

## 概要

**K-meansクラスタリング**は、$n$ 個のサンプルを $K$ 個のクラスタに分割する**ベクトル量子化**手法である。各サンプルを最も近い**セントロイド（クラスタ中心）**に割り当て、セントロイドをクラスタ内サンプルの平均位置に更新することを反復する。この反復は**Lloydアルゴリズム**として知られ、クラスタ内平方和（**WCSS, Within-Cluster Sum of Squares**）を局所的に最小化する。本実装では、セントロイドの初期化に **k-means++** を採用し、収束速度と解の品質を向上させている。

### 主な特徴

- **非階層的クラスタリング**: 事前に指定した $K$ 個のクラスタに分割
- **高速**: 各反復で $O(nKd)$（$n$: サンプル数、$K$: クラスタ数、$d$: 次元数）
- **k-means++初期化**: 期待値の意味で $O(\log K)$ 競争比を保証する初期化
- **Lloyd反復**: E-step（割り当て）とM-step（更新）の交互反復
- **慣性（Inertia）**: クラスタ内平方和を品質指標として出力

---

## 理論的背景

### 問題の定式化

$n$ 個のサンプル $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^d$ を、$K$ 個の互いに素な部分集合（クラスタ）$C_1, \ldots, C_K$ に分割する。各クラスタ $C_k$ のセントロイドを

$$
\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{\mathbf{x} \in C_k} \mathbf{x}
$$

とする。K-meansの目的は、 **クラスタ内平方和（WCSS, Within-Cluster Sum of Squares）** を最小化する分割を求めることである：

$$
\min_{C_1, \ldots, C_K} \sum_{k=1}^{K} \sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2
$$

この目的関数は **慣性（inertia）** とも呼ばれる。

### NP困難性

WCSS を厳密に最小化する問題は**NP困難**である。したがって、実用的には**局所最適解**を求めるヒューリスティックが用いられる。Lloydアルゴリズムはその代表例であり、初期値から出発して目的関数を単調に減少させ、有限回の反復で局所最適解に収束する。

### クラスタ割り当てとセントロイド

サンプル $\mathbf{x}_i$ が属するクラスタを $z_i \in \{1, \ldots, K\}$ とすると、WCSS は次のように書ける：

$$
J(\{z_i\}, \{\boldsymbol{\mu}_k\}) = \sum_{i=1}^{n} \|\mathbf{x}_i - \boldsymbol{\mu}_{z_i}\|^2
$$

この目的関数を最小化するため、以下の2つの操作を交互に行う：

1. **割り当て（E-step）**: 固定した $\boldsymbol{\mu}_k$ に対し、各 $\mathbf{x}_i$ を最も近いセントロイドに割り当てる
2. **更新（M-step）**: 固定した $z_i$ に対し、各クラスタのセントロイドを再計算する

---

## k-means++初期化

### 動機

K-meansの収束解は初期セントロイドに大きく依存する。ランダム初期化では悪い局所最適解に収束する可能性があり、複数回実行して最良解を選ぶ必要がある。**k-means++**（Arthur & Vassilvitskii, 2007）は、セントロイドを分散させて配置する確率的初期化法であり、期待値の意味で最適解の $O(\log K)$ 倍以内の WCSS を保証する。

### アルゴリズム

1. 最初のセントロイド $\boldsymbol{\mu}_1$ を、データ点からランダムに一様に選ぶ
2. $c = 2, \ldots, K$ について以下を繰り返す：
   - 各サンプル $\mathbf{x}_i$ について、すでに選ばれたセントロイドまでの最短二乗距離を計算する：
     $$
     D(\mathbf{x}_i)^2 = \min_{j < c} \|\mathbf{x}_i - \boldsymbol{\mu}_j\|^2
     $$
   - $\mathbf{x}_i$ を確率 $\frac{D(\mathbf{x}_i)^2}{\sum_{j=1}^{n} D(\mathbf{x}_j)^2}$ で選び、新しいセントロイド $\boldsymbol{\mu}_c$ とする
3. すべてのセントロイドが選ばれたら、Lloydアルゴリズムに渡す

### 確率的サンプリングの意味

$D(\mathbf{x})^2$ に比例する確率でサンプリングすることで、すでに選ばれたセントロイドから**遠い**点ほど高い確率で選ばれる。これにより、セントロイドがデータ空間に分散して配置され、悪い局所最適解に陥りにくくなる。

### 疑似コード

```
function kmeanspp_init(X, K, rng):
    n = X.rows
    centers = empty (K, d)

    // 1. 最初の中心をランダムに選択
    idx = uniform_int(0, n-1, rng)
    centers[0] = X[idx]

    // 2. 残りの中心を順次選択
    closest_dist_sq = squared_distances(X, centers[0])  // (n,)

    for c = 1 to K-1:
        total = sum(closest_dist_sq)
        if total <= 0:
            break

        // 累積和で重み付きサンプリング
        cumsum = cumulative_sum(closest_dist_sq)
        r = uniform(0, 1, rng) * total
        idx = find_first_index_where(cumsum >= r)

        centers[c] = X[idx]

        // 最近傍距離を更新
        dist_to_new = squared_distances(X, centers[c])
        closest_dist_sq = element_wise_min(closest_dist_sq, dist_to_new)

    return centers
```

---

## Lloydアルゴリズム

### 概要

Lloydアルゴリズムは、K-meansの最も基本的な反復法である。 **E-step（Expectation）** でサンプルをセントロイドに割り当て、 **M-step（Maximization）** でセントロイドを更新する。この名前は、信号処理の文脈で Stuart Lloyd（1957, 1982）が提案したことに由来する。

### アルゴリズムの流れ

#### ステップ1: 初期化

セントロイド $\boldsymbol{\mu}_1^{(0)}, \ldots, \boldsymbol{\mu}_K^{(0)}$ を k-means++ または指定された初期値で設定する。

#### ステップ2: 反復

各反復 $t = 0, 1, 2, \ldots$ で以下を実行：

##### 2.1 E-step（割り当て）

各サンプル $\mathbf{x}_i$ を、最も近いセントロイドのクラスタに割り当てる：

$$
z_i^{(t+1)} = \arg\min_{k \in \{1, \ldots, K\}} \|\mathbf{x}_i - \boldsymbol{\mu}_k^{(t)}\|^2
$$

同点の場合は任意のルール（例: 最小インデックス）で決定する。

##### 2.2 M-step（更新）

各クラスタ $k$ のセントロイドを、そのクラスタに属するサンプルの平均で更新する：

$$
\boldsymbol{\mu}_k^{(t+1)} = \frac{1}{|C_k^{(t+1)}|} \sum_{i : z_i^{(t+1)} = k} \mathbf{x}_i
$$

**空クラスタの処理**: $|C_k| = 0$ となった場合、セントロイドが未定義になる。本実装では、ランダムなサンプルで置き換える。

#### ステップ3: 収束判定

以下のいずれかを満たしたら終了：

1. ラベル $z_i$ が変化しない
2. セントロイドの変化 $\sum_k \|\boldsymbol{\mu}_k^{(t+1)} - \boldsymbol{\mu}_k^{(t)}\|^2$ が許容誤差 $\tau$ 以下
3. 最大反復回数に達した

### 収束性

**定理**: Lloydアルゴリズムは有限回の反復で収束する。

**証明の概略**:
1. E-stepでは、各サンプルを最も近いセントロイドに割り当てるため、WCSS は減少または不変
2. M-stepでは、クラスタ内の平均がそのクラスタの WCSS を最小化するため、WCSS は減少または不変
3. WCSS は下に有界（$\geq 0$）
4. 可能なラベル割り当ては有限個（$K^n$ 通り）

したがって、アルゴリズムは単調減少列を生成し、有限回で停止する。ただし、収束点は**大域的最適解とは限らない**。

### 疑似コード

```
function lloyd(X, centers_init, max_iter, tol):
    centers = centers_init
    labels = assign(X, centers)
    
    for iter = 1 to max_iter:
        // M-step: 中心の更新
        centers_new = zeros(K, d)
        counts = zeros(K)
        for i = 1 to n:
            k = labels[i]
            centers_new[k] += X[i]
            counts[k] += 1
        for k = 1 to K:
            if counts[k] > 0:
                centers_new[k] /= counts[k]
            else:
                centers_new[k] = X[random_index]  // 空クラスタ処理
        
        // E-step: 割り当て
        labels_new = assign(X, centers_new)
        
        // 収束判定
        if labels_new == labels:
            return (centers_new, labels_new, inertia(X, centers_new, labels_new), iter)
        
        center_shift = sum_squared(centers_new - centers)
        if center_shift <= tol:
            return (centers_new, labels_new, inertia(X, centers_new, labels_new), iter)
        
        centers = centers_new
        labels = labels_new
    
    return (centers, labels, inertia(X, centers, labels), max_iter)

function assign(X, centers):
    // 各サンプルを最近傍のセントロイドに割り当て
    labels = empty(n)
    for i = 1 to n:
        labels[i] = argmin_k ||X[i] - centers[k]||^2
    return labels

function inertia(X, centers, labels):
    sum = 0
    for i = 1 to n:
        sum += ||X[i] - centers[labels[i]]||^2
    return sum
```

---

## 収束判定

### ラベルベースの収束

前回の反復から割り当てが変化しなければ、アルゴリズムは収束したと見なす：

$$
z_i^{(t+1)} = z_i^{(t)} \quad \forall i
$$

これは最も厳密な収束条件であり、ラベルが安定したことを保証する。

### セントロイド変化ベースの収束

セントロイドの変化量が許容誤差以下であれば収束と見なす：

$$
\sum_{k=1}^{K} \|\boldsymbol{\mu}_k^{(t+1)} - \boldsymbol{\mu}_k^{(t)}\|^2 \leq \tau
$$

本実装では、許容誤差 $\tau$ をデータの分散に基づいて正規化する：

$$
\tau = \text{tol} \times \frac{1}{d} \sum_{j=1}^{d} \text{Var}(X_{\cdot j})
$$

ここで $X_{\cdot j}$ は $j$ 列目の値である。これにより、データのスケールに依存しない収束判定が可能になる。

### 最大反復回数

上記の条件が満たされなくても、最大反復回数に達したら終了する。これは無限ループを防ぐ安全策である。

---

## 実装との対応

| 項目 | 数学 | 実装 |
|------|------|------|
| クラスタ数 | $K$ | `n_clusters_` |
| サンプル数 | $n$ | `n_samples_`（`X.rows()`） |
| 特徴量次元 | $d$ | `n_features_`（`X.cols()`） |
| セントロイド | $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K$ | `cluster_centers_`（$K \times d$） |
| クラスタラベル | $z_1, \ldots, z_n$ | `labels_`（$n$ 次元整数ベクトル） |
| 慣性（WCSS） | $\sum_{i=1}^{n} \|\mathbf{x}_i - \boldsymbol{\mu}_{z_i}\|^2$ | `inertia_` |
| 反復回数 | — | `n_iter_` |
| 二乗距離行列 | $D_{ij} = \|\mathbf{x}_i - \mathbf{y}_j\|^2$ | `squared_euclidean_distances_()` |
| k-means++初期化 | — | `init_kmeans_plusplus_()` |
| Lloyd反復 | E-step / M-step | `lloyd_iter_()` |
| 収束許容誤差 | $\tau$ | `tol_`、`tolerance_()` |

### 二乗ユークリッド距離の計算

実装では、二乗ユークリッド距離を効率的に計算するため、以下の展開公式を使用する：

$$
\|\mathbf{x}_i - \mathbf{y}_j\|^2 = \|\mathbf{x}_i\|^2 + \|\mathbf{y}_j\|^2 - 2\mathbf{x}_i^\top \mathbf{y}_j
$$

行列形式では、

$$
D = \mathbf{a}\mathbf{1}^\top + \mathbf{1}\mathbf{b}^\top - 2XY^\top
$$

ここで $\mathbf{a}_i = \|\mathbf{x}_i\|^2$、$\mathbf{b}_j = \|\mathbf{y}_j\|^2$ である。これにより、$O(nKd)$ で距離行列を計算できる。

---

## 使用例

### 基本的な使用法

```cpp
#include "clustering/k-means/kmeans.hpp"

// K-meansオブジェクトの作成
gprcpp::clustering::KMeans kmeans(
    5,           // n_clusters: クラスタ数
    300,         // max_iter: 最大反復回数
    1e-4,        // tol: 収束許容誤差
    42           // random_state: 乱数シード
);

// データに対してフィット
Eigen::MatrixXd X(n_samples, n_features);
kmeans.fit(X);

// 結果の取得
const Eigen::MatrixXd& centers = kmeans.cluster_centers();  // (K, d)
const Eigen::VectorXi& labels = kmeans.labels();            // (n,)
double inertia = kmeans.inertia();
int n_iter = kmeans.n_iter();
```

### fit_predict

```cpp
// フィットとラベル取得を一括で行う
Eigen::VectorXi labels = kmeans.fit_predict(X);
```

### predict（新規データへの予測）

```cpp
// 学習済みモデルで新規データのラベルを予測
Eigen::MatrixXd X_new(n_new_samples, n_features);
Eigen::VectorXi labels_new = kmeans.predict(X_new);
```

### transform（クラスタ距離空間への変換）

```cpp
// 各サンプルから各クラスタ中心への距離を計算
Eigen::MatrixXd distances = kmeans.transform(X);  // (n, K)
```

### 初期中心の指定

```cpp
Eigen::MatrixXd init_centers(5, n_features);  // 手動で初期中心を設定
// ... init_centers を初期化 ...

gprcpp::clustering::KMeans kmeans(
    5,                              // n_clusters
    300,                            // max_iter
    1e-4,                           // tol
    std::nullopt,                   // random_state
    init_centers                    // init: 初期中心
);
kmeans.fit(X);
```

---

## 利点と制限

### 利点

1. **高速**: 各反復が $O(nKd)$ で、大規模データにも適用可能
2. **シンプル**: 実装が容易で、解釈しやすい
3. **スケーラビリティ**: サンプル数 $n$ に対して線形の計算量
4. **k-means++初期化**: ランダム初期化よりも高品質な解を得やすい
5. **収束保証**: 有限回の反復で必ず収束する

### 制限

1. **$K$ の事前指定**: クラスタ数を事前に決める必要がある（エルボー法やシルエット法で選択可能）
2. **局所最適解**: 大域的最適解は保証されない。複数回実行して最良解を選ぶことが推奨される
3. **球状クラスタの仮定**: 各クラスタが球状（等方的）であることを暗に仮定。楕円形や複雑な形状のクラスタには不向き
4. **スケール依存**: 特徴量のスケールが異なると、スケールの大きい特徴量が支配的になる。必要に応じて事前に標準化を行う
5. **外れ値に敏感**: 外れ値がセントロイドを引き寄せ、解の品質を低下させる

### 適用が適切な場面

- 大規模データの高速クラスタリング
- クラスタが凸状・球状であることが期待される場合
- ベクトル量子化（画像圧縮、特徴量の離散化など）
- 他のアルゴリズムの前処理（例: 階層クラスタリングの初期化）

### 適用が不適切な場面

- クラスタ数が未知で、データから決定したい場合（DBSCAN、階層クラスタリングを検討）
- 非凸・複雑な形状のクラスタ（DBSCAN、スペクトラルクラスタリングを検討）
- 外れ値が多いデータ（K-medoidsを検討）
- カテゴリカルデータ（K-modesを検討）

---

## 参考文献

### 主要な文献

1. **Lloyd, S. P. (1982)**
   - "Least Squares Quantization in PCM"
   - *IEEE Transactions on Information Theory*, 28(2), 129–137.
   - K-meansアルゴリズム（Lloyd法）の原論文（1957年の内部レポートが元）

2. **MacQueen, J. (1967)**
   - "Some Methods for Classification and Analysis of Multivariate Observations"
   - *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1, 281–297.
   - "K-means" という名前を導入した論文

3. **Arthur, D., & Vassilvitskii, S. (2007)**
   - "k-means++: The Advantages of Careful Seeding"
   - *Proceedings of the 18th Annual ACM-SIAM Symposium on Discrete Algorithms (SODA)*, 1027–1035.
   - k-means++初期化法の提案と $O(\log K)$ 競争比の証明

### 計算量・収束性

4. **Inaba, M., Katoh, N., & Imai, H. (1994)**
   - "Applications of Weighted Voronoi Diagrams and Randomization to Variance-Based k-Clustering"
   - *Proceedings of the 10th ACM Symposium on Computational Geometry*, 332–339.
   - K-means問題のNP困難性

5. **Bottou, L., & Bengio, Y. (1995)**
   - "Convergence Properties of the K-Means Algorithms"
   - *Advances in Neural Information Processing Systems (NIPS)*, 7, 585–592.
   - Lloyd法の収束性の解析

### 実装

6. **scikit-learn: sklearn.cluster.KMeans**
   - https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
   - 本実装が挙動を参考にしている API

### 関連手法

- **K-medoids (PAM)**: セントロイドの代わりにメドイド（実データ点）を使用
- **Mini-batch K-means**: ミニバッチで更新し、大規模データに対応
- **K-means||**: 並列化に適した初期化法
- **Gaussian Mixture Models (GMM)**: 確率的クラスタリング、EMアルゴリズム
- **DBSCAN**: 密度ベースのクラスタリング、$K$ の事前指定不要
- **階層クラスタリング**: デンドログラムでクラスタ構造を可視化

---

**関連ドキュメント**

- [include/README.md](../../README.md): ガウス過程回帰をはじめとする本プロジェクトの概要

---

**最終更新**: 2026年1月31日
