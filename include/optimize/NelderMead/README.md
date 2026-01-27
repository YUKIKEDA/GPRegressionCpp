# Nelder-Mead最適化アルゴリズム

## 目次

1. [概要](#概要)
2. [理論的背景](#理論的背景)
3. [アルゴリズムの詳細](#アルゴリズムの詳細)
4. [パラメータ設定](#パラメータ設定)
5. [収束判定](#収束判定)
6. [利点と制限](#利点と制限)
7. [実装の注意点](#実装の注意点)
8. [参考文献](#参考文献)

---

## 概要

Nelder-Mead法（シンプレックス法、Downhill Simplex法とも呼ばれる）は、1965年にJohn NelderとRoger Meadによって提案された、導関数を必要としない直接探索型の最適化アルゴリズムです。このアルゴリズムは、目的関数の勾配情報が利用できない場合や、目的関数が非滑らかである場合に特に有効です。

### 主な特徴

- **導関数不要**: 目的関数の勾配やヘッセ行列を計算する必要がない
- **直接探索**: 関数値のみを使用して最適解を探索
- **シンプレックス**: n次元空間における(n+1)個の頂点からなるシンプレックス（単体）を操作
- **適応的**: シンプレックスの形状を適応的に変更しながら最適解に収束

---

## 理論的背景

### シンプレックスとは

n次元空間における**シンプレックス（単体）**は、n+1個の線形独立な点（頂点）からなる幾何学的形状です。

- **1次元**: 2点からなる線分
- **2次元**: 3点からなる三角形
- **3次元**: 4点からなる四面体
- **n次元**: n+1点からなる超多面体

### 基本的な考え方

Nelder-Mead法は、シンプレックスの各頂点における目的関数の値を評価し、以下の操作を繰り返すことで最適解を探索します：

1. **評価**: 各頂点での目的関数値を計算
2. **並び替え**: 頂点を目的関数値の順に並び替え（最良、次点、最悪）
3. **反射・拡張・収縮・縮小**: シンプレックスの形状を変更
4. **収束判定**: 終了条件をチェック

### 数学的定式化

n次元最適化問題を考える：

$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$

初期シンプレックスは、初期点 $\mathbf{x}_0$ から生成される：

$$
\mathbf{x}_i = \mathbf{x}_0 + h_i \mathbf{e}_i, \quad i = 1, 2, \ldots, n
$$

ここで、$h_i$ は初期ステップサイズ、$\mathbf{e}_i$ は単位ベクトルです。

---

## アルゴリズムの詳細

### アルゴリズムの流れ

#### ステップ1: 初期化

1. 初期点 $\mathbf{x}_0$ と初期ステップサイズ $h$ を設定
2. 初期シンプレックス $\mathcal{S} = \{\mathbf{x}_0, \mathbf{x}_1, \ldots, \mathbf{x}_n\}$ を生成
3. 各頂点での目的関数値 $f(\mathbf{x}_i)$ を計算

#### ステップ2: 反復処理

各反復で以下の手順を実行：

##### 2.1 頂点の並び替え

目的関数値に基づいて頂点を並び替え：

- **最良点** $\mathbf{x}_b$: $f(\mathbf{x}_b) = \min_i f(\mathbf{x}_i)$
- **次点** $\mathbf{x}_s$: 2番目に良い点
- **最悪点** $\mathbf{x}_w$: $f(\mathbf{x}_w) = \max_i f(\mathbf{x}_i)$
- **重心** $\mathbf{x}_c$: 最悪点を除く全頂点の重心

$$
\mathbf{x}_c = \frac{1}{n} \sum_{i \neq w} \mathbf{x}_i
$$

##### 2.2 反射（Reflection）

最悪点を重心の反対側に反射：

$$
\mathbf{x}_r = \mathbf{x}_c + \alpha (\mathbf{x}_c - \mathbf{x}_w)
$$

ここで、$\alpha > 0$ は反射係数（通常 $\alpha = 1$）

- $f(\mathbf{x}_r) < f(\mathbf{x}_b)$ の場合 → **拡張**へ
- $f(\mathbf{x}_b) \leq f(\mathbf{x}_r) < f(\mathbf{x}_s)$ の場合 → $\mathbf{x}_w$ を $\mathbf{x}_r$ で置換して次反復へ
- $f(\mathbf{x}_s) \leq f(\mathbf{x}_r) < f(\mathbf{x}_w)$ の場合 → **外側収縮**へ
- $f(\mathbf{x}_r) \geq f(\mathbf{x}_w)$ の場合 → **内側収縮**へ

##### 2.3 拡張（Expansion）

反射が成功した場合、さらに遠くまで探索：

$$
\mathbf{x}_e = \mathbf{x}_c + \gamma (\mathbf{x}_r - \mathbf{x}_c)
$$

ここで、$\gamma > 1$ は拡張係数（通常 $\gamma = 2$）

- $f(\mathbf{x}_e) < f(\mathbf{x}_r)$ の場合 → $\mathbf{x}_w$ を $\mathbf{x}_e$ で置換
- それ以外 → $\mathbf{x}_w$ を $\mathbf{x}_r$ で置換

##### 2.4 外側収縮（Outside Contraction）

反射が部分的に成功した場合：

$$
\mathbf{x}_{oc} = \mathbf{x}_c + \rho (\mathbf{x}_r - \mathbf{x}_c)
$$

ここで、$0 < \rho < 1$ は収縮係数（通常 $\rho = 0.5$）

- $f(\mathbf{x}_{oc}) \leq f(\mathbf{x}_r)$ の場合 → $\mathbf{x}_w$ を $\mathbf{x}_{oc}$ で置換
- それ以外 → **縮小**へ

##### 2.5 内側収縮（Inside Contraction）

反射が失敗した場合：

$$
\mathbf{x}_{ic} = \mathbf{x}_c - \rho (\mathbf{x}_c - \mathbf{x}_w)
$$

- $f(\mathbf{x}_{ic}) < f(\mathbf{x}_w)$ の場合 → $\mathbf{x}_w$ を $\mathbf{x}_{ic}$ で置換
- それ以外 → **縮小**へ

##### 2.6 縮小（Shrink）

シンプレックス全体を最良点に向かって縮小：

$$
\mathbf{x}_i = \mathbf{x}_b + \sigma (\mathbf{x}_i - \mathbf{x}_b), \quad i \neq b
$$

ここで、$0 < \sigma < 1$ は縮小係数（通常 $\sigma = 0.5$）

#### ステップ3: 収束判定

以下のいずれかの条件が満たされた場合、アルゴリズムを終了：

1. **関数値の収束**: 
   $$
   \frac{f(\mathbf{x}_w) - f(\mathbf{x}_b)}{|f(\mathbf{x}_b)| + \epsilon} < \tau
   $$
   ここで、$\tau$ は許容誤差、$\epsilon$ は数値安定性のための小さな値

2. **シンプレックスの収束**:
   $$
   \max_i \|\mathbf{x}_i - \mathbf{x}_c\| < \tau
   $$

3. **最大反復回数**: 指定された最大反復回数に達した

### アルゴリズムの疑似コード

```
function NelderMead(f, x0, options):
    // 初期化
    S = initialize_simplex(x0, h)
    evaluate_all_vertices(S, f)
    
    for iteration = 1 to max_iterations:
        // 並び替え
        sort_vertices(S)  // 最良、次点、最悪を特定
        x_b, x_s, x_w = S.best, S.second, S.worst
        x_c = compute_centroid(S, exclude=x_w)
        
        // 収束判定
        if converged(S, options.tolerance):
            return OptimizationResult(x_b, f(x_b), converged=true)
        
        // 反射
        x_r = x_c + alpha * (x_c - x_w)
        f_r = f(x_r)
        
        if f_r < f(x_b):
            // 拡張
            x_e = x_c + gamma * (x_r - x_c)
            if f(x_e) < f_r:
                replace(S, x_w, x_e)
            else:
                replace(S, x_w, x_r)
        else if f_r < f(x_s):
            replace(S, x_w, x_r)
        else if f_r < f(x_w):
            // 外側収縮
            x_oc = x_c + rho * (x_r - x_c)
            if f(x_oc) <= f_r:
                replace(S, x_w, x_oc)
            else:
                shrink(S, x_b, sigma)
        else:
            // 内側収縮
            x_ic = x_c - rho * (x_c - x_w)
            if f(x_ic) < f(x_w):
                replace(S, x_w, x_ic)
            else:
                shrink(S, x_b, sigma)
    
    return OptimizationResult(x_b, f(x_b), converged=false)
```

---

## パラメータ設定

### 標準的な係数値

Nelder-Mead法では、以下の係数が使用されます（標準的な値）：

| 係数 | 記号 | 標準値 | 説明 |
|------|------|--------|------|
| 反射係数 | $\alpha$ | 1.0 | 反射操作の強度 |
| 拡張係数 | $\gamma$ | 2.0 | 拡張操作の強度 |
| 収縮係数 | $\rho$ | 0.5 | 収縮操作の強度 |
| 縮小係数 | $\sigma$ | 0.5 | 縮小操作の強度 |

### 初期ステップサイズ

初期シンプレックスの生成に使用されるステップサイズ $h$ は、問題のスケールに応じて設定します：

- **固定値**: 各次元で同じ値（例: $h = 0.1$）
- **適応的**: 各次元のスケールに応じて調整
- **相対的**: 初期点の値に対する割合（例: 5%）

### 推奨設定

一般的な問題に対しては、以下の設定が推奨されます：

```cpp
double alpha = 1.0;   // 反射係数
double gamma = 2.0;   // 拡張係数
double rho = 0.5;     // 収縮係数
double sigma = 0.5;   // 縮小係数
double initial_step = 0.1;  // 初期ステップサイズ
int max_iterations = 1000;  // 最大反復回数
double tolerance = 1e-6;    // 収束許容誤差
```

---

## 収束判定

### 関数値ベースの収束判定

目的関数値の相対的な変化が十分に小さくなった場合に収束と判定：

$$
\frac{f(\mathbf{x}_w) - f(\mathbf{x}_b)}{|f(\mathbf{x}_b)| + \epsilon} < \tau
$$

ここで：
- $f(\mathbf{x}_w)$: 最悪点での目的関数値
- $f(\mathbf{x}_b)$: 最良点での目的関数値
- $\epsilon$: 数値安定性のための小さな値（例: $10^{-10}$）
- $\tau$: 許容誤差（例: $10^{-6}$）

### シンプレックスサイズベースの収束判定

シンプレックスの各頂点が重心に十分近づいた場合に収束と判定：

$$
\max_i \|\mathbf{x}_i - \mathbf{x}_c\| < \tau
$$

ここで：
- $\mathbf{x}_c$: 重心
- $\|\cdot\|$: ユークリッドノルム
- $\tau$: 許容誤差

### 実装上の注意点

1. **数値安定性**: ゼロ除算を避けるため、分母に小さな値 $\epsilon$ を加える
2. **相対誤差**: 絶対誤差ではなく相対誤差を使用することで、スケールに依存しない判定が可能
3. **複数条件**: 関数値とシンプレックスサイズの両方をチェックすることで、より確実な収束判定が可能

---

## 利点と制限

### 利点

1. **導関数不要**: 目的関数の勾配やヘッセ行列を計算する必要がない
2. **実装が簡単**: アルゴリズムが比較的シンプルで実装しやすい
3. **非滑らかな関数**: 目的関数が非滑らかでも適用可能
4. **ノイズ耐性**: 目的関数にノイズが含まれていても比較的安定
5. **低次元問題**: 低次元（$n \leq 10$）の問題に対して効果的

### 制限

1. **高次元問題**: 次元が高い（$n > 10$）場合、収束が遅くなる
2. **局所最適解**: 大域的最適解を保証しない（局所最適解に収束する可能性）
3. **収束保証**: 理論的な収束保証がない（実用的には多くの場合収束する）
4. **パラメータ依存**: 係数の選択が性能に影響を与える
5. **境界制約**: 元のアルゴリズムは制約なし最適化用（境界制約の処理には追加の工夫が必要）

### 適用が適切な問題

- 目的関数の導関数が計算困難または利用不可能
- 目的関数が非滑らかまたはノイズを含む
- 低次元から中次元（$n \leq 20$）の最適化問題
- 初期値が最適解に近い場合
- 計算コストが高い目的関数（関数評価回数を最小化したい）

### 適用が不適切な問題

- 高次元（$n > 50$）の最適化問題
- 大域的最適解が必須な問題
- 目的関数が滑らかで勾配情報が利用可能な問題（勾配法の方が効率的）
- 強い制約条件がある問題

---

## 実装の注意点

実装の詳細については、実装本体（`nelder_mead.hpp`）を参照してください。ここでは、パフォーマンスと数値安定性の観点からの注意点をまとめます。

### パフォーマンスに関する注意点

1. **関数評価の最適化**: 目的関数の評価は最も計算コストが高い操作の一つです。同じ点での再評価を避けるため、必要に応じてキャッシュを検討してください。

2. **メモリ効率**: 
   - シンプレックスの頂点を保持する際、不要なコピーを避ける
   - Eigenの`VectorXd`を使用する場合、適切なメモリ管理を考慮

3. **早期終了**: 収束判定を各反復でチェックし、収束した場合は即座に終了することで、不要な計算を避けられます。

4. **並列化**: 複数の頂点での関数評価を並列化することは可能ですが、通常シンプレックスのサイズが小さい（n+1個）ため、オーバーヘッドを考慮すると効果は限定的です。

### 数値安定性に関する注意点

1. **ゼロ除算の回避**: 収束判定の際、分母に小さな値（例: $10^{-10}$）を加えることで、ゼロ除算を避けます。

2. **境界制約の処理**: 反射・拡張・収縮後の点が境界外に出た場合、適切にクリップする必要があります。単純なクリップでは最適解が境界上にある場合に問題が生じる可能性があるため、実装時には注意が必要です。

3. **シンプレックスの退化**: シンプレックスが退化（頂点が一直線上に並ぶ）した場合、アルゴリズムが停滞する可能性があります。必要に応じて、シンプレックスの再初期化を検討してください。

---

## 参考文献

### 主要な論文

1. **Nelder, J. A., & Mead, R. (1965)**
   - "A Simplex Method for Function Minimization"
   - *The Computer Journal*, 7(4), 308-313
   - オリジナルのNelder-Mead法の提案論文

2. **Lagarias, J. C., et al. (1998)**
   - "Convergence Properties of the Nelder-Mead Simplex Method in Low Dimensions"
   - *SIAM Journal on Optimization*, 9(1), 112-147
   - 低次元での収束性の理論的解析

### 実装に関する参考資料

3. **Gao, F., & Han, L. (2012)**
   - "Implementing the Nelder-Mead simplex algorithm with adaptive parameters"
   - *Computational Optimization and Applications*, 51(1), 259-277
   - 適応的パラメータを用いた改良版の実装

4. **Numerical Recipes** (Press et al.)
   - "Downhill Simplex Method in Multidimensions"
   - 実装の詳細と実用的なアドバイス

### オンラインリソース

5. **Wikipedia: Nelder-Mead method**
   - https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

6. **SciPy Documentation: scipy.optimize.minimize**
   - Nelder-Mead法の実装と使用例

### 関連アルゴリズム

- **Powell's Method**: 導関数不要の別の直接探索法
- **Particle Swarm Optimization (PSO)**: 群知能に基づく最適化
- **Differential Evolution**: 進化的アルゴリズム
- **Simulated Annealing**: メタヒューリスティック最適化

---

## 付録: アルゴリズムの可視化

### 2次元問題での動作

2次元最適化問題では、シンプレックスは三角形として可視化できます：

1. **初期状態**: 3つの頂点からなる三角形
2. **反射**: 最悪点を重心の反対側に反射
3. **拡張**: 反射が成功した場合、さらに遠くまで探索
4. **収縮**: 反射が部分的に成功した場合、シンプレックスを縮小
5. **縮小**: すべての操作が失敗した場合、最良点に向かって全体を縮小

このプロセスを繰り返すことで、シンプレックスは最適解に向かって移動・収縮していきます。

---

**最終更新**: 2026年1月27日
