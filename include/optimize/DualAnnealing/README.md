# Dual Annealing最適化アルゴリズム

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

Dual Annealing（二重アニーリング法）は、Xiang et al. (1997)によって提案されたGeneralized Simulated Annealing (GSA)に基づく大域的最適化アルゴリズムです。このアルゴリズムは、Classical Simulated Annealing (CSA)とFast Simulated Annealing (FSA)の特性を組み合わせ、さらに局所探索（Local Search）を統合することで、多峰性関数の大域的最適解を効率的に探索します。

### 主な特徴

- **大域的最適化**: 複数の局所最適解を持つ問題に対して、大域的最適解を探索可能
- **Generalized Simulated Annealing**: Tsallis統計力学に基づく一般化されたアニーリングスケジュール
- **二重アニーリング**: 訪問分布と受容確率を独立に制御する二重の温度パラメータ
- **局所探索の統合**: アニーリングで受け入れられた解に対して局所最適化を適用
- **導関数不要**: 目的関数の勾配情報を必要としない

---

## 理論的背景

### Simulated Annealingの基礎

Simulated Annealing（SA）は、金属の焼きなまし（annealing）プロセスにヒントを得たメタヒューリスティック最適化アルゴリズムです。高温から徐々に温度を下げることで、金属の結晶構造が最適な状態に収束する過程を模倣します。

最適化問題では、以下のように対応します：

- **状態**: 探索空間内の点 $\mathbf{x}$
- **エネルギー**: 目的関数値 $f(\mathbf{x})$
- **温度**: 探索のランダム性を制御するパラメータ $T$
- **平衡状態**: 最適解

### Classical Simulated Annealing (CSA)

古典的なSimulated Annealingでは、新しい状態 $\mathbf{x}'$ を現在の状態 $\mathbf{x}$ から生成し、以下の確率で受け入れます：

$$
P_{\text{accept}}(\mathbf{x} \to \mathbf{x}') = \min\left(1, \exp\left(-\frac{\Delta E}{T}\right)\right)
$$

ここで、$\Delta E = f(\mathbf{x}') - f(\mathbf{x})$ はエネルギー差、$T$ は現在の温度です。

### Fast Simulated Annealing (FSA)

Fast Simulated Annealingは、より広範囲な探索を可能にするため、訪問分布（visiting distribution）を変更したバリアントです。Cauchy分布に基づく重い裾を持つ分布を使用することで、より遠方へのジャンプを可能にします。

### Generalized Simulated Annealing (GSA)

Xiang et al. (1997)は、Tsallis統計力学に基づいて、CSAとFSAを一般化したGeneralized Simulated Annealingを提案しました。GSAでは、訪問分布と受容確率を独立に制御する2つのパラメータ $q_v$（訪問パラメータ）と $q_a$（受容パラメータ）を導入します。

### 訪問分布（Visiting Distribution）

新しい状態の生成に使用される分布は、以下の一般化されたCauchy-Lorentz分布に基づきます：

$$
g_q(\Delta \mathbf{x}, T) \propto \frac{T^{-\frac{D}{3-q_v}}}{\left[1 + (q_v - 1)\frac{(\Delta \mathbf{x})^2}{T^{\frac{2}{3-q_v}}}\right]^{\frac{1}{q_v-1} + \frac{D-1}{2}}}
$$

ここで：
- $D$: 次元数
- $q_v$: 訪問パラメータ（$1 < q_v < 3$）
- $T$: 現在の温度
- $\Delta \mathbf{x}$: 現在の点からの変位

$q_v \to 1$ の極限では、GSAはCSAに近づき、$q_v \to 3$ に近づくとFSAに近づきます。

### 受容確率（Acceptance Probability）

新しい状態を受け入れる確率は、以下の一般化された形式で与えられます：

$$
P_{\text{accept}}(\mathbf{x} \to \mathbf{x}') = \min\left(1, \left[1 + (q_a - 1)\frac{\Delta E}{T}\right]^{-\frac{1}{q_a-1}}\right)
$$

ここで：
- $q_a$: 受容パラメータ（$q_a < 1$）
- $\Delta E = f(\mathbf{x}') - f(\mathbf{x})$
- $T$: 現在の温度

$q_a \to 1$ の極限では、古典的なSAの受容確率に収束します。

### 温度スケジュール

温度は、訪問パラメータ $q_v$ に依存するべき乗則に従って減少します：

$$
T(k) = T_0 \left(1 - \frac{k}{K}\right)^{\frac{1}{q_v-1}}
$$

ここで：
- $T_0$: 初期温度
- $k$: 現在の反復回数
- $K$: 最大反復回数

---

## アルゴリズムの詳細

### アルゴリズムの流れ

#### ステップ1: 初期化

1. 初期点 $\mathbf{x}_0$ を設定
2. 初期温度 $T_0$ を設定
3. 訪問パラメータ $q_v$ と受容パラメータ $q_a$ を設定
4. 最大反復回数 $K$ を設定
5. 現在の最良解 $\mathbf{x}_{\text{best}} = \mathbf{x}_0$、$f_{\text{best}} = f(\mathbf{x}_0)$ を記録

#### ステップ2: アニーリング反復

各反復 $k = 1, 2, \ldots, K$ で以下の手順を実行：

##### 2.1 温度の更新

$$
T(k) = T_0 \left(1 - \frac{k}{K}\right)^{\frac{1}{q_v-1}}
$$

##### 2.2 新しい状態の生成

現在の状態 $\mathbf{x}$ から、訪問分布に従って新しい状態 $\mathbf{x}'$ を生成：

$$
\mathbf{x}' = \mathbf{x} + \Delta \mathbf{x}
$$

ここで、$\Delta \mathbf{x}$ は一般化されたCauchy-Lorentz分布からサンプリングされます。

境界制約がある場合、生成された点を境界内にクリップします。

##### 2.3 受容判定

新しい状態 $\mathbf{x}'$ の目的関数値 $f(\mathbf{x}')$ を計算し、以下の確率で受け入れます：

$$
P_{\text{accept}} = \min\left(1, \left[1 + (q_a - 1)\frac{f(\mathbf{x}') - f(\mathbf{x})}{T(k)}\right]^{-\frac{1}{q_a-1}}\right)
$$

- 受け入れられた場合: $\mathbf{x} \leftarrow \mathbf{x}'$
- 拒否された場合: 現在の状態を維持

##### 2.4 最良解の更新

$f(\mathbf{x}') < f_{\text{best}}$ の場合：
- $\mathbf{x}_{\text{best}} \leftarrow \mathbf{x}'$
- $f_{\text{best}} \leftarrow f(\mathbf{x}')$

##### 2.5 局所探索の適用（オプション）

受け入れられた状態に対して、局所最適化アルゴリズム（例: L-BFGS-B、Nelder-Mead）を適用して、近傍の局所最適解を探索します。局所探索で改善された解があれば、それを新しい状態として採用します。

##### 2.6 再アニーリング（Reannealing）

温度が一定の閾値（例: $T(k) < T_0 \times \text{restart\_temp\_ratio}$）以下になった場合、温度を再初期化して探索を継続します。これにより、局所最適解から脱出する機会を提供します。

#### ステップ3: 終了判定

以下のいずれかの条件が満たされた場合、アルゴリズムを終了：

1. **最大反復回数**: $k \geq K$
2. **最大関数評価回数**: 目的関数の評価回数が上限に達した
3. **収束判定**: 最良解の改善が一定期間見られない場合（実装依存）

### アルゴリズムの疑似コード

```
function DualAnnealing(f, x0, options):
    // 初期化
    x = x0
    x_best = x0
    f_best = f(x0)
    T0 = options.initial_temp
    qv = options.visit_param      // 訪問パラメータ
    qa = options.accept_param      // 受容パラメータ
    K = options.max_iterations
    
    for k = 1 to K:
        // 温度の更新
        T = T0 * (1 - k/K)^(1/(qv-1))
        
        // 新しい状態の生成
        delta_x = sample_visiting_distribution(T, qv)
        x_new = x + delta_x
        x_new = clip_to_bounds(x_new, bounds)
        
        // 目的関数の評価
        f_new = f(x_new)
        delta_E = f_new - f(x)
        
        // 受容判定
        if delta_E < 0:
            accept = true
        else:
            accept_prob = min(1, [1 + (qa-1)*delta_E/T]^(-1/(qa-1)))
            accept = (random() < accept_prob)
        
        if accept:
            x = x_new
            f_current = f_new
            
            // 最良解の更新
            if f_new < f_best:
                x_best = x_new
                f_best = f_new
            
            // 局所探索の適用
            if options.use_local_search:
                x_local = local_minimize(f, x, local_options)
                if f(x_local) < f_best:
                    x_best = x_local
                    f_best = f(x_local)
                    x = x_local
        
        // 再アニーリング
        if T < T0 * options.restart_temp_ratio:
            T = T0 * options.restart_temp_ratio
    
    return OptimizationResult(x_best, f_best)
```

### 訪問分布のサンプリング

一般化されたCauchy-Lorentz分布からのサンプリングは、以下の手順で行います：

1. 一様乱数 $u \in [0, 1]$ を生成
2. 以下の変換を使用して変位を計算：

$$
\Delta x_i = T^{\frac{1}{3-q_v}} \cdot \text{sign}(u - 0.5) \cdot \left[\left((2u - 1)^{q_v-1} + 1\right)^{\frac{1}{q_v-1}} - 1\right]
$$

各次元 $i = 1, 2, \ldots, D$ に対して独立にサンプリングします。

---

## パラメータ設定

### 主要パラメータ

| パラメータ | 記号 | 推奨値 | 説明 |
|-----------|------|--------|------|
| 初期温度 | $T_0$ | 5230 | 探索の広さを制御（範囲: 0.01-50000） |
| 訪問パラメータ | $q_v$ | 2.62 | 訪問分布の形状（範囲: 1-3、推奨: 2.0-2.7） |
| 受容パラメータ | $q_a$ | -5.0 | 受容確率の形状（範囲: -10000 ～ -5） |
| 再アニーリング比率 | - | 2e-5 | 再アニーリングをトリガーする温度比率 |
| 最大反復回数 | $K$ | 1000 | グローバル探索の最大反復回数 |
| 最大関数評価回数 | - | 1e7 | 目的関数評価のソフトリミット |

### パラメータの選択指針

#### 初期温度 $T_0$

初期温度は、探索空間の広さと目的関数のスケールに応じて設定します：

- **低い値** ($T_0 < 100$): 局所探索に近い動作、収束が速いが大域的最適解を見逃す可能性
- **高い値** ($T_0 > 10000$): 広範囲な探索、収束が遅いが大域的最適解を見つけやすい
- **推奨**: 目的関数値の範囲に応じて調整（例: $T_0 \approx 100 \times |f(\mathbf{x}_0)|$）

#### 訪問パラメータ $q_v$

訪問パラメータは、探索のジャンプ幅を制御します：

- **$q_v \to 1$**: CSAに近い、局所的な探索
- **$q_v \approx 2.62$**: バランスの取れた探索（推奨）
- **$q_v \to 3$**: FSAに近い、広範囲な探索

#### 受容パラメータ $q_a$

受容パラメータは、悪い解を受け入れる確率を制御します：

- **$q_a \to -\infty$**: 悪い解をほとんど受け入れない（貪欲的）
- **$q_a \approx -5.0$**: バランスの取れた受容（推奨）
- **$q_a \to 1$**: 古典的なSAの受容確率

### 推奨設定

一般的な問題に対しては、以下の設定が推奨されます：

```cpp
double initial_temp = 5230.0;        // 初期温度
double visit_param = 2.62;           // 訪問パラメータ
double accept_param = -5.0;          // 受容パラメータ
double restart_temp_ratio = 2e-5;   // 再アニーリング比率
int max_iterations = 1000;           // 最大反復回数
int max_function_evaluations = 1e7;  // 最大関数評価回数
bool use_local_search = true;        // 局所探索の使用
```

### 局所探索の設定

局所探索には、以下のアルゴリズムが使用可能です：

- **L-BFGS-B**: 境界制約付き準ニュートン法（推奨、導関数が必要）
- **Nelder-Mead**: シンプレックス法（導関数不要）
- **Powell**: 導関数不要の共役方向法

局所探索のパラメータ（最大反復回数、許容誤差など）も適切に設定する必要があります。

---

## 収束判定

### 反復回数ベースの終了

最大反復回数 $K$ に達した場合、アルゴリズムを終了します。これは最も単純な終了条件です。

### 関数評価回数ベースの終了

目的関数の評価回数が上限（例: $10^7$）に達した場合、アルゴリズムを終了します。計算コストが高い目的関数に対して有効です。

### 改善ベースの終了（実装依存）

一定期間（例: 100反復）にわたって最良解が改善されない場合、アルゴリズムを終了します。ただし、Dual Annealingは本質的に確率的なアルゴリズムであるため、改善が一時的に停止しても後で改善する可能性があります。

### 実装上の注意点

1. **確率的アルゴリズム**: Dual Annealingは確率的なアルゴリズムであるため、収束の定義が明確ではありません。通常は、最大反復回数または関数評価回数で終了します。

2. **再アニーリング**: 温度が低くなりすぎた場合、再アニーリングによって探索を継続します。これにより、局所最適解から脱出する機会が提供されます。

3. **局所探索の効果**: 局所探索を適用することで、近傍の局所最適解に素早く収束できますが、大域的最適解を見逃す可能性もあります。バランスの取れた設定が重要です。

---

## 利点と制限

### 利点

1. **大域的最適化**: 複数の局所最適解を持つ問題に対して、大域的最適解を探索可能
2. **導関数不要**: 目的関数の勾配やヘッセ行列を計算する必要がない
3. **柔軟な探索**: 訪問パラメータと受容パラメータを調整することで、探索戦略をカスタマイズ可能
4. **局所探索の統合**: アニーリングと局所探索を組み合わせることで、効率的な探索が可能
5. **境界制約対応**: 境界制約付き最適化問題に適用可能
6. **非滑らかな関数**: 目的関数が非滑らかでも適用可能

### 制限

1. **計算コスト**: 多くの関数評価が必要で、計算コストが高い
2. **パラメータ調整**: 適切なパラメータ設定が性能に大きく影響する
3. **収束保証**: 大域的最適解への収束は確率的に保証されるが、有限時間での保証はない
4. **高次元問題**: 次元が高い（$D > 50$）場合、探索空間が広すぎて効率が低下する可能性
5. **確率的アルゴリズム**: 同じ問題を複数回実行しても、異なる結果が得られる可能性がある

### 適用が適切な問題

- 複数の局所最適解を持つ非凸最適化問題
- 目的関数の導関数が計算困難または利用不可能
- 目的関数が非滑らかまたはノイズを含む
- 中次元から高次元（$D \leq 100$）の最適化問題
- 境界制約付き最適化問題
- 大域的最適解が重要な問題

### 適用が不適切な問題

- 非常に高次元（$D > 200$）の最適化問題
- 目的関数が滑らかで勾配情報が利用可能な問題（勾配法の方が効率的）
- 計算コストが非常に高い目的関数（関数評価回数を最小化したい場合）
- リアルタイム最適化が必要な問題

---

## 実装の注意点

実装の詳細については、実装本体（`dual_annealing.hpp`）を参照してください。ここでは、パフォーマンスと数値安定性の観点からの注意点をまとめます。

### パフォーマンスに関する注意点

1. **関数評価の最適化**: 目的関数の評価は最も計算コストが高い操作の一つです。可能であれば、関数評価の結果をキャッシュすることを検討してください。

2. **局所探索の頻度**: すべての受け入れられた状態に対して局所探索を適用すると、計算コストが非常に高くなります。一定の間隔で局所探索を適用する、または改善が見られた場合のみ適用するなどの戦略を検討してください。

3. **並列化**: 複数の独立なアニーリングプロセスを並列実行し、最良の結果を選択することで、大域的最適解を見つける確率を向上させることができます。

4. **メモリ効率**: 訪問分布のサンプリングや状態の保持に使用するメモリを適切に管理してください。

### 数値安定性に関する注意点

1. **訪問分布のサンプリング**: $q_v \to 3$ に近い場合、訪問分布のサンプリングで数値的不安定性が生じる可能性があります。適切な数値的手法（例: 対数空間での計算）を使用してください。

2. **受容確率の計算**: $q_a$ が非常に小さい（負の大きな絶対値）場合、受容確率の計算で数値的オーバーフローが発生する可能性があります。適切なクリッピングや対数空間での計算を検討してください。

3. **温度の更新**: 温度が非常に小さくなった場合、除算による数値的不安定性が生じる可能性があります。最小温度の閾値を設定することを推奨します。

4. **境界制約の処理**: 生成された点が境界外に出た場合、単純なクリップでは最適解が境界上にある場合に問題が生じる可能性があります。反射や折り返しなどの手法を検討してください。

### 再アニーリングの実装

再アニーリングは、局所最適解から脱出するために重要です。実装時には、以下の点を考慮してください：

1. **再アニーリングの頻度**: 頻繁すぎる再アニーリングは探索を無駄にし、少なすぎる再アニーリングは局所最適解に留まる可能性があります。

2. **温度の再設定**: 再アニーリング時に、初期温度の一定比率（例: `restart_temp_ratio`）に温度を設定します。

3. **状態の保持**: 再アニーリング時も、これまでに見つかった最良解を保持し続けます。

---

## 参考文献

### 主要な論文

1. **Xiang, Y., et al. (1997)**
   - "Generalized Simulated Annealing Algorithm and Its Application to the Thomson Model"
   - *Physics Letters A*, 233(3), 216-220
   - Generalized Simulated Annealing (GSA)の基礎となる論文

2. **Tsallis, C. (1988)**
   - "Possible generalization of Boltzmann-Gibbs statistics"
   - *Journal of Statistical Physics*, 52(1-2), 479-487
   - Tsallis統計力学の基礎理論

3. **Xiang, Y., et al. (2013)**
   - "Generalized Simulated Annealing for Global Optimization: The GenSA Package"
   - *The R Journal*, 5(1), 13-29
   - RパッケージGenSAのドキュメントと理論的背景

### 実装に関する参考資料

4. **SciPy Documentation: scipy.optimize.dual_annealing**
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html
   - SciPyのdual_annealing実装のドキュメント

5. **SciPy Source Code: _dual_annealing.py**
   - https://github.com/scipy/scipy/blob/main/scipy/optimize/_dual_annealing.py
   - SciPyの実装ソースコード

### オンラインリソース

6. **Wikipedia: Simulated Annealing**
   - https://en.wikipedia.org/wiki/Simulated_annealing
   - Simulated Annealingの基礎理論

7. **Wikipedia: Tsallis Statistics**
   - https://en.wikipedia.org/wiki/Tsallis_statistics
   - Tsallis統計力学の概要

### 関連アルゴリズム

- **Classical Simulated Annealing (CSA)**: 古典的な焼きなまし法
- **Fast Simulated Annealing (FSA)**: 高速焼きなまし法
- **Differential Evolution**: 進化的アルゴリズム
- **Particle Swarm Optimization (PSO)**: 群知能に基づく最適化
- **Genetic Algorithms**: 遺伝的アルゴリズム

---

## 付録: アルゴリズムの可視化

### 探索プロセスの可視化

Dual Annealingの探索プロセスは、以下のように可視化できます：

1. **初期段階（高温）**: 広範囲にわたってランダムに探索し、大まかな最適解の位置を特定
2. **中間段階（中温）**: 有望な領域を集中的に探索し、局所最適解を発見
3. **後期段階（低温）**: 局所探索を適用して、局所最適解を精密化
4. **再アニーリング**: 温度を再設定して、新しい領域を探索

### パラメータの影響

- **$q_v$ の影響**: $q_v$ が大きいほど、より遠方へのジャンプが可能になり、大域的な探索が強化されます
- **$q_a$ の影響**: $q_a$ が小さい（負の大きな絶対値）ほど、悪い解を受け入れにくくなり、より貪欲的な探索になります
- **$T_0$ の影響**: $T_0$ が大きいほど、初期段階での探索範囲が広がります

---

**最終更新**: 2026年1月27日
