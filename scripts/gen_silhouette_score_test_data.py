#!/usr/bin/env python3
"""
シルエットスコア テストデータ生成スクリプト

C++ で実装したシルエット係数（Silhouette Coefficient）の検証のため、
scikit-learn の silhouette_samples / silhouette_score で期待値を計算し、
JSON 形式で出力します。

データ行列 X の形状（scikit-learn と同じ convention）:
    X は (n_samples, n_features)
    - 行 (1次元目) = サンプル数 (n_samples)
    - 列 (2次元目) = 説明変数・特徴量の数 (n_features)

必要なパッケージ:
    pip install numpy scikit-learn

使用方法:
    python scripts/gen_silhouette_score_test_data.py

出力:
    tests/metrics/SilhouetteScore/data/silhouette_score_test_data.json
"""

import json
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    silhouette_samples,
    silhouette_score,
    pairwise_distances,
)


def get_output_path():
    """出力先ディレクトリとファイルパスを取得"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "tests" / "metrics" / "SilhouetteScore" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "silhouette_score_test_data.json"


class NumpyEncoder(json.JSONEncoder):
    """NumPy 配列・スカラーを JSON で扱える型に変換するエンコーダ"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        return super().default(obj)


def run_case_1_basic():
    """ケース1: 基本（明確に分離されたクラスタ、n_clusters=3）"""
    print("Generating Case 1 (basic)...")
    n_samples, n_features = 150, 2
    n_clusters = 3

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=42,
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        random_state=42,
    )
    labels = kmeans.fit_predict(X)

    sil_samples = silhouette_samples(X, labels, metric="euclidean")
    sil_score = silhouette_score(X, labels, metric="euclidean")

    return {
        "name": "case1_basic",
        "description": "n_samples=150, n_features=2, n_clusters=3, well-separated blobs",
        "config": {"n_clusters": n_clusters, "random_state": 42},
        "data": {"X": X, "labels": labels},
        "expected": {
            "silhouette_samples": sil_samples,
            "silhouette_score": float(sil_score),
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def run_case_2_high_dimensional():
    """ケース2: 高次元データ（n_features=5, n_clusters=4）"""
    print("Generating Case 2 (high dimensional)...")
    n_samples, n_features = 100, 5
    n_clusters = 4

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=123,
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        random_state=123,
    )
    labels = kmeans.fit_predict(X)

    sil_samples = silhouette_samples(X, labels, metric="euclidean")
    sil_score = silhouette_score(X, labels, metric="euclidean")

    return {
        "name": "case2_high_dimensional",
        "description": "n_samples=100, n_features=5, n_clusters=4",
        "config": {"n_clusters": n_clusters, "random_state": 123},
        "data": {"X": X, "labels": labels},
        "expected": {
            "silhouette_samples": sil_samples,
            "silhouette_score": float(sil_score),
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def run_case_3_two_clusters():
    """ケース3: 2クラスタ（最小のクラスタ数）"""
    print("Generating Case 3 (two clusters)...")
    n_samples, n_features = 80, 3
    n_clusters = 2

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.8,
        random_state=7,
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        random_state=7,
    )
    labels = kmeans.fit_predict(X)

    sil_samples = silhouette_samples(X, labels, metric="euclidean")
    sil_score = silhouette_score(X, labels, metric="euclidean")

    return {
        "name": "case3_two_clusters",
        "description": "n_samples=80, n_features=3, n_clusters=2",
        "config": {"n_clusters": n_clusters, "random_state": 7},
        "data": {"X": X, "labels": labels},
        "expected": {
            "silhouette_samples": sil_samples,
            "silhouette_score": float(sil_score),
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def run_case_4_precomputed():
    """ケース4: 事前計算距離行列（metric='precomputed'）"""
    print("Generating Case 4 (precomputed distances)...")
    n_samples, n_features = 60, 4
    n_clusters = 3

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.6,
        random_state=0,
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        random_state=0,
    )
    labels = kmeans.fit_predict(X)

    # ユークリッド距離行列 (n_samples, n_samples)
    D = pairwise_distances(X, metric="euclidean")

    sil_samples = silhouette_samples(D, labels, metric="precomputed")
    sil_score = silhouette_score(D, labels, metric="precomputed")

    return {
        "name": "case4_precomputed",
        "description": "n_samples=60, n_features=4, n_clusters=3, precomputed distance matrix",
        "config": {"n_clusters": n_clusters, "random_state": 0},
        "data": {"D": D, "labels": labels},
        "expected": {
            "silhouette_samples": sil_samples,
            "silhouette_score": float(sil_score),
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def run_case_5_overlapping():
    """ケース5: 重なりのあるクラスタ（シルエット係数が低め）"""
    print("Generating Case 5 (overlapping clusters)...")
    n_samples, n_features = 120, 2
    n_clusters = 3

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=2.0,
        random_state=999,
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        random_state=999,
    )
    labels = kmeans.fit_predict(X)

    sil_samples = silhouette_samples(X, labels, metric="euclidean")
    sil_score = silhouette_score(X, labels, metric="euclidean")

    return {
        "name": "case5_overlapping",
        "description": "n_samples=120, n_features=2, n_clusters=3, overlapping (cluster_std=2.0)",
        "config": {"n_clusters": n_clusters, "random_state": 999},
        "data": {"X": X, "labels": labels},
        "expected": {
            "silhouette_samples": sil_samples,
            "silhouette_score": float(sil_score),
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def run_case_6_non_consecutive_labels():
    """ケース6: 非連続ラベル（C++ 側で 0..K-1 にエンコードされることを検証）"""
    print("Generating Case 6 (non-consecutive labels)...")
    n_samples, n_features = 50, 2
    n_clusters = 3

    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=42,
    )
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        max_iter=300,
        random_state=42,
    )
    # 0,1,2 を 10, 20, 30 に写像（シルエット値は同じになるはず）
    raw_labels = kmeans.fit_predict(X)
    labels = np.where(raw_labels == 0, 10, np.where(raw_labels == 1, 20, 30))

    sil_samples = silhouette_samples(X, labels, metric="euclidean")
    sil_score = silhouette_score(X, labels, metric="euclidean")

    return {
        "name": "case6_non_consecutive_labels",
        "description": "labels 10, 20, 30 (non-consecutive); silhouette should match 0,1,2",
        "config": {"n_clusters": n_clusters, "random_state": 42},
        "data": {"X": X, "labels": labels},
        "expected": {
            "silhouette_samples": sil_samples,
            "silhouette_score": float(sil_score),
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def main():
    """メイン関数"""
    print("シルエットスコア テストデータ生成を開始...")
    output_path = get_output_path()

    run_fns = [
        run_case_1_basic,
        run_case_2_high_dimensional,
        run_case_3_two_clusters,
        run_case_4_precomputed,
        run_case_5_overlapping,
        run_case_6_non_consecutive_labels,
    ]
    test_cases = []
    for run_fn in run_fns:
        tc = run_fn()
        test_cases.append(tc)
        meta = tc["meta"]
        print(
            f"  → {tc['name']}: n_samples={meta['n_samples']}, "
            f"n_features={meta['n_features']}, n_clusters={meta['n_clusters']}"
        )

    output_data = {
        "description": "シルエット係数のテストデータ（scikit-learn 期待値）",
        "generated_by": "gen_silhouette_score_test_data.py",
        "data_layout": {
            "X_rows": "n_samples",
            "X_cols": "n_features",
            "note": "X は (n_samples, n_features)。precomputed ケースでは data.D が (n_samples, n_samples) の距離行列。",
        },
        "test_cases": test_cases,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    print(f"\nテストデータを {output_path} に保存しました。")
    print(f"合計 {len(test_cases)} 個のテストケースを生成しました。")


if __name__ == "__main__":
    main()
