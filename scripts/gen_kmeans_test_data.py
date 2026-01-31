#!/usr/bin/env python3
"""
K-means テストデータ生成スクリプト

C++ で実装した K-means クラスタリングの検証のため、
scikit-learn の KMeans で期待値を計算し、JSON 形式で出力します。

データ行列 X の形状（scikit-learn と同じ convention）:
    X は (n_samples, n_features)
    - 行 (1次元目) = サンプル数 (n_samples)
    - 列 (2次元目) = 説明変数・特徴量の数 (n_features)

必要なパッケージ:
    pip install numpy scikit-learn

使用方法:
    python scripts/gen_kmeans_test_data.py

出力:
    tests/clustering/k-means/data/kmeans_test_data.json
"""

import json
import numpy as np
from pathlib import Path

from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs


def get_output_path():
    """出力先ディレクトリとファイルパスを取得"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "tests" / "clustering" / "k-means" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "kmeans_test_data.json"


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
    
    # 明確に分離されたクラスタを生成
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.5,
        random_state=42
    )

    kmeans = SklearnKMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        tol=1e-4,
        random_state=42
    )
    kmeans.fit(X)
    
    # transform と predict も実行
    X_transformed = kmeans.transform(X)
    labels_pred = kmeans.predict(X)

    return {
        "name": "case1_basic",
        "description": "n_samples=150, n_features=2, n_clusters=3, well-separated blobs",
        "config": {
            "n_clusters": n_clusters,
            "max_iter": 300,
            "tol": 1e-4,
            "random_state": 42,
        },
        "data": {
            "X": X,
            "true_labels": true_labels,
        },
        "expected": {
            "cluster_centers": kmeans.cluster_centers_,
            "labels": kmeans.labels_,
            "inertia": float(kmeans.inertia_),
            "n_iter": int(kmeans.n_iter_),
            "X_transformed": X_transformed,
            "labels_pred": labels_pred,
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
    
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=123
    )

    kmeans = SklearnKMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        tol=1e-4,
        random_state=123
    )
    kmeans.fit(X)
    
    X_transformed = kmeans.transform(X)
    labels_pred = kmeans.predict(X)

    return {
        "name": "case2_high_dimensional",
        "description": "n_samples=100, n_features=5, n_clusters=4",
        "config": {
            "n_clusters": n_clusters,
            "max_iter": 300,
            "tol": 1e-4,
            "random_state": 123,
        },
        "data": {
            "X": X,
            "true_labels": true_labels,
        },
        "expected": {
            "cluster_centers": kmeans.cluster_centers_,
            "labels": kmeans.labels_,
            "inertia": float(kmeans.inertia_),
            "n_iter": int(kmeans.n_iter_),
            "X_transformed": X_transformed,
            "labels_pred": labels_pred,
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
    
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.8,
        random_state=7
    )

    kmeans = SklearnKMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        tol=1e-4,
        random_state=7
    )
    kmeans.fit(X)
    
    X_transformed = kmeans.transform(X)
    labels_pred = kmeans.predict(X)

    return {
        "name": "case3_two_clusters",
        "description": "n_samples=80, n_features=3, n_clusters=2",
        "config": {
            "n_clusters": n_clusters,
            "max_iter": 300,
            "tol": 1e-4,
            "random_state": 7,
        },
        "data": {
            "X": X,
            "true_labels": true_labels,
        },
        "expected": {
            "cluster_centers": kmeans.cluster_centers_,
            "labels": kmeans.labels_,
            "inertia": float(kmeans.inertia_),
            "n_iter": int(kmeans.n_iter_),
            "X_transformed": X_transformed,
            "labels_pred": labels_pred,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def run_case_4_many_clusters():
    """ケース4: 多数のクラスタ（n_clusters=8）
    初期中心を X の先頭 n_clusters 行に固定し、Python/C++ で同一結果になるようにする。
    """
    print("Generating Case 4 (many clusters)...")
    n_samples, n_features = 200, 4
    n_clusters = 8
    
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.6,
        random_state=0
    )

    # 初期中心を先頭 n_clusters 行に固定（RNG に依存しない）
    init_centers = X[:n_clusters].copy()
    kmeans = SklearnKMeans(
        n_clusters=n_clusters,
        init=init_centers,
        max_iter=300,
        tol=1e-4,
        random_state=0
    )
    kmeans.fit(X)
    
    X_transformed = kmeans.transform(X)
    labels_pred = kmeans.predict(X)

    return {
        "name": "case4_many_clusters",
        "description": "n_samples=200, n_features=4, n_clusters=8",
        "config": {
            "n_clusters": n_clusters,
            "max_iter": 300,
            "tol": 1e-4,
            "random_state": 0,
            "init_centers": init_centers,
        },
        "data": {
            "X": X,
            "true_labels": true_labels,
        },
        "expected": {
            "cluster_centers": kmeans.cluster_centers_,
            "labels": kmeans.labels_,
            "inertia": float(kmeans.inertia_),
            "n_iter": int(kmeans.n_iter_),
            "X_transformed": X_transformed,
            "labels_pred": labels_pred,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def run_case_5_overlapping_clusters():
    """ケース5: 重なりのあるクラスタ（cluster_std が大きい）"""
    print("Generating Case 5 (overlapping clusters)...")
    n_samples, n_features = 120, 2
    n_clusters = 3
    
    # 標準偏差を大きくして重なりを作る
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=2.0,
        random_state=999
    )

    kmeans = SklearnKMeans(
        n_clusters=n_clusters,
        init='k-means++',
        max_iter=300,
        tol=1e-4,
        random_state=999
    )
    kmeans.fit(X)
    
    X_transformed = kmeans.transform(X)
    labels_pred = kmeans.predict(X)

    return {
        "name": "case5_overlapping_clusters",
        "description": "n_samples=120, n_features=2, n_clusters=3, overlapping (cluster_std=2.0)",
        "config": {
            "n_clusters": n_clusters,
            "max_iter": 300,
            "tol": 1e-4,
            "random_state": 999,
        },
        "data": {
            "X": X,
            "true_labels": true_labels,
        },
        "expected": {
            "cluster_centers": kmeans.cluster_centers_,
            "labels": kmeans.labels_,
            "inertia": float(kmeans.inertia_),
            "n_iter": int(kmeans.n_iter_),
            "X_transformed": X_transformed,
            "labels_pred": labels_pred,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def run_case_6_fit_predict():
    """ケース6: fit_predict の一致確認用（他ケースと同じ expected 形式・init で決定的）"""
    print("Generating Case 6 (fit_predict)...")
    n_samples, n_features = 90, 3
    n_clusters = 5

    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=0.7,
        random_state=555
    )

    init_centers = X[:n_clusters].copy()
    kmeans = SklearnKMeans(
        n_clusters=n_clusters,
        init=init_centers,
        max_iter=300,
        tol=1e-4,
        random_state=555
    )
    labels_fit_predict = kmeans.fit_predict(X)

    X_transformed = kmeans.transform(X)
    labels_pred = kmeans.predict(X)

    return {
        "name": "case6_fit_predict",
        "description": "fit_predict 期待値: n_samples=90, n_features=3, n_clusters=5",
        "config": {
            "n_clusters": n_clusters,
            "max_iter": 300,
            "tol": 1e-4,
            "random_state": 555,
            "init_centers": init_centers,
        },
        "data": {
            "X": X,
            "true_labels": true_labels,
        },
        "expected": {
            "cluster_centers": kmeans.cluster_centers_,
            "labels": labels_fit_predict,
            "inertia": float(kmeans.inertia_),
            "n_iter": int(kmeans.n_iter_),
            "X_transformed": X_transformed,
            "labels_pred": labels_pred,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_clusters": n_clusters,
        },
    }


def main():
    """メイン関数"""
    print("K-means テストデータ生成を開始...")
    output_path = get_output_path()

    run_fns = [
        run_case_1_basic,
        run_case_2_high_dimensional,
        run_case_3_two_clusters,
        run_case_4_many_clusters,
        run_case_5_overlapping_clusters,
        run_case_6_fit_predict,
    ]
    test_cases = []
    for run_fn in run_fns:
        tc = run_fn()
        test_cases.append(tc)
        n, d, c = tc["meta"]["n_samples"], tc["meta"]["n_features"], tc["meta"]["n_clusters"]
        print(f"  → {tc['name']}: X ({n}x{d}), n_clusters={c}")

    output_data = {
        "description": "K-means クラスタリングのテストデータ（scikit-learn 期待値）",
        "generated_by": "gen_kmeans_test_data.py",
        "data_layout": {
            "X_rows": "n_samples",
            "X_cols": "n_features",
            "note": "データ行列 X は 行=サンプル数, 列=説明変数(特徴量)数。scikit-learn と同じ (n_samples, n_features)。",
        },
        "test_cases": test_cases,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    print(f"\nテストデータを {output_path} に保存しました。")
    print(f"合計 {len(test_cases)} 個のテストケースを生成しました。")


if __name__ == "__main__":
    main()
