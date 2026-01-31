#!/usr/bin/env python3
"""
PCA テストデータ生成スクリプト

C++ で実装した PCA（SVD ベース）の検証のため、
scikit-learn の PCA で期待値を計算し、JSON 形式で出力します。

データ行列 X の形状（scikit-learn と同じ convention）:
    X は (n_samples, n_features)
    - 行 (1次元目) = サンプル数 (n_samples)
    - 列 (2次元目) = 説明変数・特徴量の数 (n_features)
    正方行列でも「行=サンプル、列=特徴量」で解釈すること。

必要なパッケージ:
    pip install numpy scikit-learn

使用方法:
    python scripts/gen_pca_test_data.py

出力:
    tests/decomposition/PCA/data/pca_test_data.json
"""

import json
import numpy as np
from pathlib import Path

from sklearn.decomposition import PCA as SklearnPCA


def get_output_path():
    """出力先ディレクトリとファイルパスを取得"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "tests" / "decomposition" / "PCA" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "pca_test_data.json"


class NumpyEncoder(json.JSONEncoder):
    """NumPy 配列・スカラーを JSON で扱える型に変換するエンコーダ"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return super().default(obj)


def run_case_1_basic():
    """ケース1: 基本（n_samples > n_features, n_components=2, whiten=False）"""
    print("Generating Case 1 (basic)...")
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 5
    X = rng.standard_normal((n_samples, n_features))

    pca = SklearnPCA(n_components=2, whiten=False, random_state=42)
    pca.fit(X)
    X_transformed = pca.transform(X)

    return {
        "name": "case1_basic",
        "description": "n_samples=20, n_features=5, n_components=2, whiten=False",
        "config": {
            "n_components": 2,
            "whiten": False,
        },
        "data": {"X": X},
        "expected": {
            "mean": pca.mean_,
            "components": pca.components_,
            "singular_values": pca.singular_values_,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "noise_variance": float(pca.noise_variance_),
            "X_transformed": X_transformed,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": int(pca.n_components_),
        },
    }


def run_case_2_whiten():
    """ケース2: 白化あり（同上データで whiten=True）"""
    print("Generating Case 2 (whiten=True)...")
    rng = np.random.RandomState(42)
    n_samples, n_features = 20, 5
    X = rng.standard_normal((n_samples, n_features))

    pca = SklearnPCA(n_components=2, whiten=True, random_state=42)
    pca.fit(X)
    X_transformed = pca.transform(X)

    return {
        "name": "case2_whiten",
        "description": "n_samples=20, n_features=5, n_components=2, whiten=True",
        "config": {
            "n_components": 2,
            "whiten": True,
        },
        "data": {"X": X},
        "expected": {
            "mean": pca.mean_,
            "components": pca.components_,
            "singular_values": pca.singular_values_,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "noise_variance": float(pca.noise_variance_),
            "X_transformed": X_transformed,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": int(pca.n_components_),
        },
    }


def run_case_3_all_components():
    """ケース3: 全成分（n_components=None → min(n_samples, n_features)）"""
    print("Generating Case 3 (all components)...")
    rng = np.random.RandomState(123)
    n_samples, n_features = 15, 4
    X = rng.standard_normal((n_samples, n_features))

    # sklearn: n_components=None は min(n_samples, n_features) を保持
    pca = SklearnPCA(n_components=None, whiten=False, random_state=42)
    pca.fit(X)
    X_transformed = pca.transform(X)

    # JSON に null を出さないため、実際に使った n_components を config に書く
    n_comp = int(pca.n_components_)

    return {
        "name": "case3_all_components",
        "description": "n_components=None (all), n_samples=15, n_features=4",
        "config": {
            "n_components": n_comp,
            "whiten": False,
        },
        "data": {"X": X},
        "expected": {
            "mean": pca.mean_,
            "components": pca.components_,
            "singular_values": pca.singular_values_,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "noise_variance": float(pca.noise_variance_),
            "X_transformed": X_transformed,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": int(pca.n_components_),
        },
    }


def run_case_4_more_features_than_samples():
    """ケース4: 特徴量 > サンプル数（n_features > n_samples）"""
    print("Generating Case 4 (n_features > n_samples)...")
    rng = np.random.RandomState(7)
    n_samples, n_features = 8, 12
    X = rng.standard_normal((n_samples, n_features))

    pca = SklearnPCA(n_components=3, whiten=False, random_state=42)
    pca.fit(X)
    X_transformed = pca.transform(X)

    return {
        "name": "case4_more_features_than_samples",
        "description": "n_samples=8, n_features=12, n_components=3",
        "config": {
            "n_components": 3,
            "whiten": False,
        },
        "data": {"X": X},
        "expected": {
            "mean": pca.mean_,
            "components": pca.components_,
            "singular_values": pca.singular_values_,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "noise_variance": float(pca.noise_variance_),
            "X_transformed": X_transformed,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": int(pca.n_components_),
        },
    }


def run_case_5_square_and_single_component():
    """ケース5: 正方に近い行列、n_components=1"""
    print("Generating Case 5 (square, n_components=1)...")
    rng = np.random.RandomState(0)
    n_samples, n_features = 10, 10
    X = rng.standard_normal((n_samples, n_features))

    pca = SklearnPCA(n_components=1, whiten=False, random_state=42)
    pca.fit(X)
    X_transformed = pca.transform(X)

    return {
        "name": "case5_single_component",
        "description": "n_samples=10, n_features=10, n_components=1",
        "config": {
            "n_components": 1,
            "whiten": False,
        },
        "data": {"X": X},
        "expected": {
            "mean": pca.mean_,
            "components": pca.components_,
            "singular_values": pca.singular_values_,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "noise_variance": float(pca.noise_variance_),
            "X_transformed": X_transformed,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": int(pca.n_components_),
        },
    }


def run_case_6_deterministic_fit_transform():
    """ケース6: fit_transform の一致確認用（小さい整数に近いデータ）"""
    print("Generating Case 6 (deterministic fit_transform)...")
    rng = np.random.RandomState(999)
    n_samples, n_features = 12, 6
    X = np.round(rng.standard_normal((n_samples, n_features)) * 10) / 10.0

    pca = SklearnPCA(n_components=4, whiten=False, random_state=42)
    X_transformed = pca.fit_transform(X)

    return {
        "name": "case6_fit_transform",
        "description": "fit_transform 期待値: n_samples=12, n_features=6, n_components=4",
        "config": {
            "n_components": 4,
            "whiten": False,
        },
        "data": {"X": X},
        "expected": {
            "mean": pca.mean_,
            "components": pca.components_,
            "singular_values": pca.singular_values_,
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "noise_variance": float(pca.noise_variance_),
            "X_transformed": X_transformed,
        },
        "meta": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_components": int(pca.n_components_),
        },
    }


def main():
    """メイン関数"""
    print("PCA テストデータ生成を開始...")
    output_path = get_output_path()

    run_fns = [
        run_case_1_basic,
        run_case_2_whiten,
        run_case_3_all_components,
        run_case_4_more_features_than_samples,
        run_case_5_square_and_single_component,
        run_case_6_deterministic_fit_transform,
    ]
    test_cases = []
    for run_fn in run_fns:
        tc = run_fn()
        test_cases.append(tc)
        n, d, c = tc["meta"]["n_samples"], tc["meta"]["n_features"], tc["meta"]["n_components"]
        print(f"  → {tc['name']}: X ({n}x{d}), n_components={c}")

    output_data = {
        "description": "PCA（SVD ベース）のテストデータ（scikit-learn 期待値）",
        "generated_by": "gen_pca_test_data.py",
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
