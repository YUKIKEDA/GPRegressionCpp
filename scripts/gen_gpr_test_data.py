#!/usr/bin/env python3
"""
ガウス過程回帰（GPR）のテストデータ生成スクリプト

このスクリプトは、C++で実装したガウス過程回帰の検証のための
テストデータを生成します。sklearn.gaussian_process.GaussianProcessRegressor
を使用して期待値を計算し、JSON形式で出力します。

必要なパッケージ:
    pip install numpy scipy scikit-learn matplotlib

使用方法:
    python scripts/gen_gpr_test_data.py

出力:
    tests/regressor/data/gpr_test_data.json
    tests/regressor/images/*.png（確認用プロット）
"""

import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUIなしで画像保存
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as SkConstantKernel
from scipy.optimize import dual_annealing, fmin_l_bfgs_b


def get_output_paths():
    """出力先ディレクトリとファイルパスを取得"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "tests" / "regressor" / "data"
    image_dir = project_root / "tests" / "regressor" / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "gpr_test_data.json", image_dir


OUTPUT_JSON, IMAGE_DIR = get_output_paths()


class NumpyEncoder(json.JSONEncoder):
    """Numpy配列をJSONで扱えるリストに変換するエンコーダ"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        return super().default(obj)


def plot_result(X, y, X_test, y_mean, y_std, title, filename):
    """結果を画像として保存（確認用）"""
    plt.figure(figsize=(10, 6))

    # 1次元入力の場合のみプロット
    if X.shape[1] == 1:
        # ターゲットが多次元の場合は最初の次元のみプロット
        if y.ndim > 1 and y.shape[1] > 1:
            y_plot = y[:, 0]
            mean_plot = y_mean[:, 0]
            std_plot = y_std[:, 0]
            title += " (Target 0)"
        else:
            y_plot = y.flatten()
            mean_plot = y_mean.flatten()
            std_plot = y_std.flatten()

        # ソート（プロットのため）
        idx = np.argsort(X_test.flatten())
        X_test_sorted = X_test.flatten()[idx]
        mean_sorted = mean_plot[idx]
        std_sorted = std_plot[idx]

        plt.plot(X_test_sorted, mean_sorted, 'b-', label='Prediction')
        plt.fill_between(
            X_test_sorted,
            mean_sorted - 1.96 * std_sorted,
            mean_sorted + 1.96 * std_sorted,
            alpha=0.2, color='b', label='95% conf interval'
        )
        plt.scatter(X, y_plot, c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label='Observations')

    # 2次元入力(ARD)の場合（Feature 1 に投影して表示）
    elif X.shape[1] == 2:
        # x1 でソートしてから描画（信頼区間の帯がつながるように）
        idx = np.argsort(X_test[:, 0])
        x1_sorted = X_test[:, 0][idx]
        mean_sorted = y_mean.flatten()[idx]
        std_sorted = y_std.flatten()[idx]
        plt.fill_between(
            x1_sorted,
            mean_sorted - 1.96 * std_sorted,
            mean_sorted + 1.96 * std_sorted,
            alpha=0.2, color='b', label='95% conf interval'
        )
        plt.plot(x1_sorted, mean_sorted, 'b-', label='Prediction')
        plt.scatter(X[:, 0], y.flatten(), c='r', s=50, zorder=10, edgecolors=(0, 0, 0), label='Train')
        plt.xlabel("Feature 1")

    # 高次元入力（D > 2）の場合：Feature 1 に投影して表示
    else:
        idx = np.argsort(X_test[:, 0])
        x1_sorted = X_test[:, 0][idx]
        mean_sorted = y_mean.flatten()[idx]
        std_sorted = y_std.flatten()[idx]
        plt.fill_between(
            x1_sorted,
            mean_sorted - 1.96 * std_sorted,
            mean_sorted + 1.96 * std_sorted,
            alpha=0.2, color='b', label='95% conf interval'
        )
        plt.plot(x1_sorted, mean_sorted, 'b-', label='Prediction')
        plt.scatter(X[:, 0], y.flatten(), c='r', s=20, alpha=0.6, zorder=10, label='Train')
        plt.xlabel("Feature 1")

    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(IMAGE_DIR / filename)
    plt.close()
    print(f"  Saved image: {filename}")


def run_case_1_simple():
    """ケース1: 単純な1次元、正規化なし"""
    print("Generating Case 1...")
    rng = np.random.RandomState(42)
    X_train = np.atleast_2d(np.linspace(0, 10, 40)).T
    y_train = X_train * np.sin(X_train) + rng.normal(0, 0.5, size=X_train.shape)
    y_train = y_train.flatten()

    X_test = np.atleast_2d(np.linspace(-2, 12, 50)).T

    kernel = 1.0 * RBF(length_scale=1.0)
    alpha = 0.5

    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=False, n_restarts_optimizer=0
    )
    t0 = time.perf_counter()
    gpr.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    predict_time = time.perf_counter() - t0

    plot_result(X_train, y_train, X_test, y_mean, y_std, "Case 1: Simple 1D", "case1_simple.png")

    return {
        "name": "case1_simple",
        "description": "1D Input, Scalar Output, No Normalization, RBF Kernel",
        "config": {
            "normalize_y": False,
            "alpha": alpha,
            "kernel": "ConstantKernel * RBF"
        },
        "data": {
            "X_train": X_train,
            "y_train": y_train.reshape(-1, 1),
            "X_test": X_test
        },
        "expected": {
            "mean": y_mean.reshape(-1, 1),
            "std": y_std.reshape(-1, 1),
            "lml": gpr.log_marginal_likelihood(gpr.kernel_.theta),
            "optimized_theta": gpr.kernel_.theta.tolist()
        },
        "fit_time_seconds": float(fit_time),
        "predict_time_seconds": float(predict_time),
    }


def run_case_2_multidim_normalized():
    """ケース2: 多次元入力・多次元出力、正規化あり"""
    print("Generating Case 2...")
    rng = np.random.RandomState(99)
    n_samples, n_features, n_targets = 120, 3, 2

    X_train = rng.uniform(-5, 5, (n_samples, n_features))
    y1 = X_train @ np.array([0.5, -0.2, 1.0]) + rng.normal(0, 0.1, n_samples)
    y2 = np.sum(X_train**2, axis=1) * 0.1 + rng.normal(0, 0.1, n_samples)
    y_train = np.column_stack([y1, y2])

    # プロットで左端まで信頼区間が描けるよう、x1 を等間隔にしてテスト点を多めに用意
    n_test = 80
    x1_grid = np.linspace(-6, 6, n_test)
    X_test = np.column_stack([x1_grid, rng.uniform(-6, 6, (n_test, n_features - 1))])

    kernel = RBF(length_scale=1.5)
    alpha = 1e-10

    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=True,
        n_restarts_optimizer=5, random_state=42
    )
    t0 = time.perf_counter()
    gpr.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_mean = gpr.predict(X_test)
    predict_time = time.perf_counter() - t0

    # ターゲットごとのStdを手動計算（sklearnのvector outputの挙動回避のため）
    K_trans = gpr.kernel_(X_test, gpr.X_train_)
    K_inv = np.linalg.inv(gpr.kernel_(gpr.X_train_) + np.eye(n_samples) * alpha)
    K_test = gpr.kernel_(X_test)
    var_norm = np.diag(K_test - K_trans @ K_inv @ K_trans.T).copy()
    var_norm[var_norm < 0] = 0
    std_norm = np.sqrt(var_norm)

    expected_std = np.zeros((X_test.shape[0], n_targets))
    for j in range(n_targets):
        expected_std[:, j] = std_norm * gpr._y_train_std[j]

    plot_result(
        X_train[:, 0:1], y_train, X_test[:, 0:1], y_mean, expected_std,
        "Case 2: Multi-Output (Target 0, Proj onto Dim 0)", "case2_multidim.png"
    )

    return {
        "name": "case2_multidim",
        "description": "3D Input, 2D Output, Normalize_y=True",
        "config": {
            "normalize_y": True,
            "alpha": alpha,
            "kernel": "RBF"
        },
        "data": {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test
        },
        "expected": {
            "mean": y_mean,
            "std": expected_std,
            "lml": gpr.log_marginal_likelihood(gpr.kernel_.theta),
            "optimized_theta": gpr.kernel_.theta.tolist(),
            "y_train_mean": gpr._y_train_mean.tolist(),
            "y_train_std": gpr._y_train_std.tolist()
        },
        "fit_time_seconds": float(fit_time),
        "predict_time_seconds": float(predict_time),
    }


def run_case_3_ard():
    """ケース3: 異方性カーネル (ARD)"""
    print("Generating Case 3...")
    rng = np.random.RandomState(42)
    X_train = rng.uniform(-5, 5, (50, 2))
    y_train = np.sin(X_train[:, 0]) + rng.normal(0, 0.1, X_train.shape[0])

    X_test = rng.uniform(-6, 6, (50, 2))

    # 無関係な次元の length_scale が上界に張り付くのを防ぐため上界を広くする
    kernel = SkConstantKernel(1.0) * RBF(
        length_scale=[1.0, 1.0],
        length_scale_bounds=(1e-2, 1e8)
    )
    alpha = 0.1

    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=False,
        n_restarts_optimizer=10, random_state=42
    )
    t0 = time.perf_counter()
    gpr.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    predict_time = time.perf_counter() - t0

    plot_result(
        X_train, y_train, X_test, y_mean, y_std,
        "Case 3: ARD (Proj onto Feature 1)", "case3_ard.png"
    )

    return {
        "name": "case3_ard",
        "description": "2D Input (ARD), x2 is irrelevant",
        "config": {
            "normalize_y": False,
            "alpha": alpha,
            "kernel": "Constant * RBF(ard)"
        },
        "data": {
            "X_train": X_train,
            "y_train": y_train.reshape(-1, 1),
            "X_test": X_test
        },
        "expected": {
            "mean": y_mean.reshape(-1, 1),
            "std": y_std.reshape(-1, 1),
            "lml": gpr.log_marginal_likelihood(gpr.kernel_.theta),
            "optimized_theta": gpr.kernel_.theta.tolist()
        },
        "fit_time_seconds": float(fit_time),
        "predict_time_seconds": float(predict_time),
    }


def run_case_4_dual_annealing():
    """ケース4: Dual Annealing (大域最適化)"""
    print("Generating Case 4...")
    rng = np.random.RandomState(1)
    X_train = np.atleast_2d(np.concatenate([
        np.linspace(0, 4, 30), np.linspace(6, 10, 30)
    ])).T
    y_train = X_train * np.sin(X_train) + rng.normal(0, 0.2, (60, 1))
    y_train = y_train.flatten()

    X_test = np.atleast_2d(np.linspace(-1, 11, 100)).T

    kernel = SkConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=0.1, normalize_y=True, optimizer=None
    )
    t0 = time.perf_counter()
    gpr.fit(X_train, y_train)
    def obj_func(theta):
        return -gpr.log_marginal_likelihood(theta, clone_kernel=False)
    res = dual_annealing(obj_func, bounds=gpr.kernel_.bounds, seed=42, maxiter=1000)
    gpr.kernel_.theta = res.x
    gpr.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    predict_time = time.perf_counter() - t0

    plot_result(
        X_train, y_train, X_test, y_mean, y_std,
        "Case 4: Dual Annealing Result", "case4_da.png"
    )

    return {
        "name": "case4_dual_annealing",
        "description": "Global Optimization check using Dual Annealing",
        "config": {
            "normalize_y": True,
            "alpha": 0.1,
            "kernel": "Constant * RBF"
        },
        "data": {
            "X_train": X_train,
            "y_train": y_train.reshape(-1, 1),
            "X_test": X_test
        },
        "expected": {
            "mean": y_mean.reshape(-1, 1),
            "std": y_std.reshape(-1, 1),
            "lml": float(-res.fun),
            "optimized_theta": res.x.tolist()
        },
        "fit_time_seconds": float(fit_time),
        "predict_time_seconds": float(predict_time),
    }


def run_case_5_large_scale():
    """ケース5: 大規模（N=1000, D=32）、ARD＋Scikit-learn標準最適化（L-BFGS-B）"""
    print("Generating Case 5 (Large Scale: N=1000, D=32, ARD + L-BFGS-B)...")
    rng = np.random.RandomState(123)
    n_train = 1000
    n_features = 32
    n_test = 200

    # 最初の数次元だけが有効なターゲット（高次元でスパースな依存 → ARDで検出）
    w = np.zeros(n_features)
    w[0] = 0.8
    w[1] = -0.3
    w[2] = 0.5
    w[3:8] = rng.uniform(-0.2, 0.2, 5)

    X_train = rng.uniform(-3, 3, (n_train, n_features))
    y_train = X_train @ w + 0.1 * np.sin(X_train[:, 0]) + rng.normal(0, 0.15, n_train)

    X_test = rng.uniform(-3, 3, (n_test, n_features))

    kernel = SkConstantKernel(1.0) * RBF(
        length_scale=np.ones(n_features),
        length_scale_bounds=(1e-2, 1e12)
    )
    alpha = 0.1

    n_evals = [0]  # mutable for closure

    def lbfgsb_optimizer_with_counting(obj_func, initial_theta, bounds):
        def wrapped_obj(theta):
            n_evals[0] += 1
            if n_evals[0] == 1 or n_evals[0] % 100 == 0:
                print(f"  [L-BFGS-B] 目的関数評価: {n_evals[0]} 回", flush=True)
            return obj_func(theta, eval_gradient=True)

        theta_opt, func_min, _ = fmin_l_bfgs_b(
            wrapped_obj, initial_theta, bounds=bounds, approx_grad=False
        )
        return theta_opt, func_min

    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=True,
        optimizer=lbfgsb_optimizer_with_counting,
        n_restarts_optimizer=5, random_state=42,
    )
    t0 = time.perf_counter()
    gpr.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    predict_time = time.perf_counter() - t0

    lml = gpr.log_marginal_likelihood(gpr.kernel_.theta)

    plot_result(
        X_train, y_train, X_test, y_mean, y_std,
        f"Case 5: Large Scale (sklearn default) (N={n_train}, D={n_features})", "case5_large_scale.png"
    )

    return {
        "name": "case5_large_scale",
        "description": f"Large scale sklearn default: N={n_train}, D={n_features}, scalar output",
        "config": {
            "normalize_y": True,
            "alpha": alpha,
            "kernel": "Constant * RBF(ARD)",
            "optimizer": "fmin_l_bfgs_b",
            "n_restarts_optimizer": 5,
            "n_train": n_train,
            "n_features": n_features,
            "n_test": n_test,
        },
        "data": {
            "X_train": X_train,
            "y_train": y_train.reshape(-1, 1),
            "X_test": X_test
        },
        "expected": {
            "mean": y_mean.reshape(-1, 1),
            "std": y_std.reshape(-1, 1),
            "lml": float(lml),
            "optimized_theta": gpr.kernel_.theta.tolist(),
            "y_train_mean": gpr._y_train_mean.tolist(),
            "y_train_std": gpr._y_train_std.tolist()
        },
        "fit_time_seconds": float(fit_time),
        "predict_time_seconds": float(predict_time),
    }


def run_case_6_large_scale_lbfgsb():
    """ケース6: 大規模（N=1000, D=32）、ARD＋Dual Annealing で最適化"""
    print("Generating Case 6 (Large Scale: N=1000, D=32, ARD + Dual Annealing)...")
    rng = np.random.RandomState(123)
    n_train = 1000
    n_features = 32
    n_test = 200

    # Case5と同じデータ生成
    w = np.zeros(n_features)
    w[0] = 0.8
    w[1] = -0.3
    w[2] = 0.5
    w[3:8] = rng.uniform(-0.2, 0.2, 5)

    X_train = rng.uniform(-3, 3, (n_train, n_features))
    y_train = X_train @ w + 0.1 * np.sin(X_train[:, 0]) + rng.normal(0, 0.15, n_train)

    X_test = rng.uniform(-3, 3, (n_test, n_features))

    # ARD: 次元ごとの length_scale（無関係な次元は上界に張り付くのを防ぐため上界を十分広く）
    kernel = SkConstantKernel(1.0) * RBF(
        length_scale=np.ones(n_features),
        length_scale_bounds=(1e-2, 1e12)
    )
    alpha = 0.1

    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=True, optimizer=None
    )
    t0 = time.perf_counter()
    gpr.fit(X_train, y_train)

    n_evals = [0]  # mutable for closure

    def obj_func(theta):
        n_evals[0] += 1
        if n_evals[0] == 1 or n_evals[0] % 100 == 0:
            print(f"  [Dual Annealing] 目的関数評価: {n_evals[0]} 回", flush=True)
        return -gpr.log_marginal_likelihood(theta, clone_kernel=False)

    def da_callback(x, f, context):
        da_callback.n_found = getattr(da_callback, "n_found", 0) + 1
        ctx_str = ("annealing", "local search", "dual")[context]
        print(
            f"  [Dual Annealing] 最小値 #{da_callback.n_found}: -LML={f:.2f} ({ctx_str})",
            flush=True,
        )

    res = dual_annealing(
        obj_func,
        bounds=gpr.kernel_.bounds,
        seed=42,
        maxiter=2000,
        callback=da_callback,
    )
    gpr.kernel_.theta = res.x
    gpr.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    predict_time = time.perf_counter() - t0

    plot_result(
        X_train, y_train, X_test, y_mean, y_std,
        f"Case 6: Large Scale (N={n_train}, D={n_features})", "case6_large_scale_lbfgsb.png"
    )

    return {
        "name": "case6_large_scale_lbfgsb",
        "description": f"Large scale: N={n_train}, D={n_features}, scalar output",
        "config": {
            "normalize_y": True,
            "alpha": alpha,
            "kernel": "Constant * RBF(ARD)",
            "optimizer": "Dual Annealing",
            "n_train": n_train,
            "n_features": n_features,
            "n_test": n_test,
        },
        "data": {
            "X_train": X_train,
            "y_train": y_train.reshape(-1, 1),
            "X_test": X_test
        },
        "expected": {
            "mean": y_mean.reshape(-1, 1),
            "std": y_std.reshape(-1, 1),
            "lml": float(-res.fun),
            "optimized_theta": res.x.tolist(),
            "y_train_mean": gpr._y_train_mean.tolist(),
            "y_train_std": gpr._y_train_std.tolist()
        },
        "fit_time_seconds": float(fit_time),
        "predict_time_seconds": float(predict_time),
    }


def main():
    """メイン関数"""
    print("ガウス過程回帰テストデータ生成を開始...")

    run_fns = [
        run_case_1_simple,
        run_case_2_multidim_normalized,
        run_case_3_ard,
        run_case_4_dual_annealing,
        # run_case_5_large_scale,
        # run_case_6_large_scale_lbfgsb,
    ]
    total_start = time.perf_counter()
    test_cases = []

    for run_fn in run_fns:
        start = time.perf_counter()
        tc = run_fn()
        tc["execution_time_seconds"] = float(time.perf_counter() - start)
        test_cases.append(tc)
        print(f"  → 学習: {tc['fit_time_seconds']:.2f}秒, 予測: {tc['predict_time_seconds']:.2f}秒 (計 {tc['execution_time_seconds']:.2f}秒)")

    total_elapsed = time.perf_counter() - total_start

    execution_times = [tc["execution_time_seconds"] for tc in test_cases]
    fit_times = [tc["fit_time_seconds"] for tc in test_cases]
    predict_times = [tc["predict_time_seconds"] for tc in test_cases]
    n = len(test_cases)
    output_data = {
        "description": "ガウス過程回帰（GPR）のテストデータ",
        "generated_by": "gen_gpr_test_data.py",
        "generation_time_seconds": float(total_elapsed),
        "statistics": {
            "total_test_cases": n,
            "total_execution_time_seconds": float(sum(execution_times)),
            "average_execution_time_seconds": float(sum(execution_times) / n),
            "min_execution_time_seconds": float(min(execution_times)),
            "max_execution_time_seconds": float(max(execution_times)),
            "total_fit_time_seconds": float(sum(fit_times)),
            "average_fit_time_seconds": float(sum(fit_times) / n),
            "min_fit_time_seconds": float(min(fit_times)),
            "max_fit_time_seconds": float(max(fit_times)),
            "total_predict_time_seconds": float(sum(predict_times)),
            "average_predict_time_seconds": float(sum(predict_times) / n),
            "min_predict_time_seconds": float(min(predict_times)),
            "max_predict_time_seconds": float(max(predict_times)),
        },
        "test_cases": test_cases,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_data, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    print(f"\nテストデータを {OUTPUT_JSON} に保存しました。")
    print(f"合計 {len(test_cases)} 個のテストケースを生成しました。")
    print(f"総実行時間: {total_elapsed:.2f}秒")
    print(f"  学習時間: 合計 {sum(fit_times):.2f}秒 / 平均 {sum(fit_times)/n:.2f}秒 (最小 {min(fit_times):.2f} / 最大 {max(fit_times):.2f})")
    print(f"  予測時間: 合計 {sum(predict_times):.2f}秒 / 平均 {sum(predict_times)/n:.2f}秒 (最小 {min(predict_times):.2f} / 最大 {max(predict_times):.2f})")
    print(f"画像は {IMAGE_DIR} に保存しました。")


if __name__ == "__main__":
    main()
