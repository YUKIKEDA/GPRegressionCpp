#!/usr/bin/env python3
"""
Python (sklearn) と C++ (gp_regression_test) の GPR 結果を比較するプロットを生成するスクリプト

gen_gpr_test_data.py で生成した期待値（gpr_test_data.json）と、
gp_regression_test が cpp-test-result に保存した結果を左右に並べて比較します。

必要なパッケージ:
    pip install numpy matplotlib

使用方法:
    python scripts/compare_gpr_result.py

入力:
    tests/regressor/data/gpr_test_data.json  (Python/sklearn の期待値)
    tests/regressor/cpp-test-result/*.json   (C++ の予測結果)

出力:
    tests/regressor/comparison/*.png  (比較画像)
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def get_paths():
    """入力・出力パスを取得"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "tests" / "regressor" / "data"
    cpp_result_dir = project_root / "tests" / "regressor" / "cpp-test-result"
    out_dir = project_root / "tests" / "regressor" / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "gpr_test_data.json", cpp_result_dir, out_dir


def to_array(j):
    """JSON のネストしたリストを numpy 配列に"""
    return np.array(j)


def plot_single_panel(ax, X, y, X_test, mean, std, title, color='C0'):
    """
    1つのサブプロットに GPR の結果を描画（gen_gpr_test_data.plot_result と同様のロジック）
    """
    X = np.atleast_2d(X) if X.ndim == 1 else X
    y = np.asarray(y)
    mean = np.asarray(mean)
    std = np.asarray(std)
    X_test = np.atleast_2d(X_test) if np.asarray(X_test).ndim == 1 else np.asarray(X_test)

    if X.shape[1] == 1:
        if y.ndim > 1 and y.shape[1] > 1:
            y_plot = y[:, 0]
            mean_plot = mean[:, 0]
            std_plot = std[:, 0]
        else:
            y_plot = np.asarray(y).flatten()
            mean_plot = mean.flatten()
            std_plot = std.flatten()

        idx = np.argsort(X_test.flatten())
        X_sorted = X_test.flatten()[idx]
        mean_sorted = mean_plot[idx]
        std_sorted = std_plot[idx]

        ax.plot(X_sorted, mean_sorted, color=color, linestyle='-', label='Prediction')
        ax.fill_between(
            X_sorted,
            mean_sorted - 1.96 * std_sorted,
            mean_sorted + 1.96 * std_sorted,
            alpha=0.2, color=color, label='95% conf'
        )
        ax.scatter(X.flatten(), y_plot, c='red', s=50, zorder=10, edgecolors='black', label='Train')

    elif X.shape[1] == 2:
        if y.ndim > 1 and y.shape[1] > 1:
            mean_plot = mean[:, 0]
            std_plot = std[:, 0]
            y_plot = y[:, 0]
        else:
            mean_plot = mean.flatten()
            std_plot = std.flatten()
            y_plot = np.asarray(y).flatten()
        idx = np.argsort(X_test[:, 0])
        x1_sorted = X_test[:, 0][idx]
        mean_sorted = mean_plot[idx]
        std_sorted = std_plot[idx]
        ax.fill_between(
            x1_sorted,
            mean_sorted - 1.96 * std_sorted,
            mean_sorted + 1.96 * std_sorted,
            alpha=0.2, color=color, label='95% conf'
        )
        ax.plot(x1_sorted, mean_sorted, color=color, linestyle='-', label='Prediction')
        ax.scatter(X[:, 0], y_plot, c='red', s=50, zorder=10, edgecolors='black', label='Train')
        ax.set_xlabel('Feature 1')

    else:
        if y.ndim > 1 and y.shape[1] > 1:
            mean_plot = mean[:, 0]
            std_plot = std[:, 0]
            y_plot = y[:, 0]
        else:
            mean_plot = mean.flatten()
            std_plot = std.flatten()
            y_plot = np.asarray(y).flatten()
        idx = np.argsort(X_test[:, 0])
        x1_sorted = X_test[:, 0][idx]
        mean_sorted = mean_plot[idx]
        std_sorted = std_plot[idx]
        ax.fill_between(
            x1_sorted,
            mean_sorted - 1.96 * std_sorted,
            mean_sorted + 1.96 * std_sorted,
            alpha=0.2, color=color, label='95% conf'
        )
        ax.plot(x1_sorted, mean_sorted, color=color, linestyle='-', label='Prediction')
        ax.scatter(X[:, 0], y_plot, c='red', s=20, alpha=0.6, zorder=10, label='Train')
        ax.set_xlabel('Feature 1')

    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def load_python_case(test_data, name):
    """gpr_test_data.json から指定名のテストケースを取得（Python/sklearn の期待値）"""
    for tc in test_data.get("test_cases", []):
        if tc.get("name") == name:
            data = tc["data"]
            expected = tc["expected"]
            return {
                "name": name,
                "description": tc.get("description", ""),
                "X_train": to_array(data["X_train"]),
                "y_train": to_array(data["y_train"]),
                "X_test": to_array(data["X_test"]),
                "mean": to_array(expected["mean"]),
                "std": to_array(expected["std"]),
            }
    return None


def load_cpp_case(cpp_result_dir, name):
    """cpp-test-result から指定名の JSON を読み込み"""
    path = cpp_result_dir / f"{name}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {
        "name": d.get("name", name),
        "description": d.get("description", ""),
        "X_train": to_array(d["X_train"]),
        "y_train": to_array(d["y_train"]),
        "X_test": to_array(d["X_test"]),
        "mean": to_array(d["mean"]),
        "std": to_array(d["std"]),
    }


def _y_range_for_case(X, y, mean, std):
    """1ケース分のデータから Y 軸に含める範囲（訓練点・予測±95%信頼）を計算"""
    y_arr = np.asarray(y)
    mean_arr = np.asarray(mean)
    std_arr = np.asarray(std)
    y_plot = y_arr[:, 0] if y_arr.ndim > 1 and y_arr.shape[1] > 1 else y_arr.flatten()
    mean_plot = mean_arr[:, 0] if mean_arr.ndim > 1 and mean_arr.shape[1] > 1 else mean_arr.flatten()
    std_plot = std_arr[:, 0] if std_arr.ndim > 1 and std_arr.shape[1] > 1 else std_arr.flatten()
    low = min(np.min(y_plot), np.min(mean_plot - 1.96 * std_plot))
    high = max(np.max(y_plot), np.max(mean_plot + 1.96 * std_plot))
    return low, high


def compare_one_case(py_case, cpp_case, out_dir):
    """1ケース分の比較図を描画・保存"""
    if py_case is None or cpp_case is None:
        return False

    name = py_case["name"]
    fig, (ax_py, ax_cpp) = plt.subplots(1, 2, figsize=(12, 5))

    plot_single_panel(
        ax_py,
        py_case["X_train"],
        py_case["y_train"],
        py_case["X_test"],
        py_case["mean"],
        py_case["std"],
        f"Python (sklearn)\n{py_case['description']}",
        color='C0',
    )
    plot_single_panel(
        ax_cpp,
        cpp_case["X_train"],
        cpp_case["y_train"],
        cpp_case["X_test"],
        cpp_case["mean"],
        cpp_case["std"],
        f"C++ (this project)\n{cpp_case['description']}",
        color='C1',
    )

    # 左右で Y 軸スケールを揃える（両方のデータ範囲を包含）
    py_lo, py_hi = _y_range_for_case(
        py_case["X_train"], py_case["y_train"],
        py_case["mean"], py_case["std"]
    )
    cpp_lo, cpp_hi = _y_range_for_case(
        cpp_case["X_train"], cpp_case["y_train"],
        cpp_case["mean"], cpp_case["std"]
    )
    y_min = min(py_lo, cpp_lo)
    y_max = max(py_hi, cpp_hi)
    margin = (y_max - y_min) * 0.05
    if margin == 0:
        margin = 1.0
    ax_py.set_ylim(y_min - margin, y_max + margin)
    ax_cpp.set_ylim(y_min - margin, y_max + margin)

    fig.suptitle(f"GPR comparison: {name}", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / f"compare_{name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return True


def main():
    json_path, cpp_result_dir, out_dir = get_paths()

    if not json_path.exists():
        print(f"Error: Test data not found: {json_path}")
        print("Run: python scripts/gen_gpr_test_data.py")
        return 1

    if not cpp_result_dir.exists():
        print(f"Error: C++ results not found: {cpp_result_dir}")
        print("Run the gp_regression_test first so that it writes JSON to cpp-test-result/")
        return 1

    with open(json_path, encoding="utf-8") as f:
        test_data = json.load(f)

    # C++ 側に存在するケース名を基準に比較
    cpp_names = sorted(p.stem for p in cpp_result_dir.glob("*.json"))
    if not cpp_names:
        print("No JSON files in cpp-test-result/")
        return 1

    print("Generating comparison plots...")
    for name in cpp_names:
        py_case = load_python_case(test_data, name)
        cpp_case = load_cpp_case(cpp_result_dir, name)
        if not compare_one_case(py_case, cpp_case, out_dir):
            print(f"  Skip {name}: missing Python or C++ data")

    print(f"Done. Output directory: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
