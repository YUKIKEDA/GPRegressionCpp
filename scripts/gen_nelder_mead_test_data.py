#!/usr/bin/env python3
"""
Nelder-Meadアルゴリズムのテストデータ生成スクリプト

このスクリプトは、C++で実装したNelder-Meadアルゴリズムの検証のための
テストデータを生成します。scipy.optimize.minimizeのNelder-Mead実装を
使用して期待値を計算し、JSON形式で出力します。
"""

import numpy as np
import json
import time
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class TestFunction:
    """テスト関数の基底クラス"""
    
    def __init__(self, name: str, dimension: int, 
                 optimal_point: np.ndarray, optimal_value: float,
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        self.name = name
        self.dimension = dimension
        self.optimal_point = optimal_point
        self.optimal_value = optimal_value
        self.bounds = bounds
    
    def __call__(self, x: np.ndarray) -> float:
        """目的関数を評価"""
        raise NotImplementedError


class SphereFunction(TestFunction):
    """Sphere関数: f(x) = sum(x_i^2)"""
    
    def __init__(self, dimension: int = 2):
        optimal_point = np.zeros(dimension)
        optimal_value = 0.0
        super().__init__("Sphere", dimension, optimal_point, optimal_value)
    
    def __call__(self, x: np.ndarray) -> float:
        return np.sum(x ** 2)


class RosenbrockFunction(TestFunction):
    """Rosenbrock関数: f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)"""
    
    def __init__(self, dimension: int = 2):
        optimal_point = np.ones(dimension)
        optimal_value = 0.0
        super().__init__("Rosenbrock", dimension, optimal_point, optimal_value)
    
    def __call__(self, x: np.ndarray) -> float:
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


class AckleyFunction(TestFunction):
    """Ackley関数: 多峰性のテスト関数"""
    
    def __init__(self, dimension: int = 2):
        optimal_point = np.zeros(dimension)
        optimal_value = 0.0
        bounds = (np.full(dimension, -5.0), np.full(dimension, 5.0))
        super().__init__("Ackley", dimension, optimal_point, optimal_value, bounds)
    
    def __call__(self, x: np.ndarray) -> float:
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e


class RastriginFunction(TestFunction):
    """Rastrigin関数: 多峰性のテスト関数"""
    
    def __init__(self, dimension: int = 2):
        optimal_point = np.zeros(dimension)
        optimal_value = 0.0
        bounds = (np.full(dimension, -5.12), np.full(dimension, 5.12))
        super().__init__("Rastrigin", dimension, optimal_point, optimal_value, bounds)
    
    def __call__(self, x: np.ndarray) -> float:
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))


class BealeFunction(TestFunction):
    """Beale関数: 2次元のテスト関数"""
    
    def __init__(self):
        optimal_point = np.array([3.0, 0.5])
        optimal_value = 0.0
        bounds = (np.array([-4.5, -4.5]), np.array([4.5, 4.5]))
        super().__init__("Beale", 2, optimal_point, optimal_value, bounds)
    
    def __call__(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        term1 = (1.5 - x1 + x1*x2)**2
        term2 = (2.25 - x1 + x1*x2**2)**2
        term3 = (2.625 - x1 + x1*x2**3)**2
        return term1 + term2 + term3


class BoothFunction(TestFunction):
    """Booth関数: 2次元のテスト関数"""
    
    def __init__(self):
        optimal_point = np.array([1.0, 3.0])
        optimal_value = 0.0
        bounds = (np.array([-10.0, -10.0]), np.array([10.0, 10.0]))
        super().__init__("Booth", 2, optimal_point, optimal_value, bounds)
    
    def __call__(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2


class MatyasFunction(TestFunction):
    """Matyas関数: 2次元のテスト関数"""
    
    def __init__(self):
        optimal_point = np.array([0.0, 0.0])
        optimal_value = 0.0
        bounds = (np.array([-10.0, -10.0]), np.array([10.0, 10.0]))
        super().__init__("Matyas", 2, optimal_point, optimal_value, bounds)
    
    def __call__(self, x: np.ndarray) -> float:
        x1, x2 = x[0], x[1]
        return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def generate_initial_point(function: TestFunction, seed: Optional[int] = None) -> np.ndarray:
    """初期点を生成"""
    if seed is not None:
        np.random.seed(seed)
    
    if function.bounds is not None:
        lower, upper = function.bounds
        # 境界内のランダムな点を生成
        initial = np.random.uniform(lower, upper)
    else:
        # 境界がない場合、最適解から離れた点を生成
        initial = function.optimal_point + np.random.uniform(-2.0, 2.0, function.dimension)
    
    return initial


def run_scipy_optimization(function: TestFunction, 
                          initial_point: np.ndarray,
                          max_iterations: int = 1000,
                          tolerance: float = 1e-6) -> Tuple[Dict, float]:
    """scipy.optimize.minimizeを使用して最適化を実行
    
    Returns:
        Tuple[Dict, float]: (最適化結果, 実行時間[秒])
    """
    
    options = {
        'maxiter': max_iterations,
        'xatol': tolerance,
        'fatol': tolerance,
        'disp': False
    }
    
    bounds = None
    if function.bounds is not None:
        lower, upper = function.bounds
        bounds = list(zip(lower, upper))
    
    # 実行時間を計測
    start_time = time.perf_counter()
    result = minimize(
        function,
        initial_point,
        method='Nelder-Mead',
        options=options,
        bounds=bounds
    )
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    optimization_result = {
        'optimal_parameters': result.x.tolist(),
        'optimal_value': float(result.fun),
        'converged': result.success,
        'iterations': int(result.nit),
        'message': result.message,
        'execution_time_seconds': float(execution_time)
    }
    
    return optimization_result, execution_time


def generate_test_case(function: TestFunction,
                      initial_point: np.ndarray,
                      test_name: str,
                      max_iterations: int = 1000,
                      tolerance: float = 1e-6,
                      seed: Optional[int] = None,
                      verbose: bool = True) -> Dict:
    """単一のテストケースを生成"""
    
    if verbose:
        print(f"  処理中: {test_name} (次元={function.dimension})...", end=" ", flush=True)
    
    # scipyで最適化を実行（実行時間も取得）
    scipy_result, execution_time = run_scipy_optimization(function, initial_point, max_iterations, tolerance)
    
    if verbose:
        status = "✓" if scipy_result['converged'] else "✗"
        print(f"{status} 反復={scipy_result['iterations']}, 値={scipy_result['optimal_value']:.6e}, "
              f"時間={execution_time:.3f}秒")
    
    # テストケースの情報を構築
    test_case = {
        'name': test_name,
        'function': {
            'name': function.name,
            'dimension': function.dimension,
            'optimal_point': function.optimal_point.tolist(),
            'optimal_value': float(function.optimal_value)
        },
        'initial_parameters': initial_point.tolist(),
        'options': {
            'max_iterations': max_iterations,
            'tolerance': tolerance
        },
        'expected_result': scipy_result,
        'seed': seed
    }
    
    # 境界条件がある場合は追加
    if function.bounds is not None:
        lower, upper = function.bounds
        test_case['bounds'] = {
            'lower': lower.tolist(),
            'upper': upper.tolist()
        }
    
    return test_case


def generate_all_test_cases(verbose: bool = True) -> List[Dict]:
    """すべてのテストケースを生成"""
    test_cases = []
    np.random.seed(42)  # 再現性のためのシード設定
    
    if verbose:
        print("テストケースを生成中...")
    
    # 低次元のテストケース
    # 1. Sphere関数（1次元、境界なし）
    sphere1d = SphereFunction(1)
    initial1d = np.array([5.0])
    test_cases.append(generate_test_case(sphere1d, initial1d, "Sphere_1D", seed=42, verbose=verbose))
    
    # 2. Sphere関数（2次元、境界なし）
    sphere = SphereFunction(2)
    initial = np.array([3.0, 4.0])
    test_cases.append(generate_test_case(sphere, initial, "Sphere_2D", seed=42, verbose=verbose))
    
    # 3. Sphere関数（3次元、境界なし）
    sphere3d = SphereFunction(3)
    initial3d = np.array([1.0, 2.0, 3.0])
    test_cases.append(generate_test_case(sphere3d, initial3d, "Sphere_3D", seed=42, verbose=verbose))
    
    # 4. Rosenbrock関数（2次元、境界なし）
    rosenbrock = RosenbrockFunction(2)
    initial_rosen = np.array([-1.2, 1.0])
    test_cases.append(generate_test_case(rosenbrock, initial_rosen, "Rosenbrock_2D", seed=42, verbose=verbose))
    
    # 5. Rosenbrock関数（3次元、境界なし）
    rosenbrock3d = RosenbrockFunction(3)
    initial_rosen3d = np.array([-1.2, 1.0, -1.0])
    test_cases.append(generate_test_case(rosenbrock3d, initial_rosen3d, "Rosenbrock_3D", seed=42, verbose=verbose))
    
    # 6. Ackley関数（2次元、境界あり）
    ackley = AckleyFunction(2)
    initial_ackley = np.array([2.0, 2.0])
    test_cases.append(generate_test_case(ackley, initial_ackley, "Ackley_2D", seed=42, verbose=verbose))
    
    # 7. Rastrigin関数（2次元、境界あり）
    rastrigin = RastriginFunction(2)
    initial_rastrigin = np.array([2.0, 2.0])
    test_cases.append(generate_test_case(rastrigin, initial_rastrigin, "Rastrigin_2D", seed=42, verbose=verbose))
    
    # 8. Beale関数（2次元、境界あり）
    beale = BealeFunction()
    initial_beale = np.array([1.0, 1.0])
    test_cases.append(generate_test_case(beale, initial_beale, "Beale_2D", seed=42, verbose=verbose))
    
    # 9. Booth関数（2次元、境界あり）
    booth = BoothFunction()
    initial_booth = np.array([0.0, 0.0])
    test_cases.append(generate_test_case(booth, initial_booth, "Booth_2D", seed=42, verbose=verbose))
    
    # 10. Matyas関数（2次元、境界あり）
    matyas = MatyasFunction()
    initial_matyas = np.array([5.0, 5.0])
    test_cases.append(generate_test_case(matyas, initial_matyas, "Matyas_2D", seed=42, verbose=verbose))
    
    # 中次元のテストケース
    # 11. Sphere関数（5次元、境界なし）
    sphere5d = SphereFunction(5)
    initial5d = np.random.uniform(-5.0, 5.0, 5)
    test_cases.append(generate_test_case(sphere5d, initial5d, "Sphere_5D", seed=42, verbose=verbose))
    
    # 12. Sphere関数（10次元、境界なし）
    sphere10d = SphereFunction(10)
    initial10d = np.random.uniform(-5.0, 5.0, 10)
    test_cases.append(generate_test_case(sphere10d, initial10d, "Sphere_10D", seed=42, verbose=verbose))
    
    # 13. Rosenbrock関数（5次元、境界なし）
    rosenbrock5d = RosenbrockFunction(5)
    initial_rosen5d = np.random.uniform(-2.0, 2.0, 5)
    test_cases.append(generate_test_case(rosenbrock5d, initial_rosen5d, "Rosenbrock_5D", seed=42, verbose=verbose))
    
    # 14. Rosenbrock関数（10次元、境界なし）
    rosenbrock10d = RosenbrockFunction(10)
    initial_rosen10d = np.random.uniform(-2.0, 2.0, 10)
    test_cases.append(generate_test_case(rosenbrock10d, initial_rosen10d, "Rosenbrock_10D", seed=42, verbose=verbose))
    
    # 15. Ackley関数（5次元、境界あり）
    ackley5d = AckleyFunction(5)
    initial_ackley5d = np.random.uniform(-3.0, 3.0, 5)
    test_cases.append(generate_test_case(ackley5d, initial_ackley5d, "Ackley_5D", seed=42, verbose=verbose))
    
    # 16. Ackley関数（10次元、境界あり）
    ackley10d = AckleyFunction(10)
    initial_ackley10d = np.random.uniform(-3.0, 3.0, 10)
    test_cases.append(generate_test_case(ackley10d, initial_ackley10d, "Ackley_10D", seed=42, verbose=verbose))
    
    # 17. Rastrigin関数（5次元、境界あり）
    rastrigin5d = RastriginFunction(5)
    initial_rastrigin5d = np.random.uniform(-3.0, 3.0, 5)
    test_cases.append(generate_test_case(rastrigin5d, initial_rastrigin5d, "Rastrigin_5D", seed=42, verbose=verbose))
    
    # 18. Rastrigin関数（10次元、境界あり）
    rastrigin10d = RastriginFunction(10)
    initial_rastrigin10d = np.random.uniform(-3.0, 3.0, 10)
    test_cases.append(generate_test_case(rastrigin10d, initial_rastrigin10d, "Rastrigin_10D", seed=42, verbose=verbose))
    
    # 高次元のテストケース（最大反復回数を増やす）
    # 19. Sphere関数（20次元、境界なし）
    sphere20d = SphereFunction(20)
    initial20d = np.random.uniform(-5.0, 5.0, 20)
    test_cases.append(generate_test_case(sphere20d, initial20d, "Sphere_20D", 
                                         max_iterations=5000, seed=42, verbose=verbose))
    
    # 20. Sphere関数（50次元、境界なし）
    sphere50d = SphereFunction(50)
    initial50d = np.random.uniform(-5.0, 5.0, 50)
    test_cases.append(generate_test_case(sphere50d, initial50d, "Sphere_50D", 
                                         max_iterations=10000, seed=42, verbose=verbose))
    
    # 21. Sphere関数（100次元、境界なし）
    sphere100d = SphereFunction(100)
    initial100d = np.random.uniform(-5.0, 5.0, 100)
    test_cases.append(generate_test_case(sphere100d, initial100d, "Sphere_100D", 
                                         max_iterations=20000, seed=42, verbose=verbose))
    
    # 22. Rosenbrock関数（20次元、境界なし）
    rosenbrock20d = RosenbrockFunction(20)
    initial_rosen20d = np.random.uniform(-2.0, 2.0, 20)
    test_cases.append(generate_test_case(rosenbrock20d, initial_rosen20d, "Rosenbrock_20D", 
                                         max_iterations=5000, seed=42, verbose=verbose))
    
    # 23. Rosenbrock関数（50次元、境界なし）
    rosenbrock50d = RosenbrockFunction(50)
    initial_rosen50d = np.random.uniform(-2.0, 2.0, 50)
    test_cases.append(generate_test_case(rosenbrock50d, initial_rosen50d, "Rosenbrock_50D", 
                                         max_iterations=10000, seed=42, verbose=verbose))
    
    # 24. Rosenbrock関数（100次元、境界なし）
    rosenbrock100d = RosenbrockFunction(100)
    initial_rosen100d = np.random.uniform(-2.0, 2.0, 100)
    test_cases.append(generate_test_case(rosenbrock100d, initial_rosen100d, "Rosenbrock_100D", 
                                         max_iterations=20000, seed=42, verbose=verbose))
    
    # 25. Ackley関数（20次元、境界あり）
    ackley20d = AckleyFunction(20)
    initial_ackley20d = np.random.uniform(-3.0, 3.0, 20)
    test_cases.append(generate_test_case(ackley20d, initial_ackley20d, "Ackley_20D", 
                                         max_iterations=5000, seed=42, verbose=verbose))
    
    # 26. Ackley関数（50次元、境界あり）
    ackley50d = AckleyFunction(50)
    initial_ackley50d = np.random.uniform(-3.0, 3.0, 50)
    test_cases.append(generate_test_case(ackley50d, initial_ackley50d, "Ackley_50D", 
                                         max_iterations=10000, seed=42, verbose=verbose))
    
    # 27. Ackley関数（100次元、境界あり）
    ackley100d = AckleyFunction(100)
    initial_ackley100d = np.random.uniform(-3.0, 3.0, 100)
    test_cases.append(generate_test_case(ackley100d, initial_ackley100d, "Ackley_100D", 
                                         max_iterations=20000, seed=42, verbose=verbose))
    
    # 28. Rastrigin関数（20次元、境界あり）
    rastrigin20d = RastriginFunction(20)
    initial_rastrigin20d = np.random.uniform(-3.0, 3.0, 20)
    test_cases.append(generate_test_case(rastrigin20d, initial_rastrigin20d, "Rastrigin_20D", 
                                         max_iterations=5000, seed=42, verbose=verbose))
    
    # 29. Rastrigin関数（50次元、境界あり）
    rastrigin50d = RastriginFunction(50)
    initial_rastrigin50d = np.random.uniform(-3.0, 3.0, 50)
    test_cases.append(generate_test_case(rastrigin50d, initial_rastrigin50d, "Rastrigin_50D", 
                                         max_iterations=10000, seed=42, verbose=verbose))
    
    # 30. Rastrigin関数（100次元、境界あり）
    rastrigin100d = RastriginFunction(100)
    initial_rastrigin100d = np.random.uniform(-3.0, 3.0, 100)
    test_cases.append(generate_test_case(rastrigin100d, initial_rastrigin100d, "Rastrigin_100D", 
                                         max_iterations=20000, seed=42, verbose=verbose))
    
    return test_cases


def main():
    """メイン関数"""
    print("Nelder-Meadテストデータ生成を開始...")
    
    total_start_time = time.perf_counter()
    
    # テストケースを生成
    test_cases = generate_all_test_cases(verbose=True)
    
    total_end_time = time.perf_counter()
    total_execution_time = total_end_time - total_start_time
    
    # 出力ディレクトリを取得
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "tests" / "optimize" / "NelderMead" / "data"
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "nelder_mead_test_data.json"
    
    # 実行時間の統計を計算
    execution_times = [tc['expected_result']['execution_time_seconds'] for tc in test_cases]
    total_time = sum(execution_times)
    avg_time = total_time / len(execution_times) if execution_times else 0.0
    min_time = min(execution_times) if execution_times else 0.0
    max_time = max(execution_times) if execution_times else 0.0
    
    # JSON形式で保存
    output_data = {
        'description': 'Nelder-Meadアルゴリズムのテストデータ',
        'generated_by': 'gen_nelder_mead_test_data.py',
        'generation_time_seconds': float(total_execution_time),
        'statistics': {
            'total_test_cases': len(test_cases),
            'total_optimization_time_seconds': float(total_time),
            'average_optimization_time_seconds': float(avg_time),
            'min_optimization_time_seconds': float(min_time),
            'max_optimization_time_seconds': float(max_time)
        },
        'test_cases': test_cases
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nテストデータを {output_file} に保存しました。")
    print(f"合計 {len(test_cases)} 個のテストケースを生成しました。")
    print(f"総実行時間: {total_execution_time:.2f}秒")
    print(f"最適化の総時間: {total_time:.2f}秒")
    print(f"平均最適化時間: {avg_time:.3f}秒")
    print(f"最小最適化時間: {min_time:.3f}秒")
    print(f"最大最適化時間: {max_time:.3f}秒")
    
    # 各テストケースの概要を表示
    print("\n生成されたテストケース:")
    for i, test_case in enumerate(test_cases, 1):
        exec_time = test_case['expected_result']['execution_time_seconds']
        print(f"{i}. {test_case['name']}: "
              f"次元={test_case['function']['dimension']}, "
              f"期待値={test_case['expected_result']['optimal_value']:.6e}, "
              f"収束={test_case['expected_result']['converged']}, "
              f"反復回数={test_case['expected_result']['iterations']}, "
              f"実行時間={exec_time:.3f}秒")


if __name__ == "__main__":
    main()
