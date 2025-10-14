"""
AAD Edge-Pushing 快速开始指南
============================

5分钟上手Hessian计算
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aad_edge_pushing.algo3.algo4_optimized import compute_hessian
from aad_edge_pushing.aad.engine import global_tape
from aad_edge_pushing.aad.advar import ADVar
import numpy as np


def example1_simple_quadratic():
    """
    示例1：简单二次函数
    f(x, y) = x² + xy + y²
    """
    print("=" * 60)
    print("示例1：f(x, y) = x² + xy + y²")
    print("=" * 60)

    # 重置tape
    global_tape.reset()

    # 定义变量
    x = ADVar(2.0)  # x = 2
    y = ADVar(3.0)  # y = 3

    # 计算函数
    z = x*x + x*y + y*y

    print(f"函数值: f(2, 3) = {z.value}")

    # 计算Hessian
    H = compute_hessian(global_tape, seed=1.0)

    # 提取Hessian矩阵
    print("\nHessian矩阵:")
    print(f"  ∂²f/∂x² = {H.get(0, 0)}")
    print(f"  ∂²f/∂x∂y = {H.get(0, 1)}")
    print(f"  ∂²f/∂y² = {H.get(1, 1)}")

    # 理论值
    print("\n理论值:")
    print("  ∂²f/∂x² = 2")
    print("  ∂²f/∂x∂y = 1")
    print("  ∂²f/∂y² = 2")


def example2_transcendental():
    """
    示例2：超越函数
    f(x) = exp(x²)
    """
    print("\n" + "=" * 60)
    print("示例2：f(x) = exp(x²)")
    print("=" * 60)

    global_tape.reset()

    x = ADVar(1.0)
    z = (x * x).exp()

    print(f"函数值: f(1) = {z.value:.6f}")

    H = compute_hessian(global_tape, seed=1.0)

    # f(x) = exp(x²)
    # f'(x) = 2x·exp(x²)
    # f''(x) = (2 + 4x²)·exp(x²)
    # 在x=1: f''(1) = 6·exp(1) ≈ 16.31

    print(f"\nHessian: ∂²f/∂x² = {H.get(0, 0):.6f}")
    print(f"理论值: 6·exp(1) = {6 * np.exp(1):.6f}")


def example3_three_variables():
    """
    示例3：三变量函数
    f(x, y, z) = x²y + yz² + xz
    """
    print("\n" + "=" * 60)
    print("示例3：f(x, y, z) = x²y + yz² + xz")
    print("=" * 60)

    global_tape.reset()

    x = ADVar(1.0)
    y = ADVar(2.0)
    z = ADVar(3.0)

    result = x*x*y + y*z*z + x*z

    print(f"函数值: f(1, 2, 3) = {result.value}")

    H = compute_hessian(global_tape, seed=1.0)

    # 转换为NumPy数组以便查看
    n = 3
    H_dense = np.array([[H.get(i, j) for j in range(n)] for i in range(n)])

    print("\nHessian矩阵 (3×3):")
    print(H_dense)

    print("\n理论Hessian:")
    print("[[2y,  2x,  1 ]    [[4,  2,  1]")
    print(" [2x,  2z², 2z]  =  [2, 18,  6]")
    print(" [1,   2z,  2y]]    [1,  6,  4]]")


def example4_composition():
    """
    示例4：函数复合
    f(x, y) = sin(x·y) + log(x + y)
    """
    print("\n" + "=" * 60)
    print("示例4：f(x, y) = sin(x·y) + log(x + y)")
    print("=" * 60)

    global_tape.reset()

    x = ADVar(1.0)
    y = ADVar(2.0)

    result = (x * y).sin() + (x + y).log()

    print(f"函数值: f(1, 2) = {result.value:.6f}")

    H = compute_hessian(global_tape, seed=1.0)

    print("\nHessian矩阵:")
    print(f"  ∂²f/∂x² = {H.get(0, 0):.6f}")
    print(f"  ∂²f/∂x∂y = {H.get(0, 1):.6f}")
    print(f"  ∂²f/∂y² = {H.get(1, 1):.6f}")


def example5_sparse_hessian():
    """
    示例5：稀疏Hessian（演示优化效果）
    f(x₁, x₂, ..., x₁₀) = x₁² + x₅² + x₁₀²
    （仅3个变量有贡献，Hessian高度稀疏）
    """
    print("\n" + "=" * 60)
    print("示例5：稀疏Hessian - f = x₁² + x₅² + x₁₀²")
    print("=" * 60)

    global_tape.reset()

    n = 10
    vars = [ADVar(float(i+1)) for i in range(n)]

    # 仅使用3个变量
    result = vars[0]**2 + vars[4]**2 + vars[9]**2

    print(f"函数值: {result.value}")

    H = compute_hessian(global_tape, seed=1.0)

    # 统计非零元素
    nonzero_count = sum(1 for i in range(n) for j in range(i, n)
                       if abs(H.get(i, j)) > 1e-10)
    total_entries = n * (n + 1) // 2  # 对称矩阵的上三角

    print(f"\nHessian稀疏性:")
    print(f"  非零元素: {nonzero_count}/{total_entries}")
    print(f"  稀疏度: {100 * (1 - nonzero_count/total_entries):.1f}%")

    print(f"\n非零对角元素:")
    print(f"  ∂²f/∂x₁² = {H.get(0, 0)}")
    print(f"  ∂²f/∂x₅² = {H.get(4, 4)}")
    print(f"  ∂²f/∂x₁₀² = {H.get(9, 9)}")

    print("\n✓ 边推算法仅访问这3个非零元素，无需扫描全部55个位置！")


def main():
    """运行所有示例"""
    print("\n" + "🚀 " * 20)
    print("AAD Edge-Pushing Hessian计算 - 快速开始")
    print("🚀 " * 20)

    example1_simple_quadratic()
    example2_transcendental()
    example3_three_variables()
    example4_composition()
    example5_sparse_hessian()

    print("\n" + "=" * 60)
    print("✅ 所有示例运行完成！")
    print("=" * 60)
    print("\n下一步:")
    print("  - 查看 bsm_greeks_demo.py 了解期权Greeks计算")
    print("  - 运行 benchmarks/main_benchmark.py 查看性能对比")
    print("  - 阅读 LITERATURE_REVIEW.md 了解算法原理")
    print()


if __name__ == "__main__":
    main()
