"""
诊断脚本：检查 gene_space 约束是否正确应用
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pygad import ES


def test_gene_space_initialization():
    """
    测试初始化时 gene_space 是否正确应用
    """
    print("=" * 60)
    print("测试初始化时 gene_space 的应用")
    print("=" * 60)

    num_genes = 3

    gene_space = [
        [-5.0, 5.0],
        [0, 1, 2, 3, 4, 5],
        {"low": -10.0, "high": 10.0, "step": 2.0}
    ]

    es_instance = ES(
        num_generations=1,
        mu=5,
        lambda_=10,
        fitness_func=lambda *args: 0,
        num_genes=num_genes,
        sol_per_pop=5,
        gene_space=gene_space,
        random_seed=42,
        suppress_warnings=True
    )

    print(f"\n初始种群:")
    for i, sol in enumerate(es_instance.population):
        print(f"  个体 {i}: {sol}")

    print(f"\n检查约束:")
    all_valid = True

    for i, sol in enumerate(es_instance.population):
        valid = True

        if not (-5.0 <= sol[0] <= 5.0):
            print(f"  个体 {i} 基因0: {sol[0]} 超出 [-5, 5]")
            valid = False
            all_valid = False

        if sol[1] not in [0, 1, 2, 3, 4, 5]:
            print(f"  个体 {i} 基因1: {sol[1]} 不在 [0,1,2,3,4,5]")
            valid = False
            all_valid = False

        expected_gene2 = np.arange(-10.0, 10.0, 2.0)
        closest = expected_gene2[np.argmin(np.abs(expected_gene2 - sol[2]))]
        if not np.isclose(sol[2], closest):
            print(f"  个体 {i} 基因2: {sol[2]} 不是 2 的倍数 (应为 {closest})")
            valid = False
            all_valid = False

        if valid:
            print(f"  个体 {i}: ✓ 所有基因符合约束")

    if all_valid:
        print("\n✓ 初始化时 gene_space 约束正确应用")
    else:
        print("\n✗ 初始化时存在约束违规")

    return all_valid


def test_clip_to_space():
    """
    直接测试 _clip_to_space 方法
    """
    print("\n" + "=" * 60)
    print("测试 _clip_to_space 方法")
    print("=" * 60)

    es_instance = ES(
        num_generations=1,
        mu=1,
        lambda_=1,
        fitness_func=lambda *args: 0,
        num_genes=1,
        sol_per_pop=1,
        random_seed=42,
        suppress_warnings=True
    )

    test_cases = [
        (-6.3313, {"low": -10.0, "high": 10.0, "step": 2.0}, -6.0),
        (3.7, [0, 1, 2, 3, 4, 5], 4.0),
        (7.8, [-5.0, 5.0], 5.0),
        (-3.2, {"low": 0.0, "high": 10.0, "step": 2.0}, 0.0),
    ]

    all_passed = True
    for value, space, expected in test_cases:
        result = es_instance._clip_to_space(value, space)
        passed = np.isclose(result, expected)
        status = "✓" if passed else "✗"
        print(f"  {status} clip({value}, {space}) = {result:.4f} (预期: {expected})")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ _clip_to_space 方法正确")
    else:
        print("\n✗ _clip_to_space 方法存在问题")

    return all_passed


def test_apply_gene_space_constraint():
    """
    测试 _apply_gene_space_constraint 方法
    """
    print("\n" + "=" * 60)
    print("测试 _apply_gene_space_constraint 方法")
    print("=" * 60)

    gene_space = [
        [-5.0, 5.0],
        [0, 1, 2, 3, 4, 5],
        {"low": -10.0, "high": 10.0, "step": 2.0}
    ]

    es_instance = ES(
        num_generations=1,
        mu=1,
        lambda_=1,
        fitness_func=lambda *args: 0,
        num_genes=3,
        sol_per_pop=1,
        gene_space=gene_space,
        random_seed=42,
        suppress_warnings=True
    )

    test_values = [
        (7.8, 0, 5.0),
        (2.3, 1, 2.0),
        (-6.3313, 2, -6.0),
    ]

    all_passed = True
    for value, gene_idx, expected in test_values:
        result = es_instance._apply_gene_space_constraint(value, gene_idx)
        passed = np.isclose(result, expected)
        status = "✓" if passed else "✗"
        print(f"  {status} 基因{gene_idx}: {value} -> {result:.4f} (预期: {expected})")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ _apply_gene_space_constraint 方法正确")
    else:
        print("\n✗ _apply_gene_space_constraint 方法存在问题")

    return all_passed


def test_gene_space_with_mutation():
    """
    测试变异后约束是否正确应用
    """
    print("\n" + "=" * 60)
    print("测试变异后 gene_space 约束的应用")
    print("=" * 60)

    num_genes = 3

    gene_space = [
        [-5.0, 5.0],
        [0, 1, 2, 3, 4, 5],
        {"low": -10.0, "high": 10.0, "step": 2.0}
    ]

    violations = []

    def check_fitness(ga_instance, solution, solution_idx):
        if not (-5.0 <= solution[0] <= 5.0):
            violations.append(f"基因0: {solution[0]} 超出 [-5, 5]")
        if solution[1] not in [0, 1, 2, 3, 4, 5]:
            violations.append(f"基因1: {solution[1]} 不在离散集合")
        expected = np.arange(-10.0, 10.0, 2.0)
        closest = expected[np.argmin(np.abs(expected - solution[2]))]
        if not np.isclose(solution[2], closest):
            violations.append(f"基因2: {solution[2]} 不是步长2的倍数")
        return -np.sum(solution ** 2)

    es_instance = ES(
        num_generations=10,
        mu=5,
        lambda_=20,
        fitness_func=check_fitness,
        num_genes=num_genes,
        sol_per_pop=5,
        gene_space=gene_space,
        sigma=10.0,
        sigma_adaptation="none",
        random_seed=123,
        suppress_warnings=True
    )

    print(f"初始种群:")
    for i, sol in enumerate(es_instance.population):
        print(f"  {i}: {sol}")

    initial_violations = len(violations)
    print(f"\n初始违规数: {initial_violations}")

    es_instance.run()

    final_violations = len(violations) - initial_violations
    print(f"进化中新增违规数: {final_violations}")

    if final_violations == 0:
        print("✓ 变异后约束正确应用")
        return True
    else:
        print(f"✗ 检测到 {final_violations} 个违规!")
        for v in violations[initial_violations:][:10]:
            print(f"  - {v}")
        return False


def main():
    print("诊断 gene_space 约束问题")
    print("=" * 60)

    results = {}

    results['clip_to_space'] = test_clip_to_space()
    results['apply_gene_space_constraint'] = test_apply_gene_space_constraint()
    results['initialization'] = test_gene_space_initialization()
    results['mutation'] = test_gene_space_with_mutation()

    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)

    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {status}: {name}")


if __name__ == "__main__":
    main()
