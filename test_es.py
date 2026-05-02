"""
进化策略(ES)模块测试脚本
测试 (μ+λ)-ES 和 (μ,λ)-ES 以及自适应步长σ
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygad
from pygad import ES
print(f"PyGAD version: {pygad.__version__}")
print(f"ES class imported successfully: {ES is not None}")


def sphere_function(ga_instance, solution, solution_idx):
    """
    Sphere函数：f(x) = sum(x_i^2)
    这是一个常用的连续优化测试函数，最小值在 x = [0, 0, ..., 0]，f(x) = 0
    我们将其转化为最大化问题：fitness = -sum(x_i^2)
    """
    fitness = -np.sum(solution ** 2)
    return fitness


def rastrigin_function(ga_instance, solution, solution_idx):
    """
    Rastrigin函数：f(x) = 10n + sum(x_i^2 - 10cos(2πx_i))
    最小值在 x = [0, 0, ..., 0]，f(x) = 0
    """
    n = len(solution)
    fitness = -(10 * n + np.sum(solution ** 2 - 10 * np.cos(2 * np.pi * solution)))
    return fitness


def test_mu_plus_lambda_es():
    """
    测试 (μ+λ)-ES
    从父代和子代的并集中选择下一代
    """
    print("\n" + "=" * 60)
    print("测试 (μ+λ)-ES (log-normal自适应步长)")
    print("=" * 60)

    num_genes = 10
    
    def on_generation(ga_instance):
        if ga_instance.generations_completed % 50 == 0:
            best_sol, best_fit, _ = ga_instance.best_solution()
            print(f"代 {ga_instance.generations_completed:4d}: 最佳适应度 = {best_fit:.4e}")

    es_instance = ES(
        num_generations=500,
        mu=15,
        lambda_=100,
        fitness_func=sphere_function,
        num_genes=num_genes,
        sol_per_pop=15,
        init_range_low=-5.0,
        init_range_high=5.0,
        selection_type="plus",
        recombination_type="intermediate",
        mutation_type="gaussian",
        sigma=0.5,
        sigma_adaptation="log-normal",
        random_seed=42,
        on_generation=on_generation,
        suppress_warnings=True
    )

    print(f"初始步长范围: [{es_instance.population_sigma.min():.4f}, {es_instance.population_sigma.max():.4f}]")
    print(f"初始种群适应度范围: [{es_instance.cal_pop_fitness().min():.4f}, {es_instance.cal_pop_fitness().max():.4f}]")
    
    es_instance.run()

    best_solution, best_fitness, _ = es_instance.best_solution()
    print(f"\n最终结果:")
    print(f"  最佳适应度 = {best_fitness:.6e}")
    print(f"  最佳解的l2范数 = {np.linalg.norm(best_solution):.6e}")
    print(f"  最终步长范围: [{es_instance.population_sigma.min():.6e}, {es_instance.population_sigma.max():.6e}]")
    
    assert best_fitness > -1e-30, f"log-normal ES should find good solution, got {best_fitness}"
    print("✓ (μ+λ)-ES 测试通过!")
    
    return best_fitness


def test_one_fifth_rule_bug_fix():
    """
    测试1/5成功规则的步长调整
    验证Bug修复：步长调整应该被正确应用，而不是被覆盖
    """
    print("\n" + "=" * 60)
    print("测试 1/5成功规则 (Bug修复验证)")
    print("=" * 60)

    num_genes = 5

    initial_sigma = 0.5

    es_instance = ES(
        num_generations=30,
        mu=5,
        lambda_=20,
        fitness_func=sphere_function,
        num_genes=num_genes,
        sol_per_pop=5,
        init_range_low=-1.0,
        init_range_high=1.0,
        selection_type="plus",
        recombination_type="intermediate",
        mutation_type="gaussian",
        sigma=initial_sigma,
        sigma_adaptation="1/5-rule",
        one_fifth_c=0.85,
        one_fifth_g=5,
        random_seed=12345,
        suppress_warnings=True
    )

    initial_sigma_values = es_instance.population_sigma.copy()
    print(f"初始步长: {initial_sigma_values[0, 0]:.6f}")

    sigma_history = []
    sigma_history.append(es_instance.population_sigma.mean())

    def on_gen(inst):
        sigma_history.append(inst.population_sigma.mean())
        if inst.generations_completed % 10 == 0:
            print(f"  代 {inst.generations_completed:3d}: 平均步长 = {inst.population_sigma.mean():.6f}")

    es_instance.on_generation = on_gen
    
    es_instance.run()

    final_sigma = es_instance.population_sigma.mean()
    print(f"\n最终步长: {final_sigma:.6f}")
    print(f"步长变化: {initial_sigma_values.mean():.6f} -> {final_sigma:.6f}")

    sigma_changed = not np.allclose(initial_sigma_values, es_instance.population_sigma)
    if sigma_changed:
        print("✓ 1/5规则步长调整正确应用 (Bug修复验证通过!)")
    else:
        print("✗ 警告: 步长没有变化，可能存在问题")

    return es_instance.best_solution()[1]


def test_cumulative_step_adaptation():
    """
    测试累积步长自适应 (CSA)
    验证_path_sigma和_path_length被正确使用
    """
    print("\n" + "=" * 60)
    print("测试 累积步长自适应 (CSA/Cumulative)")
    print("=" * 60)

    num_genes = 8

    es_instance = ES(
        num_generations=100,
        mu=10,
        lambda_=50,
        fitness_func=sphere_function,
        num_genes=num_genes,
        sol_per_pop=10,
        init_range_low=-3.0,
        init_range_high=3.0,
        selection_type="plus",
        recombination_type="intermediate",
        mutation_type="gaussian",
        sigma=0.3,
        sigma_adaptation="cumulative",
        cs=0.3,
        damping=1.0,
        random_seed=999,
        suppress_warnings=True
    )

    initial_path = es_instance._path_sigma.copy()
    initial_sigma = es_instance.population_sigma.mean()
    
    print(f"初始路径向量: {initial_path[:3]}...")
    print(f"初始平均步长: {initial_sigma:.6f}")

    def on_gen(inst):
        if inst.generations_completed % 25 == 0:
            print(f"  代 {inst.generations_completed:3d}: "
                  f"路径范数 = {np.linalg.norm(inst._path_sigma):.4f}, "
                  f"平均步长 = {inst.population_sigma.mean():.6f}")

    es_instance.on_generation = on_gen
    
    es_instance.run()

    final_path = es_instance._path_sigma
    final_sigma = es_instance.population_sigma.mean()

    print(f"\n最终路径向量: {final_path[:3]}...")
    print(f"最终平均步长: {final_sigma:.6f}")
    print(f"适应度: {es_instance.best_solution()[1]:.6e}")

    path_used = not np.allclose(initial_path, final_path)
    sigma_changed = not np.isclose(initial_sigma, final_sigma)

    if path_used:
        print("✓ 路径向量 _path_sigma 被正确更新")
    else:
        print("✗ 警告: 路径向量没有变化")

    if sigma_changed:
        print("✓ 步长在累积适应中被正确调整")
    else:
        print("✗ 警告: 步长没有变化")

    best_fitness = es_instance.best_solution()[1]
    if best_fitness > -0.1:
        print("✓ 累积步长自适应测试通过!")
    
    return best_fitness


def test_gene_space_constraints():
    """
    测试 gene_space 参数的约束应用
    """
    print("\n" + "=" * 60)
    print("测试 gene_space 约束")
    print("=" * 60)

    num_genes = 3

    gene_space = [
        [-5.0, 5.0],
        [0, 1, 2, 3, 4, 5],
        {"low": -10.0, "high": 10.0, "step": 2.0}
    ]

    violations_detected = []

    def check_constraints_fitness(ga_instance, solution, solution_idx):
        for i, val in enumerate(solution):
            if i == 0:
                if not (-5.0 <= val <= 5.0):
                    violations_detected.append(f"基因{i}: {val} 超出 [-5,5]")
            elif i == 1:
                if val not in [0, 1, 2, 3, 4, 5]:
                    violations_detected.append(f"基因{i}: {val} 不在离散集合")
            elif i == 2:
                expected = np.arange(-10.0, 10.0, 2.0)
                closest = expected[np.argmin(np.abs(expected - val))]
                if not np.isclose(val, closest):
                    violations_detected.append(f"基因{i}: {val} 不是 2 的倍数 (步长)")

        return -np.sum(solution ** 2)

    es_instance = ES(
        num_generations=100,
        mu=5,
        lambda_=20,
        fitness_func=check_constraints_fitness,
        num_genes=num_genes,
        sol_per_pop=5,
        gene_space=gene_space,
        selection_type="plus",
        sigma=0.5,
        sigma_adaptation="log-normal",
        random_seed=42,
        suppress_warnings=True
    )

    print(f"初始种群:")
    for i, sol in enumerate(es_instance.population[:3]):
        print(f"  {i}: {sol}")

    es_instance.run()

    if len(violations_detected) == 0:
        print("\n✓ gene_space 约束正确应用于初始化、重组和变异")
        best_sol, best_fit, _ = es_instance.best_solution()
        print(f"  最佳解: {best_sol}")
        print(f"  适应度: {best_fit:.6f}")
    else:
        print(f"\n✗ 检测到 {len(violations_detected)} 个约束违规!")
        for v in violations_detected[:10]:
            print(f"  - {v}")

    return len(violations_detected) == 0


def test_gene_constraint_callable():
    """
    测试 gene_constraint 可调用函数
    注意: gene_constraint 应该返回修复后的有效值，而不是仅仅过滤
    """
    print("\n" + "=" * 60)
    print("测试 gene_constraint 可调用约束")
    print("=" * 60)

    num_genes = 2

    def constraint_positive(solution, values):
        repaired = []
        for v in values:
            if v > 0:
                repaired.append(v)
            else:
                repaired.append(abs(v) + 0.01)
        return repaired

    def constraint_even(solution, values):
        repaired = []
        for v in values:
            rounded = round(v)
            if rounded % 2 == 0:
                repaired.append(float(rounded))
            else:
                diff_to_lower = v - (rounded - 1)
                diff_to_higher = (rounded + 1) - v
                if diff_to_lower < diff_to_higher:
                    repaired.append(float(rounded - 1))
                else:
                    repaired.append(float(rounded + 1))
        return repaired

    gene_constraint = [constraint_positive, constraint_even]

    violations = []

    def check_fitness(ga_instance, solution, solution_idx):
        if solution[0] <= 0:
            violations.append(f"基因0非正: {solution[0]}")
        val1 = solution[1]
        if abs(val1 - round(val1)) > 0.001 or round(val1) % 2 != 0:
            violations.append(f"基因1非偶数: {val1}")
        return -np.sum(solution ** 2)

    es_instance = ES(
        num_generations=50,
        mu=5,
        lambda_=20,
        fitness_func=check_fitness,
        num_genes=num_genes,
        sol_per_pop=5,
        init_range_low=1.0,
        init_range_high=10.0,
        gene_constraint=gene_constraint,
        selection_type="plus",
        sigma=1.0,
        sigma_adaptation="log-normal",
        random_seed=123,
        suppress_warnings=True
    )

    print(f"初始种群:")
    for i, sol in enumerate(es_instance.population):
        print(f"  {i}: {sol}")

    es_instance.run()

    if len(violations) == 0:
        print("\n✓ gene_constraint 可调用约束正确应用")
        best_sol, best_fit, _ = es_instance.best_solution()
        print(f"  最佳解: {best_sol}")
    else:
        print(f"\n✗ 检测到 {len(violations)} 个约束违规")

    return len(violations) == 0


def test_fixed_sigma():
    """
    测试固定步长 (用于对比)
    """
    print("\n" + "=" * 60)
    print("测试固定步长 (无自适应)")
    print("=" * 60)

    num_genes = 10
    fixed_sigma = 0.01

    es_instance = ES(
        num_generations=200,
        mu=15,
        lambda_=50,
        fitness_func=sphere_function,
        num_genes=num_genes,
        sol_per_pop=15,
        init_range_low=-1.0,
        init_range_high=1.0,
        selection_type="plus",
        recombination_type="intermediate",
        mutation_type="gaussian",
        sigma=fixed_sigma,
        sigma_adaptation="none",
        random_seed=456,
        suppress_warnings=True
    )

    initial_sigma = es_instance.population_sigma[0, 0].copy()
    
    es_instance.run()

    final_sigma = es_instance.population_sigma[0, 0]
    best_solution, best_fitness, _ = es_instance.best_solution()
    
    print(f"初始步长: {initial_sigma:.6e}")
    print(f"最终步长: {final_sigma:.6e}")
    print(f"步长不变: {np.isclose(initial_sigma, final_sigma)}")
    print(f"最佳适应度: {best_fitness:.6e}")
    print(f"注: 固定步长通常收敛较慢，但步长保持不变")

    assert np.isclose(initial_sigma, final_sigma), "Fixed sigma should not change"
    print("✓ 固定步长测试通过!")
    
    return best_fitness


def test_callback_functions():
    """
    测试各种回调函数
    """
    print("\n" + "=" * 60)
    print("测试回调函数")
    print("=" * 60)

    num_genes = 5
    callbacks_triggered = {
        'on_start': False,
        'on_fitness': False,
        'on_recombination': False,
        'on_mutation': False,
        'on_generation': False,
        'on_stop': False
    }

    def on_start(es_instance):
        callbacks_triggered['on_start'] = True
        print("  on_start: 进化开始")

    def on_fitness(es_instance, fitness):
        callbacks_triggered['on_fitness'] = True
        return fitness

    def on_recombination(es_instance, children, children_sigma):
        callbacks_triggered['on_recombination'] = True
        return children, children_sigma

    def on_mutation(es_instance, mutated_children, mutated_sigma):
        callbacks_triggered['on_mutation'] = True
        return mutated_children, mutated_sigma

    def on_generation(es_instance):
        callbacks_triggered['on_generation'] = True
        if es_instance.generations_completed >= 50:
            return "stop"
        return None

    def on_stop(es_instance, last_fitness):
        callbacks_triggered['on_stop'] = True
        print(f"  on_stop: 进化停止，完成代数: {es_instance.generations_completed}")

    es_instance = ES(
        num_generations=100,
        mu=10,
        lambda_=30,
        fitness_func=sphere_function,
        num_genes=num_genes,
        sol_per_pop=10,
        init_range_low=-2.0,
        init_range_high=2.0,
        selection_type="plus",
        sigma=0.3,
        sigma_adaptation="log-normal",
        random_seed=111,
        on_start=on_start,
        on_fitness=on_fitness,
        on_recombination=on_recombination,
        on_mutation=on_mutation,
        on_generation=on_generation,
        on_stop=on_stop,
        suppress_warnings=True
    )
    
    es_instance.run()

    print(f"\n回调函数触发情况:")
    all_triggered = True
    for name, triggered in callbacks_triggered.items():
        status = "✓ 已触发" if triggered else "✗ 未触发"
        print(f"  {name}: {status}")
        if not triggered:
            all_triggered = False

    print(f"\n完成代数: {es_instance.generations_completed}")
    assert es_instance.generations_completed == 50, "on_generation stop 没有正确停止"
    assert all_triggered, "不是所有回调函数都被触发"
    print("✓ 回调函数测试通过!")


def main():
    print("=" * 60)
    print("进化策略(ES)模块测试")
    print("=" * 60)

    print("\n测试目标:")
    print("  - Sphere函数 f(x) = -sum(x_i^2)")
    print("  - 寻找最优解 x* = [0, 0, ..., 0]")
    print("  - 最优适应度 = 0.0")

    results = {}

    fit1 = test_mu_plus_lambda_es()
    results['mu_plus_lambda'] = fit1

    fit2 = test_one_fifth_rule_bug_fix()
    results['one_fifth_rule'] = fit2

    fit3 = test_cumulative_step_adaptation()
    results['cumulative'] = fit3

    constraint1_passed = test_gene_space_constraints()
    results['gene_space'] = "PASS" if constraint1_passed else "FAIL"

    constraint2_passed = test_gene_constraint_callable()
    results['gene_constraint'] = "PASS" if constraint2_passed else "FAIL"

    fit4 = test_fixed_sigma()
    results['fixed_sigma'] = fit4

    test_callback_functions()
    results['callbacks'] = 'PASS'

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"\nSphere函数优化结果 (目标: 0.0):")
    for name, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.6e}")

    print("\n功能测试:")
    for name, value in results.items():
        if isinstance(value, str):
            status = "✓" if value == "PASS" else "✗"
            print(f"  {status} {name}: {value}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
