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
    
    return best_fitness


def test_mu_comma_lambda_es():
    """
    测试 (μ,λ)-ES
    仅从子代中选择下一代
    """
    print("\n" + "=" * 60)
    print("测试 (μ,λ)-ES (1/5成功规则自适应步长)")
    print("=" * 60)

    num_genes = 10
    
    def on_generation(ga_instance):
        if ga_instance.generations_completed % 50 == 0:
            best_sol, best_fit, _ = ga_instance.best_solution()
            print(f"代 {ga_instance.generations_completed:4d}: 最佳适应度 = {best_fit:.4e}")

    es_instance = ES(
        num_generations=500,
        mu=30,
        lambda_=200,
        fitness_func=sphere_function,
        num_genes=num_genes,
        sol_per_pop=30,
        init_range_low=-5.0,
        init_range_high=5.0,
        selection_type="comma",
        recombination_type="intermediate",
        mutation_type="gaussian",
        sigma=0.3,
        sigma_adaptation="1/5-rule",
        one_fifth_c=0.85,
        one_fifth_g=5,
        random_seed=123,
        on_generation=on_generation,
        suppress_warnings=True
    )

    print(f"初始步长: {es_instance.population_sigma[0, 0]:.4f}")
    
    es_instance.run()

    best_solution, best_fitness, _ = es_instance.best_solution()
    print(f"\n最终结果:")
    print(f"  最佳适应度 = {best_fitness:.6e}")
    print(f"  最佳解的l2范数 = {np.linalg.norm(best_solution):.6e}")
    print(f"  最终步长: {es_instance.population_sigma[0, 0]:.6e}")
    
    return best_fitness


def test_rastrigin_optimization():
    """
    测试在更复杂的Rastrigin函数上的优化
    """
    print("\n" + "=" * 60)
    print("测试 Rastrigin函数优化 (更复杂的多峰函数)")
    print("=" * 60)

    num_genes = 5

    def on_generation(ga_instance):
        if ga_instance.generations_completed % 100 == 0:
            best_sol, best_fit, _ = ga_instance.best_solution()
            print(f"代 {ga_instance.generations_completed:4d}: 最佳适应度 = {best_fit:.4f}")

    es_instance = ES(
        num_generations=1000,
        mu=20,
        lambda_=100,
        fitness_func=rastrigin_function,
        num_genes=num_genes,
        sol_per_pop=20,
        init_range_low=-5.12,
        init_range_high=5.12,
        selection_type="plus",
        recombination_type="discrete",
        mutation_type="gaussian",
        sigma=1.0,
        sigma_adaptation="log-normal",
        random_seed=789,
        on_generation=on_generation,
        suppress_warnings=True
    )
    
    es_instance.run()

    best_solution, best_fitness, _ = es_instance.best_solution()
    rastrigin_value = -best_fitness
    print(f"\n最终结果:")
    print(f"  最佳适应度 = {best_fitness:.6f}")
    print(f"  Rastrigin函数值 = {rastrigin_value:.6f}")
    print(f"  最佳解 = {best_solution}")
    
    return best_fitness


def test_fixed_sigma():
    """
    测试固定步长 (用于对比
    """
    print("\n" + "=" * 60)
    print("测试固定步长 (无自适应)")
    print("=" * 60)

    num_genes = 10

    es_instance = ES(
        num_generations=300,
        mu=15,
        lambda_=50,
        fitness_func=sphere_function,
        num_genes=num_genes,
        sol_per_pop=15,
        init_range_low=-5.0,
        init_range_high=5.0,
        selection_type="plus",
        recombination_type="intermediate",
        mutation_type="gaussian",
        sigma=0.1,
        sigma_adaptation="none",
        random_seed=456,
        suppress_warnings=True
    )
    
    es_instance.run()

    best_solution, best_fitness, _ = es_instance.best_solution()
    print(f"最终结果:")
    print(f"  最佳适应度 = {best_fitness:.6e}")
    print(f"  最佳解的l2范数 = {np.linalg.norm(best_solution):.6e}")
    print(f"  步长 (固定) = {es_instance.population_sigma[0, 0]:.6e}")
    
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
    for name, triggered in callbacks_triggered.items():
        status = "✓ 已触发" if triggered else "✗ 未触发"
        print(f"  {name}: {status}")

    print(f"\n完成代数: {es_instance.generations_completed}")
    assert es_instance.generations_completed == 50, "on_generation stop 没有正确停止"
    print("回调函数测试通过!")


def main():
    print("=" * 60)
    print("进化策略(ES)模块测试")
    print("=" * 60)

    print("\n测试目标:")
    print("  - Sphere函数 f(x) = -sum(x_i^2)")
    print("  - 寻找最优解 x* = [0, 0, ..., 0]")
    print("  - 最优适应度 = 0.0")

    results = {}

    # 测试1: (μ+λ)-ES with log-normal
    fit1 = test_mu_plus_lambda_es()
    results['mu_plus_lambda'] = fit1

    # 测试2: (μ,λ)-ES with 1/5-rule
    fit2 = test_mu_comma_lambda_es()
    results['mu_comma_lambda'] = fit2

    # 测试3: Rastrigin函数
    fit3 = test_rastrigin_optimization()
    results['rastrigin'] = fit3

    # 测试4: 固定步长
    fit4 = test_fixed_sigma()
    results['fixed_sigma'] = fit4

    # 测试5: 回调函数
    test_callback_functions()
    results['callbacks'] = 'PASS'

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"\nSphere函数优化结果 (目标: 0.0):")
    for name, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.6e}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
