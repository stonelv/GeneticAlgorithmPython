import pygad
import numpy

function_inputs = [4, -2, 3.5, 5, -11, -4.7]
desired_output = 44

def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution * function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

num_generations = 100
num_parents_mating = 10
sol_per_pop = 20
num_genes = len(function_inputs)

num_islands = 4
migration_frequency = 10
num_migrants = 2

print("=" * 60)
print("示例1: 全连接拓扑 (fully_connected)")
print("=" * 60)
print(f"岛屿数量: {num_islands}")
print(f"迁移频率: 每 {migration_frequency} 代")
print(f"每次迁移个体数: {num_migrants}")
print("迁移策略: 选择最好的个体迁出，替换目标岛屿最差的个体")
print("-" * 60)

island_ga = pygad.IslandGA(
    num_islands=num_islands,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    migration_frequency=migration_frequency,
    num_migrants=num_migrants,
    migration_topology='fully_connected',
    migrant_selection='best',
    replacement='worst',
    random_seed=42
)

island_ga.run()

solution, solution_fitness, island_idx, sol_idx = island_ga.best_solution()
print(f"\n最佳解决方案来自岛屿 {island_idx + 1}")
print(f"最佳解决方案参数: {solution}")
print(f"最佳适应度值: {solution_fitness}")

prediction = numpy.sum(numpy.array(function_inputs) * solution)
print(f"基于最佳解决方案的预测输出: {prediction}")
print(f"期望输出: {desired_output}")
print(f"误差: {abs(prediction - desired_output)}")

print(f"\n执行的迁移次数: {island_ga.migrations_performed}")
print(f"完成的总代数: {island_ga.generations_completed}")

print("\n" + "=" * 60)
print("示例2: 环形拓扑 (ring) + 不同岛屿参数")
print("=" * 60)
print("环形拓扑: 每个岛屿只迁移到下一个相邻岛屿 (0→1→2→3→0)")
print("为不同岛屿设置不同的变异率:")
print("- 岛屿 1-2: 变异率 10% (默认)")
print("- 岛屿 3: 变异率 5% (更保守)")
print("- 岛屿 4: 变异率 20% (更激进)")
print("-" * 60)

island_params = [
    {'mutation_percent_genes': 10},
    {'mutation_percent_genes': 10},
    {'mutation_percent_genes': 5},
    {'mutation_percent_genes': 20},
]

island_ga_ring = pygad.IslandGA(
    num_islands=num_islands,
    num_generations=num_generations,
    num_parents_mating=num_parents_mating,
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    num_genes=num_genes,
    migration_frequency=15,
    num_migrants=3,
    migration_topology='ring',
    migrant_selection='best',
    replacement='worst',
    island_params=island_params,
    random_seed=42
)

island_ga_ring.run()

solution2, solution_fitness2, island_idx2, sol_idx2 = island_ga_ring.best_solution()
print(f"\n最佳解决方案来自岛屿 {island_idx2 + 1}")
print(f"最佳适应度值: {solution_fitness2}")

prediction2 = numpy.sum(numpy.array(function_inputs) * solution2)
print(f"基于最佳解决方案的预测输出: {prediction2}")
print(f"误差: {abs(prediction2 - desired_output)}")

print("\n各岛屿最终最佳适应度:")
for i, ga_instance in enumerate(island_ga_ring.islands):
    best_sol, best_fit, _ = ga_instance.best_solution()
    print(f"  岛屿 {i + 1}: 适应度 = {best_fit:.6f}")

print("\n" + "=" * 60)
print("示例3: 星形拓扑 (star) - 对比单种群 GA")
print("=" * 60)
print("星形拓扑: 岛屿 1 是中心岛，与其他所有岛屿相连")
print("其他岛屿只与中心岛通信，不直接互相通信")
print("-" * 60)

print("\n运行岛模型 (星形拓扑)...")
island_ga_star = pygad.IslandGA(
    num_islands=4,
    num_generations=50,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=num_genes,
    migration_frequency=5,
    num_migrants=2,
    migration_topology='star',
    random_seed=123
)
island_ga_star.run()
solution_star, fitness_star, _, _ = island_ga_star.best_solution()
pred_star = numpy.sum(numpy.array(function_inputs) * solution_star)
print(f"岛模型最佳适应度: {fitness_star}")
print(f"岛模型误差: {abs(pred_star - desired_output)}")

print("\n运行单种群 GA 作为对比...")
single_ga = pygad.GA(
    num_generations=50,
    num_parents_mating=10,
    fitness_func=fitness_func,
    sol_per_pop=80,
    num_genes=num_genes,
    random_seed=123
)
single_ga.run()
solution_single, fitness_single, _ = single_ga.best_solution()
pred_single = numpy.sum(numpy.array(function_inputs) * solution_single)
print(f"单种群最佳适应度: {fitness_single}")
print(f"单种群误差: {abs(pred_single - desired_output)}")

print("\n" + "-" * 60)
print("对比总结:")
print(f"  岛模型 (4岛 x 20个体) - 最佳适应度: {fitness_star:.6f}")
print(f"  单种群 (80个体)        - 最佳适应度: {fitness_single:.6f}")
print("=" * 60)

print("\n" + "=" * 60)
print("支持的参数配置选项")
print("=" * 60)
print("""
1. 迁移拓扑结构 (migration_topology):
   - 'fully_connected': 全连接 - 每个岛屿与所有其他岛屿相连
   - 'ring': 环形 - 每个岛屿只与相邻的下一个岛屿相连
   - 'star': 星形 - 一个中心岛屿与其他所有岛屿相连
   - 'random': 随机 - 每次迁移时随机选择目标岛屿

2. 迁移个体选择策略 (migrant_selection):
   - 'best': 选择适应度最好的个体迁出
   - 'random': 随机选择个体迁出

3. 替换策略 (replacement):
   - 'worst': 替换目标岛屿中适应度最差的个体
   - 'random': 随机替换目标岛屿中的个体

4. 其他可配置参数:
   - migration_frequency: 每隔多少代进行一次迁移
   - num_migrants: 每次迁移从每个岛屿迁出的个体数量
   - island_params: 为不同岛屿设置不同的进化参数
   - parallel_islands: 是否并行运行岛屿的进化过程
""")
