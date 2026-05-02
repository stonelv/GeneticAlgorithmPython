import numpy
import pygad
import tempfile
import os

num_generations = 20
num_parents_mating = 5
sol_per_pop = 10
num_genes = 4
random_seed = 42
num_islands = 3

def fitness_func(ga_instance, solution, solution_idx):
    return numpy.sum(solution**2)

def test_migration_updates_fitness():
    """
    关键测试：验证迁移后 last_generation_fitness 被正确更新。
    
    测试逻辑：
    1. 创建岛模型，设置较高的迁移频率
    2. 手动修改一个岛屿的个体，使其适应度显著变化
    3. 模拟迁移，将这个个体迁移到其他岛屿
    4. 验证目标岛屿的 last_generation_fitness 包含了新个体的正确适应度
    """
    print("测试: 迁移后适应度更新...")
    
    island_ga = pygad.IslandGA(
        num_islands=3,
        num_generations=30,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_genes,
        migration_frequency=10,
        num_migrants=1,
        migration_topology='fully_connected',
        random_seed=random_seed,
        suppress_warnings=True
    )
    
    for ga_instance in island_ga.islands:
        ga_instance.initialize_population(
            allow_duplicate_genes=ga_instance.allow_duplicate_genes,
            gene_type=ga_instance.gene_type,
            gene_constraint=ga_instance.gene_constraint
        )
        ga_instance.last_generation_fitness = ga_instance.cal_pop_fitness()
    
    island_0_before = island_ga.islands[0].last_generation_fitness.copy()
    
    high_fitness_solution = numpy.array([100.0, 100.0, 100.0, 100.0])
    high_fitness = fitness_func(None, high_fitness_solution, 0)
    
    island_ga.islands[0].population[0] = high_fitness_solution.copy()
    
    island_ga.islands[0].last_generation_fitness = island_ga.islands[0].cal_pop_fitness()
    assert island_ga.islands[0].last_generation_fitness[0] > max(island_0_before), \
        "高适应度个体的适应度应该被正确计算"
    
    island_1_before_fitness = island_ga.islands[1].last_generation_fitness.copy()
    
    island_ga._perform_migration()
    
    island_1_after_fitness = island_ga.islands[1].last_generation_fitness.copy()
    
    assert not numpy.array_equal(island_1_before_fitness, island_1_after_fitness), \
        "迁移后目标岛屿的适应度应该已更新"
    
    max_after = max(island_1_after_fitness)
    max_before = max(island_1_before_fitness)
    assert max_after >= max_before, \
        f"迁移后目标岛屿的最大适应度不应下降 (before={max_before}, after={max_after})"
    
    print("  ✓ 迁移后适应度正确更新")

def test_topologies():
    """
    测试所有支持的拓扑结构。
    """
    print("\n测试: 拓扑结构...")
    
    topologies = ['fully_connected', 'ring', 'star', 'random']
    
    for topology in topologies:
        print(f"  测试 {topology}...")
        island_ga = pygad.IslandGA(
            num_islands=4,
            num_generations=10,
            num_parents_mating=2,
            fitness_func=fitness_func,
            sol_per_pop=8,
            num_genes=num_genes,
            migration_frequency=5,
            num_migrants=1,
            migration_topology=topology,
            random_seed=random_seed,
            suppress_warnings=True
        )
        island_ga.run()
        
        assert island_ga.generations_completed == 10
        assert island_ga.migrations_performed >= 1
        
        solution, fitness, island_idx, sol_idx = island_ga.best_solution()
        assert solution is not None
        assert fitness is not None
        
        print(f"    ✓ {topology} 拓扑正常工作")

def test_migration_parameters():
    """
    测试迁移相关参数。
    """
    print("\n测试: 迁移参数...")
    
    print("  测试 migration_frequency...")
    island_ga = pygad.IslandGA(
        num_islands=3,
        num_generations=25,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_genes,
        migration_frequency=5,
        num_migrants=1,
        migration_topology='fully_connected',
        random_seed=random_seed,
        suppress_warnings=True
    )
    island_ga.run()
    
    expected_migrations = (25 - 1) // 5
    assert island_ga.migrations_performed == expected_migrations, \
        f"预期 {expected_migrations} 次迁移，实际 {island_ga.migrations_performed} 次"
    print(f"    ✓ migration_frequency 正确 (执行了 {island_ga.migrations_performed} 次迁移)")
    
    print("  测试 num_migrants...")
    for num_migrants in [1, 2, 3]:
        island_ga = pygad.IslandGA(
            num_islands=3,
            num_generations=10,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=num_genes,
            migration_frequency=10,
            num_migrants=num_migrants,
            migration_topology='fully_connected',
            random_seed=random_seed,
            suppress_warnings=True
        )
        island_ga.run()
        assert island_ga.generations_completed == 10
        print(f"    ✓ num_migrants={num_migrants} 正常工作")

def test_migrant_selection_and_replacement():
    """
    测试迁移个体选择策略和替换策略。
    """
    print("\n测试: 迁移选择和替换策略...")
    
    strategies = [
        ('best', 'worst'),
        ('best', 'random'),
        ('random', 'worst'),
        ('random', 'random'),
    ]
    
    for migrant_selection, replacement in strategies:
        print(f"  测试 migrant_selection='{migrant_selection}', replacement='{replacement}'...")
        island_ga = pygad.IslandGA(
            num_islands=3,
            num_generations=10,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=num_genes,
            migration_frequency=5,
            num_migrants=1,
            migration_topology='fully_connected',
            migrant_selection=migrant_selection,
            replacement=replacement,
            random_seed=random_seed,
            suppress_warnings=True
        )
        island_ga.run()
        
        assert island_ga.generations_completed == 10
        solution, fitness, _, _ = island_ga.best_solution()
        assert solution is not None
        assert fitness is not None
        print(f"    ✓ 策略组合正常工作")

def test_parameter_validation():
    """
    测试参数验证。
    """
    print("\n测试: 参数验证...")
    
    print("  测试无效的岛屿数量...")
    try:
        pygad.IslandGA(
            num_islands=1,
            num_generations=10,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=num_genes,
            suppress_warnings=True
        )
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "至少为 2" in str(e) or "num_islands" in str(e)
        print("    ✓ 岛屿数量验证正确")
    
    print("  测试无效的迁移频率...")
    try:
        pygad.IslandGA(
            num_islands=3,
            num_generations=10,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=num_genes,
            migration_frequency=0,
            suppress_warnings=True
        )
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "至少为 1" in str(e) or "migration_frequency" in str(e)
        print("    ✓ 迁移频率验证正确")
    
    print("  测试无效的迁移数量...")
    try:
        pygad.IslandGA(
            num_islands=3,
            num_generations=10,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=num_genes,
            num_migrants=0,
            suppress_warnings=True
        )
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "至少为 1" in str(e) or "num_migrants" in str(e)
        print("    ✓ 迁移数量验证正确")
    
    print("  测试无效的拓扑结构...")
    try:
        pygad.IslandGA(
            num_islands=3,
            num_generations=10,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=10,
            num_genes=num_genes,
            migration_topology='invalid_topology',
            suppress_warnings=True
        )
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "不支持的拓扑结构" in str(e) or "migration_topology" in str(e)
        print("    ✓ 拓扑结构验证正确")

def test_island_params():
    """
    测试为不同岛屿设置不同参数。
    """
    print("\n测试: 岛屿自定义参数...")
    
    island_params = [
        {'mutation_percent_genes': 5},
        {'mutation_percent_genes': 15},
        {'mutation_percent_genes': 25},
    ]
    
    island_ga = pygad.IslandGA(
        num_islands=3,
        num_generations=10,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_genes,
        migration_frequency=10,
        num_migrants=1,
        island_params=island_params,
        random_seed=random_seed,
        suppress_warnings=True
    )
    
    assert len(island_ga.islands) == 3
    
    island_ga.run()
    assert island_ga.generations_completed == 10
    print("  ✓ 岛屿自定义参数正常工作")

def test_initial_populations():
    """
    测试自定义初始种群。
    """
    print("\n测试: 自定义初始种群...")
    
    initial_populations = [
        numpy.random.rand(10, num_genes) * 10,
        numpy.random.rand(10, num_genes) * 20,
        numpy.random.rand(10, num_genes) * 30,
    ]
    
    island_ga = pygad.IslandGA(
        num_islands=3,
        num_generations=5,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_genes,
        initial_populations=initial_populations,
        migration_frequency=10,
        num_migrants=1,
        random_seed=random_seed,
        suppress_warnings=True
    )
    
    assert len(island_ga.islands) == 3
    
    island_ga.run()
    assert island_ga.generations_completed == 5
    print("  ✓ 自定义初始种群正常工作")

def test_best_solution():
    """
    测试 best_solution() 方法。
    """
    print("\n测试: best_solution() 方法...")
    
    island_ga = pygad.IslandGA(
        num_islands=3,
        num_generations=15,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_genes,
        migration_frequency=5,
        num_migrants=1,
        random_seed=random_seed,
        suppress_warnings=True
    )
    island_ga.run()
    
    solution, fitness, island_idx, sol_idx = island_ga.best_solution()
    
    assert solution is not None
    assert isinstance(fitness, (int, float, numpy.integer, numpy.floating))
    assert 0 <= island_idx < island_ga.num_islands
    assert 0 <= sol_idx < sol_per_pop
    
    island_best_solution, island_best_fitness, _ = island_ga.islands[island_idx].best_solution()
    assert abs(fitness - island_best_fitness) < 1e-10, \
        "全局最佳适应度应与所在岛屿的最佳适应度一致"
    
    print("  ✓ best_solution() 返回正确的结果")

def test_get_island():
    """
    测试 get_island() 方法。
    """
    print("\n测试: get_island() 方法...")
    
    island_ga = pygad.IslandGA(
        num_islands=3,
        num_generations=5,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_genes,
        migration_frequency=10,
        num_migrants=1,
        random_seed=random_seed,
        suppress_warnings=True
    )
    
    for i in range(3):
        island = island_ga.get_island(i)
        assert island is not None
        assert isinstance(island, pygad.GA)
    
    try:
        island_ga.get_island(100)
        assert False, "应该抛出异常"
    except Exception:
        pass
    
    print("  ✓ get_island() 方法正常工作")

def test_get_all_islands_fitness():
    """
    测试 get_all_islands_fitness() 方法。
    """
    print("\n测试: get_all_islands_fitness() 方法...")
    
    island_ga = pygad.IslandGA(
        num_islands=3,
        num_generations=10,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_genes,
        migration_frequency=5,
        num_migrants=1,
        random_seed=random_seed,
        suppress_warnings=True
    )
    island_ga.run()
    
    all_fitness = island_ga.get_all_islands_fitness()
    
    assert len(all_fitness) == island_ga.num_islands
    for fitness_arr in all_fitness:
        assert len(fitness_arr) == sol_per_pop
    
    print("  ✓ get_all_islands_fitness() 返回正确的结果")

def test_save_and_load():
    """
    测试保存和加载 IslandGA 实例。
    """
    print("\n测试: 保存和加载...")
    
    island_ga = pygad.IslandGA(
        num_islands=3,
        num_generations=10,
        num_parents_mating=5,
        fitness_func=fitness_func,
        sol_per_pop=10,
        num_genes=num_genes,
        migration_frequency=5,
        num_migrants=1,
        random_seed=random_seed,
        suppress_warnings=True
    )
    island_ga.run()
    
    solution_before, fitness_before, island_idx_before, _ = island_ga.best_solution()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, 'test_island_ga')
        
        island_ga.save(filename)
        
        loaded_ga = pygad.load_islandga(filename)
        
        assert loaded_ga.num_islands == island_ga.num_islands
        assert loaded_ga.generations_completed == island_ga.generations_completed
        assert loaded_ga.migrations_performed == island_ga.migrations_performed
        
        solution_after, fitness_after, island_idx_after, _ = loaded_ga.best_solution()
        assert abs(fitness_before - fitness_after) < 1e-10
    
    print("  ✓ 保存和加载正常工作")

def test_ring_topology_paths():
    """
    验证环形拓扑的迁移路径。
    """
    print("\n测试: 环形拓扑迁移路径...")
    
    island_ga = pygad.IslandGA(
        num_islands=4,
        num_generations=5,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=8,
        num_genes=num_genes,
        migration_frequency=10,
        num_migrants=1,
        migration_topology='ring',
        random_seed=random_seed,
        suppress_warnings=True
    )
    
    assert island_ga.migration_paths[0] == [1]
    assert island_ga.migration_paths[1] == [2]
    assert island_ga.migration_paths[2] == [3]
    assert island_ga.migration_paths[3] == [0]
    
    print("  ✓ 环形拓扑路径正确 (0→1→2→3→0)")

def test_star_topology_paths():
    """
    验证星形拓扑的迁移路径。
    """
    print("\n测试: 星形拓扑迁移路径...")
    
    island_ga = pygad.IslandGA(
        num_islands=4,
        num_generations=5,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=8,
        num_genes=num_genes,
        migration_frequency=10,
        num_migrants=1,
        migration_topology='star',
        random_seed=random_seed,
        suppress_warnings=True
    )
    
    assert island_ga.migration_paths[0] == [1, 2, 3]
    assert island_ga.migration_paths[1] == [0]
    assert island_ga.migration_paths[2] == [0]
    assert island_ga.migration_paths[3] == [0]
    
    print("  ✓ 星形拓扑路径正确 (中心=0, 其他连接到中心)")

def test_fully_connected_topology_paths():
    """
    验证全连接拓扑的迁移路径。
    """
    print("\n测试: 全连接拓扑迁移路径...")
    
    island_ga = pygad.IslandGA(
        num_islands=4,
        num_generations=5,
        num_parents_mating=2,
        fitness_func=fitness_func,
        sol_per_pop=8,
        num_genes=num_genes,
        migration_frequency=10,
        num_migrants=1,
        migration_topology='fully_connected',
        random_seed=random_seed,
        suppress_warnings=True
    )
    
    assert island_ga.migration_paths[0] == [1, 2, 3]
    assert island_ga.migration_paths[1] == [0, 2, 3]
    assert island_ga.migration_paths[2] == [0, 1, 3]
    assert island_ga.migration_paths[3] == [0, 1, 2]
    
    print("  ✓ 全连接拓扑路径正确 (每个岛屿连接到所有其他岛屿)")

if __name__ == "__main__":
    print("=" * 60)
    print("IslandGA 单元测试")
    print("=" * 60)
    
    test_migration_updates_fitness()
    test_topologies()
    test_migration_parameters()
    test_migrant_selection_and_replacement()
    test_parameter_validation()
    test_island_params()
    test_initial_populations()
    test_best_solution()
    test_get_island()
    test_get_all_islands_fitness()
    test_save_and_load()
    test_ring_topology_paths()
    test_star_topology_paths()
    test_fully_connected_topology_paths()
    
    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
