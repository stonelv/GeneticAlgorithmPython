import numpy
import random
from pygad import GA

class IslandGA:
    """
    Island Model Genetic Algorithm (岛模型遗传算法)
    
    多个子种群（岛屿）各自独立进化，每隔几代进行个体迁移。
    
    主要特性:
    - 多个独立的 GA 实例（岛屿）并行进化
    - 可配置的迁移频率：每隔多少代进行一次迁移
    - 可配置的迁移数量：每次迁移多少个个体
    - 多种拓扑结构：全连接、环形、随机、星形
    - 可配置的迁移策略：选择哪些个体迁移，如何替换目标岛屿的个体
    """
    
    supported_topologies = ['fully_connected', 'ring', 'random', 'star']
    supported_migrant_selection = ['best', 'random']
    supported_replacement = ['worst', 'random']
    
    def __init__(self,
                 num_islands,
                 num_generations,
                 num_parents_mating,
                 fitness_func,
                 sol_per_pop,
                 num_genes,
                 migration_frequency=10,
                 num_migrants=1,
                 migration_topology='fully_connected',
                 migrant_selection='best',
                 replacement='worst',
                 initial_populations=None,
                 island_params=None,
                 parallel_islands=False,
                 random_seed=None,
                 **kwargs):
        """
        Parameters
        ----------
        num_islands : int
            岛屿的数量。
        
        num_generations : int
            每个岛屿进化的总代数。
        
        num_parents_mating : int
            每个岛屿中被选为父母进行交配的解决方案数量。
        
        fitness_func : callable
            适应度函数，与 pygad.GA 相同。
        
        sol_per_pop : int
            每个岛屿种群中的解决方案数量。
        
        num_genes : int
            每个解决方案中的基因数量。
        
        migration_frequency : int, optional
            迁移频率，表示每隔多少代进行一次迁移。默认为 10。
        
        num_migrants : int, optional
            每次迁移时从每个岛屿迁出的个体数量。默认为 1。
        
        migration_topology : str, optional
            迁移拓扑结构，决定岛屿之间如何连接。
            可选值:
            - 'fully_connected': 全连接，每个岛屿与所有其他岛屿相连
            - 'ring': 环形，每个岛屿只与相邻的下一个岛屿相连
            - 'random': 随机，每次迁移时随机选择目标岛屿
            - 'star': 星形，一个中心岛屿与其他所有岛屿相连
            默认为 'fully_connected'。
        
        migrant_selection : str, optional
            迁移个体选择策略，决定选择哪些个体迁出。
            可选值:
            - 'best': 选择适应度最好的个体
            - 'random': 随机选择个体
            默认为 'best'。
        
        replacement : str, optional
            替换策略，决定如何处理迁入的个体。
            可选值:
            - 'worst': 替换适应度最差的个体
            - 'random': 随机替换个体
            默认为 'worst'。
        
        initial_populations : list, optional
            每个岛屿的初始种群列表。如果为 None，则自动生成。
            列表长度必须等于 num_islands。
        
        island_params : list, optional
            每个岛屿的自定义参数字典列表。如果为 None，则所有岛屿使用相同参数。
            可用于为不同岛屿设置不同的进化参数（如不同的变异率等）。
        
        parallel_islands : bool, optional
            是否并行运行岛屿。默认为 False。
            注意：并行运行可能会有 GIL 限制，对于计算密集型适应度函数效果更好。
        
        random_seed : int, optional
            随机种子，用于可复现的结果。默认为 None。
        
        **kwargs
            其他传递给 pygad.GA 的参数，如:
            - init_range_low, init_range_high
            - gene_type
            - parent_selection_type
            - keep_parents
            - keep_elitism
            - crossover_type
            - mutation_type
            - 等等
        """
        self.num_islands = num_islands
        self.num_generations = num_generations
        self.migration_frequency = migration_frequency
        self.num_migrants = num_migrants
        self.migration_topology = migration_topology
        self.migrant_selection = migrant_selection
        self.replacement = replacement
        self.parallel_islands = parallel_islands
        
        self.generations_completed = 0
        self.migrations_performed = 0
        self.best_solutions_fitness = []
        self.best_solutions = []
        
        if random_seed is not None:
            numpy.random.seed(random_seed)
            random.seed(random_seed)
        
        self._validate_parameters()
        
        self.islands = []
        
        for island_idx in range(num_islands):
            island_kwargs = kwargs.copy()
            
            if island_params is not None and island_idx < len(island_params):
                island_kwargs.update(island_params[island_idx])
            
            island_initial_pop = None
            if initial_populations is not None and island_idx < len(initial_populations):
                island_initial_pop = initial_populations[island_idx]
            
            ga_instance = GA(
                num_generations=0,
                num_parents_mating=num_parents_mating,
                fitness_func=fitness_func,
                initial_population=island_initial_pop,
                sol_per_pop=sol_per_pop,
                num_genes=num_genes,
                **island_kwargs
            )
            
            self.islands.append(ga_instance)
        
        self._setup_migration_paths()
    
    def _validate_parameters(self):
        """验证参数的有效性"""
        if self.num_islands < 2:
            raise ValueError(f"岛屿数量 num_islands 必须至少为 2，但当前值为 {self.num_islands}")
        
        if self.migration_frequency < 1:
            raise ValueError(f"迁移频率 migration_frequency 必须至少为 1，但当前值为 {self.migration_frequency}")
        
        if self.num_migrants < 1:
            raise ValueError(f"迁移数量 num_migrants 必须至少为 1，但当前值为 {self.num_migrants}")
        
        if self.migration_topology not in self.supported_topologies:
            raise ValueError(f"不支持的拓扑结构 '{self.migration_topology}'。支持的拓扑结构: {self.supported_topologies}")
        
        if self.migrant_selection not in self.supported_migrant_selection:
            raise ValueError(f"不支持的迁移选择策略 '{self.migrant_selection}'。支持的策略: {self.supported_migrant_selection}")
        
        if self.replacement not in self.supported_replacement:
            raise ValueError(f"不支持的替换策略 '{self.replacement}'。支持的策略: {self.supported_replacement}")
    
    def _setup_migration_paths(self):
        """根据拓扑结构设置迁移路径"""
        if self.migration_topology == 'fully_connected':
            self.migration_paths = {i: [j for j in range(self.num_islands) if j != i] 
                                     for i in range(self.num_islands)}
        
        elif self.migration_topology == 'ring':
            self.migration_paths = {i: [(i + 1) % self.num_islands] 
                                     for i in range(self.num_islands)}
        
        elif self.migration_topology == 'star':
            self.migration_paths = {0: [i for i in range(1, self.num_islands)]}
            for i in range(1, self.num_islands):
                self.migration_paths[i] = [0]
        
        elif self.migration_topology == 'random':
            pass
    
    def _get_destination_islands(self, source_island_idx):
        """获取源岛屿的目标岛屿列表"""
        if self.migration_topology == 'random':
            possible_destinations = [i for i in range(self.num_islands) if i != source_island_idx]
            num_destinations = max(1, len(possible_destinations) // 2)
            return random.sample(possible_destinations, min(num_destinations, len(possible_destinations)))
        else:
            return self.migration_paths.get(source_island_idx, [])
    
    def _select_migrants(self, ga_instance, fitness_values):
        """选择要迁移的个体"""
        if self.migrant_selection == 'best':
            sorted_indices = numpy.argsort(fitness_values)[::-1]
            migrant_indices = sorted_indices[:self.num_migrants]
        
        else:
            migrant_indices = random.sample(range(len(fitness_values)), 
                                             min(self.num_migrants, len(fitness_values)))
        
        migrants = ga_instance.population[migrant_indices].copy()
        migrant_fitness = fitness_values[migrant_indices].copy()
        
        return migrants, migrant_fitness, migrant_indices
    
    def _replace_individuals(self, ga_instance, fitness_values, new_individuals, new_fitness):
        """用迁入的个体替换目标岛屿中的个体"""
        if self.replacement == 'worst':
            sorted_indices = numpy.argsort(fitness_values)
            replace_indices = sorted_indices[:len(new_individuals)]
        
        else:
            replace_indices = random.sample(range(len(fitness_values)), 
                                             min(len(new_individuals), len(fitness_values)))
        
        for i, idx in enumerate(replace_indices):
            if i < len(new_individuals):
                ga_instance.population[idx] = new_individuals[i].copy()
    
    def _perform_migration(self):
        """执行迁移操作"""
        all_migrants = []
        modified_islands = set()
        
        for island_idx, ga_instance in enumerate(self.islands):
            fitness_values = ga_instance.last_generation_fitness
            migrants, migrant_fitness, migrant_indices = self._select_migrants(ga_instance, fitness_values)
            destinations = self._get_destination_islands(island_idx)
            
            for dest_idx in destinations:
                modified_islands.add(dest_idx)
                for migrant_idx, migrant in enumerate(migrants):
                    all_migrants.append({
                        'source': island_idx,
                        'destination': dest_idx,
                        'individual': migrant.copy(),
                        'fitness': migrant_fitness[migrant_idx]
                    })
        
        for migration in all_migrants:
            dest_idx = migration['destination']
            dest_ga = self.islands[dest_idx]
            dest_fitness = dest_ga.last_generation_fitness
            
            self._replace_individuals(
                dest_ga, 
                dest_fitness, 
                [migration['individual']],
                [migration['fitness']]
            )
        
        for island_idx in modified_islands:
            self.islands[island_idx].last_generation_fitness = self.islands[island_idx].cal_pop_fitness()
        
        self.migrations_performed += 1
    
    def _run_single_island(self, ga_instance, generations):
        """运行单个岛屿的进化"""
        ga_instance.num_generations = generations
        ga_instance.run()
        return ga_instance
    
    def run(self):
        """
        运行岛模型遗传算法。
        
        主要流程:
        1. 初始化所有岛屿
        2. 每隔 migration_frequency 代执行一次迁移
        3. 重复直到完成 num_generations 代
        """
        for island_idx, ga_instance in enumerate(self.islands):
            if ga_instance.generations_completed == 0:
                ga_instance.initialize_population(
                    allow_duplicate_genes=ga_instance.allow_duplicate_genes,
                    gene_type=ga_instance.gene_type,
                    gene_constraint=ga_instance.gene_constraint
                )
                ga_instance.last_generation_fitness = ga_instance.cal_pop_fitness()
        
        while self.generations_completed < self.num_generations:
            generations_to_run = min(
                self.migration_frequency,
                self.num_generations - self.generations_completed
            )
            
            if self.parallel_islands:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_islands) as executor:
                    futures = [executor.submit(self._run_single_island, ga, generations_to_run)
                               for ga in self.islands]
                    self.islands = [future.result() for future in futures]
            else:
                for ga_instance in self.islands:
                    self._run_single_island(ga_instance, generations_to_run)
            
            self.generations_completed += generations_to_run
            
            if self.generations_completed < self.num_generations and generations_to_run == self.migration_frequency:
                self._perform_migration()
            
            self._update_best_solutions()
        
        self._finalize()
    
    def _update_best_solutions(self):
        """更新全局最佳解决方案"""
        global_best_fitness = -float('inf')
        global_best_solution = None
        
        for ga_instance in self.islands:
            best_solution, best_fitness, _ = ga_instance.best_solution(
                pop_fitness=ga_instance.last_generation_fitness
            )
            if best_fitness > global_best_fitness:
                global_best_fitness = best_fitness
                global_best_solution = best_solution
        
        self.best_solutions_fitness.append(global_best_fitness)
        if global_best_solution is not None:
            self.best_solutions.append(list(global_best_solution))
    
    def _finalize(self):
        """完成后的收尾工作"""
        for ga_instance in self.islands:
            ga_instance.run_completed = True
        
        self.best_solutions = numpy.array(self.best_solutions)
    
    def best_solution(self):
        """
        返回所有岛屿中的最佳解决方案。
        
        Returns
        -------
        tuple
            - best_solution: 最佳解决方案
            - best_fitness: 最佳适应度
            - best_island_idx: 最佳解决方案所在的岛屿索引
            - best_solution_idx: 最佳解决方案在其岛屿中的索引
        """
        global_best_fitness = -float('inf')
        global_best_solution = None
        best_island_idx = -1
        best_solution_idx = -1
        
        for island_idx, ga_instance in enumerate(self.islands):
            best_solution, best_fitness, sol_idx = ga_instance.best_solution(
                pop_fitness=ga_instance.last_generation_fitness
            )
            if best_fitness > global_best_fitness:
                global_best_fitness = best_fitness
                global_best_solution = best_solution
                best_island_idx = island_idx
                best_solution_idx = sol_idx
        
        return global_best_solution, global_best_fitness, best_island_idx, best_solution_idx
    
    def plot_fitness(self, title="Island GA - Fitness over Generations"):
        """
        绘制所有岛屿以及全局最佳适应度的变化曲线。
        
        Parameters
        ----------
        title : str, optional
            图表标题，默认为 "Island GA - Fitness over Generations"
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        for island_idx, ga_instance in enumerate(self.islands):
            fitness_history = ga_instance.best_solutions_fitness
            plt.plot(fitness_history, label=f'Island {island_idx + 1}', alpha=0.6)
        
        plt.plot(self.best_solutions_fitness, 'k-', linewidth=2, label='Global Best')
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def save(self, filename):
        """
        保存 IslandGA 实例到文件。
        
        Parameters
        ----------
        filename : str
            文件名（不含扩展名）
        """
        import cloudpickle
        
        with open(filename + ".pkl", 'wb') as file:
            cloudpickle.dump(self, file)
    
    def get_island(self, island_idx):
        """
        获取指定索引的岛屿 GA 实例。
        
        Parameters
        ----------
        island_idx : int
            岛屿索引（从 0 开始）
        
        Returns
        -------
        pygad.GA
            指定岛屿的 GA 实例
        """
        if 0 <= island_idx < self.num_islands:
            return self.islands[island_idx]
        raise Index(f"岛屿索引 {island_idx} 超出范围 (0-{self.num_islands-1})")
    
    def get_all_islands_fitness(self):
        """
        获取所有岛屿当前种群的适应度值。
        
        Returns
        -------
        list
            每个岛屿的适应度值列表
        """
        return [island.last_generation_fitness for island in self.islands]


def load_islandga(filename):
    """
    从文件加载 IslandGA 实例。
    
    Parameters
    ----------
    filename : str
        文件名（不含扩展名）
    
    Returns
    -------
    IslandGA
        加载的 IslandGA 实例
    """
    import cloudpickle
    
    try:
        with open(filename + ".pkl", 'rb') as file:
            ga_in = cloudpickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error reading the file {filename}. Please check your inputs.")
    except:
        raise BaseException("Error loading the file.")
    return ga_in
