import numpy
import cloudpickle
import warnings
import concurrent.futures
import logging


class ES:
    """
    Evolution Strategy (ES) 类，支持 (μ+λ)-ES 和 (μ,λ)-ES，带自适应步长σ。
    
    与遗传算法不同，进化策略更适合连续优化问题，通过自适应步长来平衡探索与利用。
    
    Parameters
    ----------
    num_generations : int
        进化代数。
        
    mu : int
        父代种群大小 (μ)。从 μ 个父代中产生 λ 个子代。
        
    lambda_ : int
        子代种群大小 (λ)。通常 λ > μ。
        
    fitness_func : callable
        适应度函数。签名为 `fitness_func(es_instance, solution, solution_idx)`，
        返回适应度值（最大化问题）。
        
    initial_population : numpy.ndarray, optional
        用户定义的初始种群。形状为 (mu, num_genes)。
        
    sol_per_pop : int, optional
        种群大小。如果提供了 initial_population，则无需此参数。
        
    num_genes : int, optional
        每个解的基因数。如果提供了 initial_population，则无需此参数。
        
    init_range_low : float, optional
        初始种群的下限，默认 -4。
        
    init_range_high : float, optional
        初始种群的上限，默认 4。
        
    gene_type : type, optional
        基因类型，默认 float。
        
    selection_type : str, optional
        选择类型：
        - "plus": (μ+λ)-ES，从父代和子代中选择
        - "comma": (μ,λ)-ES，仅从子代中选择
        默认 "plus"。
        
    recombination_type : str, optional
        重组类型：
        - "none": 无重组，直接复制父代
        - "intermediate": 中间重组（加权平均）
        - "discrete": 离散重组（随机选择）
        默认 "intermediate"。
        
    recombination_probability : float, optional
        每个基因进行重组的概率，默认 1.0。
        
    mutation_type : str, optional
        变异类型：
        - "gaussian": 高斯变异，x' = x + σ * N(0, 1)
        默认 "gaussian"。
        
    sigma : float or numpy.ndarray, optional
        初始步长σ。可以是：
        - 单个浮点数：所有基因使用相同σ
        - 形状为 (num_genes,) 的数组：每个基因独立σ
        默认 0.1。
        
    sigma_adaptation : str, optional
        步长自适应方法：
        - "none": 固定σ
        - "1/5-rule": 1/5成功规则
        - "log-normal": 对数正态变异（σ自身变异）
        - "cumulative": 累积步长自适应 (CSA)
        默认 "log-normal"。
        
    tau : float, optional
        对数正态变异的学习率系数。σ的变异为：
        σ' = σ * exp(tau * N(0, 1))
        其中 tau ≈ 1 / sqrt(n)，n为基因数。
        如果为 None，将自动计算。
        
    tau_prime : float, optional
        全局学习率，用于多步长情况。如果为 None，将自动计算。
        
    one_fifth_c : float, optional
        1/5规则的调整系数 c。σ更新规则：
        - 如果成功率 > 1/5: σ *= c
        - 如果成功率 < 1/5: σ /= c
        默认 0.85（常用值在 0.8~0.9 之间）。
        
    one_fifth_g : int, optional
        1/5规则的更新频率（每g代更新一次），默认 1。
        
    gene_space : list, optional
        定义每个基因的可能取值范围。
        
    gene_constraint : list, optional
        定义每个基因的约束函数。
        
    sample_size : int, optional
        满足约束时的采样大小，默认 100。
        
    on_start : callable, optional
        进化开始前调用的回调函数。
        
    on_fitness : callable, optional
        计算完适应度后调用的回调函数。
        
    on_recombination : callable, optional
        重组后调用的回调函数。
        
    on_mutation : callable, optional
        变异后调用的回调函数。
        
    on_generation : callable, optional
        每代结束后调用的回调函数。返回 "stop" 可提前终止。
        
    on_stop : callable, optional
        进化结束时调用的回调函数。
        
    save_best_solutions : bool, optional
        是否保存每代的最佳解，默认 False。
        
    save_solutions : bool, optional
        是否保存所有解，默认 False。
        
    suppress_warnings : bool, optional
        是否抑制警告，默认 False。
        
    stop_criteria : list, optional
        停止条件，如 ["reach_100", "saturate_10"]。
        
    parallel_processing : int or tuple, optional
        并行处理设置。
        
    random_seed : int, optional
        随机种子。
        
    logger : logging.Logger, optional
        日志记录器。
    """

    supported_int_types = [int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                           numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
                           object]
    supported_float_types = [float, numpy.float16, numpy.float32, numpy.float64,
                             object]
    supported_int_float_types = supported_int_types + supported_float_types

    def __init__(self,
                 num_generations,
                 mu,
                 lambda_,
                 fitness_func,
                 fitness_batch_size=None,
                 initial_population=None,
                 sol_per_pop=None,
                 num_genes=None,
                 init_range_low=-4,
                 init_range_high=4,
                 gene_type=float,
                 selection_type="plus",
                 recombination_type="intermediate",
                 recombination_probability=1.0,
                 mutation_type="gaussian",
                 sigma=0.1,
                 sigma_adaptation="log-normal",
                 tau=None,
                 tau_prime=None,
                 one_fifth_c=0.85,
                 one_fifth_g=1,
                 gene_space=None,
                 gene_constraint=None,
                 sample_size=100,
                 allow_duplicate_genes=True,
                 on_start=None,
                 on_fitness=None,
                 on_recombination=None,
                 on_mutation=None,
                 on_generation=None,
                 on_stop=None,
                 save_best_solutions=False,
                 save_solutions=False,
                 suppress_warnings=False,
                 stop_criteria=None,
                 parallel_processing=None,
                 random_seed=None,
                 logger=None):

        self._validate_selection_type(selection_type)
        self._validate_recombination_type(recombination_type)
        self._validate_mutation_type(mutation_type)
        self._validate_sigma_adaptation(sigma_adaptation)

        self.num_generations = num_generations
        self.mu = mu
        self.lambda_ = lambda_
        self.fitness_func = fitness_func
        self.fitness_batch_size = fitness_batch_size
        self.initial_population = initial_population
        self.sol_per_pop = sol_per_pop
        self.num_genes = num_genes
        self.init_range_low = init_range_low
        self.init_range_high = init_range_high
        self.gene_type = gene_type
        self.selection_type = selection_type
        self.recombination_type = recombination_type
        self.recombination_probability = recombination_probability
        self.mutation_type = mutation_type
        self.sigma = sigma
        self.sigma_adaptation = sigma_adaptation
        self.tau = tau
        self.tau_prime = tau_prime
        self.one_fifth_c = one_fifth_c
        self.one_fifth_g = one_fifth_g
        self.gene_space = gene_space
        self.gene_constraint = gene_constraint
        self.sample_size = sample_size
        self.allow_duplicate_genes = allow_duplicate_genes
        self.on_start = on_start
        self.on_fitness = on_fitness
        self.on_recombination = on_recombination
        self.on_mutation = on_mutation
        self.on_generation = on_generation
        self.on_stop = on_stop
        self.save_best_solutions = save_best_solutions
        self.save_solutions = save_solutions
        self.suppress_warnings = suppress_warnings
        self.stop_criteria = stop_criteria
        self.parallel_processing = parallel_processing
        self.random_seed = random_seed
        self.logger = logger

        self._setup_logger()
        self._setup_random_seed()
        self._validate_parameters()
        self._initialize_population_and_sigma()
        self._initialize_learning_rates()
        self._initialize_attributes()

    def _setup_logger(self):
        if self.logger is None:
            self.logger = logging.getLogger("pygad_es")
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)

    def _setup_random_seed(self):
        if self.random_seed is not None:
            numpy.random.seed(self.random_seed)
            import random
            random.seed(self.random_seed)

    def _validate_selection_type(self, selection_type):
        valid_types = ["plus", "comma"]
        if selection_type not in valid_types:
            raise ValueError(f"selection_type must be one of {valid_types}, got {selection_type}")

    def _validate_recombination_type(self, recombination_type):
        valid_types = ["none", "intermediate", "discrete"]
        if recombination_type not in valid_types:
            raise ValueError(f"recombination_type must be one of {valid_types}, got {recombination_type}")

    def _validate_mutation_type(self, mutation_type):
        valid_types = ["gaussian"]
        if mutation_type not in valid_types:
            raise ValueError(f"mutation_type must be one of {valid_types}, got {mutation_type}")

    def _validate_sigma_adaptation(self, sigma_adaptation):
        valid_types = ["none", "1/5-rule", "log-normal", "cumulative"]
        if sigma_adaptation not in valid_types:
            raise ValueError(f"sigma_adaptation must be one of {valid_types}, got {sigma_adaptation}")

    def _validate_parameters(self):
        if self.initial_population is not None:
            self.initial_population = numpy.asarray(self.initial_population)
            if self.initial_population.ndim != 2:
                raise ValueError(f"initial_population must be 2D, got {self.initial_population.ndim}D")
            if self.initial_population.shape[0] != self.mu:
                raise ValueError(f"initial_population first dimension ({self.initial_population.shape[0]}) must equal mu ({self.mu})")
            self.num_genes = self.initial_population.shape[1]
            self.sol_per_pop = self.mu
        else:
            if self.num_genes is None or self.sol_per_pop is None:
                raise ValueError("Either initial_population or both num_genes and sol_per_pop must be provided")
            if self.sol_per_pop != self.mu:
                if not self.suppress_warnings:
                    warnings.warn(f"sol_per_pop ({self.sol_per_pop}) != mu ({self.mu}). Using mu as sol_per_pop.")
                self.sol_per_pop = self.mu

        if not callable(self.fitness_func):
            raise TypeError("fitness_func must be callable")

        if self.mu <= 0:
            raise ValueError(f"mu must be positive, got {self.mu}")
        if self.lambda_ <= 0:
            raise ValueError(f"lambda_ must be positive, got {self.lambda_}")

        if self.selection_type == "comma" and self.lambda_ < self.mu:
            raise ValueError(f"For (μ,λ)-ES, lambda_ ({self.lambda_}) must be >= mu ({self.mu})")

        self.valid_parameters = True

    def _initialize_population_and_sigma(self):
        if self.initial_population is not None:
            self.population = self.initial_population.copy()
        else:
            self.population = numpy.random.uniform(
                low=self.init_range_low,
                high=self.init_range_high,
                size=(self.mu, self.num_genes)
            )

        self.population = self._apply_gene_type(self.population)

        if isinstance(self.sigma, (int, float)):
            self.population_sigma = numpy.full((self.mu, self.num_genes), float(self.sigma))
        else:
            sigma_arr = numpy.asarray(self.sigma)
            if sigma_arr.ndim == 0:
                self.population_sigma = numpy.full((self.mu, self.num_genes), float(sigma_arr))
            elif sigma_arr.ndim == 1:
                if len(sigma_arr) != self.num_genes:
                    raise ValueError(f"sigma array length ({len(sigma_arr)}) must match num_genes ({self.num_genes})")
                self.population_sigma = numpy.tile(sigma_arr, (self.mu, 1))
            elif sigma_arr.ndim == 2:
                if sigma_arr.shape != (self.mu, self.num_genes):
                    raise ValueError(f"sigma 2D array shape ({sigma_arr.shape}) must match (mu, num_genes) = ({self.mu}, {self.num_genes})")
                self.population_sigma = sigma_arr.copy()
            else:
                raise ValueError(f"sigma must be scalar, 1D, or 2D array, got {sigma_arr.ndim}D")

        self.population_sigma = self.population_sigma.astype(float)

    def _initialize_learning_rates(self):
        if self.sigma_adaptation in ["log-normal", "cumulative"]:
            n = self.num_genes
            if self.tau is None:
                self.tau = 1.0 / numpy.sqrt(2 * numpy.sqrt(n))
            if self.tau_prime is None:
                self.tau_prime = 1.0 / numpy.sqrt(2 * n)

        if self.sigma_adaptation == "cumulative":
            self._path_length = numpy.zeros(self.num_genes)
            self._path_sigma = numpy.zeros(self.num_genes)
            self._cumulative_generation = 0

        self._successful_mutations = []

    def _initialize_attributes(self):
        self.generations_completed = 0
        self.run_completed = False
        self.best_solutions = []
        self.best_solutions_fitness = []
        self.solutions = []
        self.solutions_fitness = []
        self.last_generation_fitness = None
        self.previous_generation_fitness = None
        self.best_solution_generation = 0
        self._last_parents = None
        self._last_offspring = None
        self._last_offspring_sigma = None

    def _apply_gene_type(self, arr):
        if self.gene_type == int:
            return arr.astype(int)
        elif self.gene_type in self.supported_int_types:
            return arr.astype(self.gene_type)
        elif self.gene_type == float:
            return arr.astype(float)
        elif self.gene_type in self.supported_float_types:
            return arr.astype(self.gene_type)
        else:
            return arr

    def save(self, filename):
        cloudpickle_serialized_object = cloudpickle.dumps(self)
        with open(filename + ".pkl", 'wb') as file:
            file.write(cloudpickle_serialized_object)
            cloudpickle.dump(self, file)

    def cal_pop_fitness(self, population=None):
        if population is None:
            population = self.population

        pop_size = len(population)
        pop_fitness = ["undefined"] * pop_size

        if self.parallel_processing is None:
            for sol_idx, sol in enumerate(population):
                if self.fitness_batch_size in [1, None]:
                    fitness = self.fitness_func(self, sol, sol_idx)
                    if type(fitness) not in self.supported_int_float_types + [list, tuple, numpy.ndarray]:
                        raise ValueError(f"fitness_func returned unexpected type: {type(fitness)}")
                    pop_fitness[sol_idx] = fitness
                else:
                    continue

            if self.fitness_batch_size not in [1, None]:
                solutions_indices = [idx for idx, fit in enumerate(pop_fitness)
                                   if isinstance(fit, str) and fit == "undefined"]
                num_batches = int(numpy.ceil(len(solutions_indices) / self.fitness_batch_size))
                for batch_idx in range(num_batches):
                    batch_first = batch_idx * self.fitness_batch_size
                    batch_last = min(batch_first + self.fitness_batch_size, len(solutions_indices))
                    batch_indices = solutions_indices[batch_first:batch_last]
                    batch_solutions = population[batch_indices]
                    batch_fitness = self.fitness_func(self, batch_solutions, batch_indices)
                    if not isinstance(batch_fitness, (list, tuple, numpy.ndarray)):
                        raise TypeError("Batch fitness must return list/tuple/ndarray")
                    if len(batch_fitness) != len(batch_indices):
                        raise ValueError(f"Batch fitness length mismatch: {len(batch_fitness)} vs {len(batch_indices)}")
                    for idx, fit in zip(batch_indices, batch_fitness):
                        pop_fitness[idx] = fit
        else:
            if self.parallel_processing[0] == "process":
                ExecutorClass = concurrent.futures.ProcessPoolExecutor
            else:
                ExecutorClass = concurrent.futures.ThreadPoolExecutor

            with ExecutorClass(max_workers=self.parallel_processing[1]) as executor:
                solutions_to_submit = []
                solutions_to_submit_indices = []
                for sol_idx, sol in enumerate(population):
                    if isinstance(pop_fitness[sol_idx], str) and pop_fitness[sol_idx] == "undefined":
                        solutions_to_submit.append(sol.copy())
                        solutions_to_submit_indices.append(sol_idx)

                if self.fitness_batch_size in [1, None]:
                    for idx, fit in zip(solutions_to_submit_indices,
                                      executor.map(self.fitness_func,
                                                 [self]*len(solutions_to_submit_indices),
                                                 solutions_to_submit,
                                                 solutions_to_submit_indices)):
                        pop_fitness[idx] = fit
                else:
                    num_batches = int(numpy.ceil(len(solutions_to_submit_indices) / self.fitness_batch_size))
                    batches_solutions = []
                    batches_indices = []
                    for batch_idx in range(num_batches):
                        batch_first = batch_idx * self.fitness_batch_size
                        batch_last = min(batch_first + self.fitness_batch_size, len(solutions_to_submit_indices))
                        batch_indices = solutions_to_submit_indices[batch_first:batch_last]
                        batch_solutions = population[batch_indices]
                        batches_solutions.append(batch_solutions)
                        batches_indices.append(batch_indices)

                    for batch_indices, batch_fitness in zip(batches_indices,
                                                            executor.map(self.fitness_func,
                                                                       [self]*len(batches_indices),
                                                                       batches_solutions,
                                                                       batches_indices)):
                        if len(batch_fitness) != len(batch_indices):
                            raise ValueError(f"Batch fitness length mismatch")
                        for idx, fit in zip(batch_indices, batch_fitness):
                            pop_fitness[idx] = fit

        return numpy.array(pop_fitness)

    def _recombination(self, parent_indices):
        num_children = self.lambda_
        children = numpy.empty((num_children, self.num_genes), dtype=float)
        children_sigma = numpy.empty((num_children, self.num_genes), dtype=float)

        if self.recombination_type == "none":
            for i in range(num_children):
                parent_idx = parent_indices[i % len(parent_indices)]
                children[i] = self.population[parent_idx].copy()
                children_sigma[i] = self.population_sigma[parent_idx].copy()
            return children, children_sigma

        elif self.recombination_type == "intermediate":
            for i in range(num_children):
                p1_idx = parent_indices[i % len(parent_indices)]
                p2_idx = parent_indices[(i + 1) % len(parent_indices)]

                parent1 = self.population[p1_idx]
                parent2 = self.population[p2_idx]
                sigma1 = self.population_sigma[p1_idx]
                sigma2 = self.population_sigma[p2_idx]

                for gene_idx in range(self.num_genes):
                    if numpy.random.random() < self.recombination_probability:
                        alpha = numpy.random.random()
                        children[i, gene_idx] = alpha * parent1[gene_idx] + (1 - alpha) * parent2[gene_idx]
                        children_sigma[i, gene_idx] = alpha * sigma1[gene_idx] + (1 - alpha) * sigma2[gene_idx]
                    else:
                        children[i, gene_idx] = parent1[gene_idx]
                        children_sigma[i, gene_idx] = sigma1[gene_idx]

        elif self.recombination_type == "discrete":
            for i in range(num_children):
                p1_idx = parent_indices[i % len(parent_indices)]
                p2_idx = parent_indices[(i + 1) % len(parent_indices)]

                parent1 = self.population[p1_idx]
                parent2 = self.population[p2_idx]
                sigma1 = self.population_sigma[p1_idx]
                sigma2 = self.population_sigma[p2_idx]

                for gene_idx in range(self.num_genes):
                    if numpy.random.random() < self.recombination_probability:
                        if numpy.random.random() < 0.5:
                            children[i, gene_idx] = parent1[gene_idx]
                            children_sigma[i, gene_idx] = sigma1[gene_idx]
                        else:
                            children[i, gene_idx] = parent2[gene_idx]
                            children_sigma[i, gene_idx] = sigma2[gene_idx]
                    else:
                        children[i, gene_idx] = parent1[gene_idx]
                        children_sigma[i, gene_idx] = sigma1[gene_idx]

        return children, children_sigma

    def _mutation(self, children, children_sigma):
        num_children = len(children)
        mutated_children = children.copy()
        mutated_sigma = children_sigma.copy()

        if self.sigma_adaptation == "log-normal":
            for i in range(num_children):
                N0 = numpy.random.randn()
                N = numpy.random.randn(self.num_genes)

                mutated_sigma[i] = children_sigma[i] * numpy.exp(
                    self.tau_prime * N0 + self.tau * N
                )

                mutated_children[i] = children[i] + mutated_sigma[i] * numpy.random.randn(self.num_genes)

        elif self.sigma_adaptation == "none":
            for i in range(num_children):
                mutated_children[i] = children[i] + children_sigma[i] * numpy.random.randn(self.num_genes)

        elif self.sigma_adaptation == "1/5-rule":
            for i in range(num_children):
                mutated_children[i] = children[i] + children_sigma[i] * numpy.random.randn(self.num_genes)

        elif self.sigma_adaptation == "cumulative":
            for i in range(num_children):
                N0 = numpy.random.randn()
                N = numpy.random.randn(self.num_genes)

                mutated_sigma[i] = children_sigma[i] * numpy.exp(
                    self.tau_prime * N0 + self.tau * N
                )

                mutated_children[i] = children[i] + mutated_sigma[i] * numpy.random.randn(self.num_genes)

        return mutated_children, mutated_sigma

    def _select_parents_for_recombination(self, fitness):
        num_children = self.lambda_
        parent_indices = numpy.random.randint(0, self.mu, size=num_children * 2)
        return parent_indices

    def _selection(self, parents, parents_sigma, parents_fitness,
                   offspring, offspring_sigma, offspring_fitness):
        if self.selection_type == "plus":
            combined = numpy.vstack([parents, offspring])
            combined_sigma = numpy.vstack([parents_sigma, offspring_sigma])
            combined_fitness = numpy.concatenate([parents_fitness, offspring_fitness])

            if self.sigma_adaptation == "1/5-rule":
                parent_best_idx = numpy.argmax(parents_fitness)
                for i in range(self.lambda_):
                    if offspring_fitness[i] > parents_fitness[parent_best_idx]:
                        self._successful_mutations.append(1)
                    else:
                        self._successful_mutations.append(0)

                self._update_sigma_by_one_fifth()

        elif self.selection_type == "comma":
            combined = offspring
            combined_sigma = offspring_sigma
            combined_fitness = offspring_fitness

            if self.sigma_adaptation == "1/5-rule":
                parent_best_idx = numpy.argmax(parents_fitness)
                for i in range(self.lambda_):
                    if offspring_fitness[i] > parents_fitness[parent_best_idx]:
                        self._successful_mutations.append(1)
                    else:
                        self._successful_mutations.append(0)

                self._update_sigma_by_one_fifth()

        sorted_indices = numpy.argsort(combined_fitness)[::-1]
        selected_indices = sorted_indices[:self.mu]

        new_population = combined[selected_indices]
        new_population_sigma = combined_sigma[selected_indices]

        return new_population, new_population_sigma

    def _update_sigma_by_one_fifth(self):
        if len(self._successful_mutations) < self.one_fifth_g * self.lambda_:
            return

        recent = self._successful_mutations[-self.one_fifth_g * self.lambda_:]
        success_rate = sum(recent) / len(recent)

        if success_rate > 0.2:
            self.population_sigma /= self.one_fifth_c
        elif success_rate < 0.2:
            self.population_sigma *= self.one_fifth_c

    def best_solution(self, pop_fitness=None):
        if pop_fitness is None:
            pop_fitness = self.last_generation_fitness
            population = self.population
        else:
            population = self.population

        best_idx = numpy.argmax(pop_fitness)
        best_fitness = pop_fitness[best_idx]
        best_sol = population[best_idx]

        return best_sol, best_fitness, best_idx

    def run(self):
        if self.valid_parameters == False:
            raise Exception("Error calling the run() method: Invalid parameters.")

        if type(self.best_solutions) is numpy.ndarray:
            self.best_solutions = self.best_solutions.tolist()
        if type(self.best_solutions_fitness) is numpy.ndarray:
            self.best_solutions_fitness = list(self.best_solutions_fitness)
        if type(self.solutions) is numpy.ndarray:
            self.solutions = self.solutions.tolist()
        if type(self.solutions_fitness) is numpy.ndarray:
            self.solutions_fitness = list(self.solutions_fitness)

        if self.on_start is not None:
            self.on_start(self)

        stop_run = False

        if self.generations_completed != 0 and type(self.generations_completed) in self.supported_int_types:
            generation_first_idx = self.generations_completed
            generation_last_idx = self.num_generations + self.generations_completed
        else:
            generation_first_idx = 0
            generation_last_idx = self.num_generations

        self.last_generation_fitness = self.cal_pop_fitness(self.population)

        best_solution, best_solution_fitness, best_match_idx = self.best_solution()

        if self.save_best_solutions:
            self.best_solutions.append(list(best_solution))

        for generation in range(generation_first_idx, generation_last_idx):
            self.best_solutions_fitness.append(best_solution_fitness)

            if self.save_solutions:
                pop_as_list = [list(item) for item in self.population]
                self.solutions.extend(pop_as_list)
                self.solutions_fitness.extend(self.last_generation_fitness)

            if self.on_fitness is not None:
                on_fitness_output = self.on_fitness(self, self.last_generation_fitness)
                if on_fitness_output is not None:
                    on_fitness_output = numpy.array(on_fitness_output)
                    if on_fitness_output.shape == self.last_generation_fitness.shape:
                        self.last_generation_fitness = on_fitness_output
                    else:
                        raise ValueError(f"Size mismatch in on_fitness output")

            parent_indices = self._select_parents_for_recombination(self.last_generation_fitness)

            children, children_sigma = self._recombination(parent_indices)

            if self.on_recombination is not None:
                recomb_output = self.on_recombination(self, children, children_sigma)
                if recomb_output is not None:
                    if len(recomb_output) == 2:
                        children, children_sigma = recomb_output

            mutated_children, mutated_sigma = self._mutation(children, children_sigma)

            if self.on_mutation is not None:
                mutate_output = self.on_mutation(self, mutated_children, mutated_sigma)
                if mutate_output is not None:
                    if len(mutate_output) == 2:
                        mutated_children, mutated_sigma = mutate_output

            mutated_children = self._apply_gene_type(mutated_children)

            offspring_fitness = self.cal_pop_fitness(mutated_children)

            self._last_parents = self.population.copy()
            self._last_offspring = mutated_children
            self._last_offspring_sigma = mutated_sigma

            self.previous_generation_fitness = self.last_generation_fitness.copy()

            self.population, self.population_sigma = self._selection(
                self._last_parents, self.population_sigma, self.last_generation_fitness,
                mutated_children, mutated_sigma, offspring_fitness
            )

            self.generations_completed = generation + 1

            self.last_generation_fitness = self.cal_pop_fitness(self.population)

            best_solution, best_solution_fitness, best_match_idx = self.best_solution()

            if self.save_best_solutions:
                self.best_solutions.append(list(best_solution))

            if self.on_generation is not None:
                r = self.on_generation(self)
                if type(r) is str and r.lower() == "stop":
                    self.best_solutions_fitness.append(best_solution_fitness)
                    break

            if self.stop_criteria is not None:
                for criterion in self.stop_criteria:
                    if criterion.startswith("reach_"):
                        target = float(criterion[6:])
                        if type(self.last_generation_fitness[0]) in self.supported_int_float_types:
                            if max(self.last_generation_fitness) >= target:
                                stop_run = True
                                break
                    elif criterion.startswith("saturate_"):
                        gens = int(criterion[9:])
                        if self.generations_completed >= gens:
                            if len(self.best_solutions_fitness) >= gens:
                                old_fit = self.best_solutions_fitness[-gens]
                                new_fit = self.best_solutions_fitness[-1]
                                if isinstance(old_fit, (list, tuple, numpy.ndarray)):
                                    if all(o == n for o, n in zip(old_fit, new_fit)):
                                        stop_run = True
                                        break
                                else:
                                    if old_fit == new_fit:
                                        stop_run = True
                                        break

                if stop_run:
                    break

        if self.save_solutions:
            pop_as_list = [list(item) for item in self.population]
            self.solutions.extend(pop_as_list)
            self.solutions_fitness.extend(self.last_generation_fitness)

        _, best_solution_fitness, _ = self.best_solution()
        self.best_solutions_fitness.append(best_solution_fitness)

        if len(self.best_solutions_fitness) > 0:
            self.best_solution_generation = numpy.where(
                numpy.array(self.best_solutions_fitness) == numpy.max(numpy.array(self.best_solutions_fitness))
            )[0][0]

        self.run_completed = True

        if self.on_stop is not None:
            self.on_stop(self, self.last_generation_fitness)

        if len(self.best_solutions) > 0:
            self.best_solutions = numpy.array(self.best_solutions)


def load(filename):
    try:
        with open(filename + ".pkl", 'rb') as file:
            es_instance = cloudpickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error reading the file {filename}. Please check your inputs.")
    except:
        raise BaseException("Error loading the file.")
    return es_instance
