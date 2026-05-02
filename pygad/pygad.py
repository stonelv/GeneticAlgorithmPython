import numpy
import cloudpickle
import csv
from pygad import utils
from pygad import helper
from pygad import visualize

# Extend all the classes so that they can be referenced by just the `self` object of the `pygad.GA` class.
class GA(utils.parent_selection.ParentSelection,
         utils.crossover.Crossover,
         utils.mutation.Mutation,
         utils.nsga2.NSGA2,
         utils.validation.Validation,
         utils.engine.GAEngine,
         helper.unique.Unique,
         helper.misc.Helper,
         visualize.plot.Plot):

    supported_int_types = [int, numpy.int8, numpy.int16, numpy.int32, numpy.int64,
                           numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64,
                           object]
    supported_float_types = [float, numpy.float16, numpy.float32, numpy.float64,
                             object]
    supported_int_float_types = supported_int_types + supported_float_types

    def __init__(self,
                 num_generations,
                 num_parents_mating,
                 fitness_func,
                 fitness_batch_size=None,
                 initial_population=None,
                 sol_per_pop=None,
                 num_genes=None,
                 init_range_low=-4,
                 init_range_high=4,
                 gene_type=float,
                 parent_selection_type="sss",
                 keep_parents=-1,
                 keep_elitism=1,
                 K_tournament=3,
                 crossover_type="single_point",
                 crossover_probability=None,
                 mutation_type="random",
                 mutation_probability=None,
                 mutation_by_replacement=False,
                 mutation_percent_genes='default',
                 mutation_num_genes=None,
                 random_mutation_min_val=-1.0,
                 random_mutation_max_val=1.0,
                 gene_space=None,
                 gene_constraint=None,
                 sample_size=100,
                 allow_duplicate_genes=True,
                 on_start=None,
                 on_fitness=None,
                 on_parents=None,
                 on_crossover=None,
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
        """
        The constructor of the GA class accepts all parameters required to create an instance of the GA class. It validates such parameters.

        num_generations: Number of generations.
        num_parents_mating: Number of solutions to be selected as parents in the mating pool.

        fitness_func: Accepts a function/method and returns the fitness value of the solution. In PyGAD 2.20.0, a third parameter is passed referring to the 'pygad.GA' instance.
        fitness_batch_size: Added in PyGAD 2.19.0. Supports calculating the fitness in batches. If the value is 1 or None, then the fitness function is called for each individual solution. If given another value X where X is neither 1 nor None (e.g. X=3), then the fitness function is called once for each X (3) solutions.

        initial_population: A user-defined initial population. It is useful when the user wants to start the generations with a custom initial population. It defaults to None which means no initial population is specified by the user. In this case, PyGAD creates an initial population using the 'sol_per_pop' and 'num_genes' parameters. An exception is raised if the 'initial_population' is None while any of the 2 parameters ('sol_per_pop' or 'num_genes') is also None.
        sol_per_pop: Number of solutions in the population. 
        num_genes: Number of genes in the solution.

        init_range_low: The lower value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20 and higher.
        init_range_high: The upper value of the random range from which the gene values in the initial population are selected. It defaults to -4. Available in PyGAD 1.0.20.
        # It is OK to set the value of the 2 parameters ('init_range_low' and 'init_range_high') to be equal, higher or lower than the other parameter (i.e. init_range_low is not needed to be lower than init_range_high).

        gene_type: The type of the gene. It is assigned to any of these types (int, numpy.int8, numpy.int16, numpy.int32, numpy.int64, numpy.uint, numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, float, numpy.float16, numpy.float32, numpy.float64) and forces all the genes to be of that type.

        parent_selection_type: Type of parent selection.
        keep_parents: If 0, this means no parent in the current population will be used in the next population. If -1, this means all parents in the current population will be used in the next population. If set to a value > 0, then the specified value refers to the number of parents in the current population to be used in the next population. Some parent selection operators such as rank selection, favor population diversity and therefore keeping the parents in the next generation can be beneficial. However, some other parent selection operators, such as roulette wheel selection (RWS), have higher selection pressure and keeping more than one parent in the next generation can seriously harm population diversity. This parameter has an effect only when the keep_elitism parameter is 0. Thanks to Prof. Fernando Jiménez (http://webs.um.es/fernan) for editing this sentence.
        K_tournament: When the value of 'parent_selection_type' is 'tournament', the 'K_tournament' parameter specifies the number of solutions from which a parent is selected randomly.

        keep_elitism: Added in PyGAD 2.18.0. It can take the value 0 or a positive integer that satisfies (0 <= keep_elitism <= sol_per_pop). It defaults to 1 which means only the best solution in the current generation is kept in the next generation. If assigned 0, this means it has no effect. If assigned a positive integer K, then the best K solutions are kept in the next generation. It cannot be assigned a value greater than the value assigned to the sol_per_pop parameter. If this parameter has a value different from 0, then the keep_parents parameter will have no effect.

        crossover_type: Type of the crossover operator. If  crossover_type=None, then the crossover step is bypassed which means no crossover is applied and thus no offspring will be created in the next generations. The next generation will use the solutions in the current population.
        crossover_probability: The probability of selecting a solution for the crossover operation. If the solution probability is <= crossover_probability, the solution is selected. The value must be between 0 and 1 inclusive.

        mutation_type: Type of the mutation operator. If mutation_type=None, then the mutation step is bypassed which means no mutation is applied and thus no changes are applied to the offspring created using the crossover operation. The offspring will be used unchanged in the next generation.
        mutation_probability: The probability of selecting a gene for the mutation operation. If the gene probability is <= mutation_probability, the gene is selected. It accepts either a single value for fixed mutation or a list/tuple/numpy.ndarray of 2 values for adaptive mutation. The values must be between 0 and 1 inclusive. If specified, then no need for the 2 parameters mutation_percent_genes and mutation_num_genes.

        mutation_by_replacement: An optional bool parameter. It works only when the selected type of mutation is random (mutation_type="random"). In this case, setting mutation_by_replacement=True means replace the gene by the randomly generated value. If False, then it has no effect and random mutation works by adding the random value to the gene.

        mutation_percent_genes: Percentage of genes to mutate which defaults to the string 'default' which means 10%. This parameter has no action if any of the 2 parameters mutation_probability or mutation_num_genes exist.
        mutation_num_genes: Number of genes to mutate which defaults to None. If the parameter mutation_num_genes exists, then no need for the parameter mutation_percent_genes. This parameter has no action if the mutation_probability parameter exists.
        random_mutation_min_val: The minimum value of the range from which a random value is selected to be added to the selected gene(s) to mutate. It defaults to -1.0.
        random_mutation_max_val: The maximum value of the range from which a random value is selected to be added to the selected gene(s) to mutate. It defaults to 1.0.

        gene_space: It accepts a list of all possible values of the gene. This list is used in the mutation step. Should be used only if the gene space is a set of discrete values. No need for the 2 parameters (random_mutation_min_val and random_mutation_max_val) if the parameter gene_space exists. Added in PyGAD 2.5.0. In PyGAD 2.11.0, the gene_space can be assigned a dict.

        gene_constraint: It accepts a list of constraints for the genes. Each constraint is a Python function. Added in PyGAD 3.5.0.
        sample_size: To select a gene value that respects a constraint, this variable defines the size of the sample from which a value is selected randomly. Useful if either allow_duplicate_genes or gene_constraint is used. Added in PyGAD 3.5.0.

        on_start: Accepts a function/method to be called only once before the genetic algorithm starts its evolution. If function, then it must accept a single parameter representing the instance of the genetic algorithm. If method, then it must accept 2 parameters where the second one refers to the method's object. Added in PyGAD 2.6.0.
        on_fitness: Accepts a function/method to be called after calculating the fitness values of all solutions in the population. If function, then it must accept 2 parameters: 1) a list of all solutions' fitness values 2) the instance of the genetic algorithm. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.
        on_parents: Accepts a function/method to be called after selecting the parents that mate. If function, then it must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the selected parents. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.
        on_crossover: Accepts a function/method to be called each time the crossover operation is applied. If function, then it must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the offspring generated using crossover. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.
        on_mutation: Accepts a function/method to be called each time the mutation operation is applied. If function, then it must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one represents the offspring after applying the mutation. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.
        on_generation: Accepts a function/method to be called after each generation. If function, then it must accept a single parameter representing the instance of the genetic algorithm. If the function returned "stop", then the run() method stops without completing the other generations. If method, then it must accept 2 parameters where the second one refers to the method's object. Added in PyGAD 2.6.0.
        on_stop: Accepts a function/method to be called only once exactly before the genetic algorithm stops or when it completes all the generations. If function, then it must accept 2 parameters: the first one represents the instance of the genetic algorithm and the second one is a list of fitness values of the last population's solutions. If method, then it must accept 3 parameters where the third one refers to the method's object. Added in PyGAD 2.6.0.

        save_best_solutions: Added in PyGAD 2.9.0 and its type is bool. If True, then the best solution in each generation is saved into the 'best_solutions' attribute. Use this parameter with caution as it may cause memory overflow when either the number of generations or the number of genes is large.
        save_solutions: Added in PyGAD 2.15.0 and its type is bool. If True, then all solutions in each generation are saved into the 'solutions' attribute. Use this parameter with caution as it may cause memory overflow when either the number of generations, number of genes, or number of solutions in population is large.

        suppress_warnings: Added in PyGAD 2.10.0 and its type is bool. If True, then no warning messages will be displayed. It defaults to False.

        allow_duplicate_genes: Added in PyGAD 2.13.0. If True, then a solution/chromosome may have duplicate gene values. If False, then each gene will have a unique value in its solution.

        stop_criteria: Added in PyGAD 2.15.0. It is assigned to some criteria to stop the evolution if at least one criterion holds.

        parallel_processing: Added in PyGAD 2.17.0. Defaults to `None` which means no parallel processing is used. If a positive integer is assigned, it specifies the number of threads to be used. If a list or a tuple of exactly 2 elements is assigned, then: 1) The first element can be either "process" or "thread" to specify whether processes or threads are used, respectively. 2) The second element can be: 1) A positive integer to select the maximum number of processes or threads to be used. 2) 0 to indicate that parallel processing is not used. This is identical to setting 'parallel_processing=None'. 3) None to use the default value as calculated by the concurrent.futures module.

        random_seed: Added in PyGAD 2.18.0. It defines the random seed to be used by the random function generators (we use random functions in the NumPy and random modules). This helps to reproduce the same results by setting the same random seed.

        logger: Added in PyGAD 2.20.0. It accepts a logger object of the 'logging.Logger' class to log the messages. If no logger is passed, then a default logger is created to log/print the messages to the console exactly like using the 'print()' function.
        """
        try:
            self.validate_parameters(num_generations=num_generations,
                                     num_parents_mating=num_parents_mating,
                                     fitness_func=fitness_func,
                                     fitness_batch_size=fitness_batch_size,
                                     initial_population=initial_population,
                                     sol_per_pop=sol_per_pop,
                                     num_genes=num_genes,
                                     init_range_low=init_range_low,
                                     init_range_high=init_range_high,
                                     gene_type=gene_type,
                                     parent_selection_type=parent_selection_type,
                                     keep_parents=keep_parents,
                                     keep_elitism=keep_elitism,
                                     K_tournament=K_tournament,
                                     crossover_type=crossover_type,
                                     crossover_probability=crossover_probability,
                                     mutation_type=mutation_type,
                                     mutation_probability=mutation_probability,
                                     mutation_by_replacement=mutation_by_replacement,
                                     mutation_percent_genes=mutation_percent_genes,
                                     mutation_num_genes=mutation_num_genes,
                                     random_mutation_min_val=random_mutation_min_val,
                                     random_mutation_max_val=random_mutation_max_val,
                                     gene_space=gene_space,
                                     gene_constraint=gene_constraint,
                                     sample_size=sample_size,
                                     allow_duplicate_genes=allow_duplicate_genes,
                                     on_start=on_start,
                                     on_fitness=on_fitness,
                                     on_parents=on_parents,
                                     on_crossover=on_crossover,
                                     on_mutation=on_mutation,
                                     on_generation=on_generation,
                                     on_stop=on_stop,
                                     save_best_solutions=save_best_solutions,
                                     save_solutions=save_solutions,
                                     suppress_warnings=suppress_warnings,
                                     stop_criteria=stop_criteria,
                                     parallel_processing=parallel_processing,
                                     random_seed=random_seed,
                                     logger=logger)
        except Exception as e:
            self.logger.exception(e)
            # sys.exit(-1)
            raise e

    def save(self, filename):
        """
        Saves the genetic algorithm instance:
            -filename: Name of the file to save the instance. No extension is needed.
        """

        cloudpickle_serialized_object = cloudpickle.dumps(self)
        with open(filename + ".pkl", 'wb') as file:
            file.write(cloudpickle_serialized_object)
            cloudpickle.dump(self, file)

    def to_csv(self, filename, delimiter=','):
        """
        Exports the recorded run metrics to a CSV file.
        
        This method exports the metrics recorded during the `run()` method to a CSV file.
        The metrics include generation number, time elapsed, best fitness, mean fitness,
        and population diversity (gene variance).
        
        Parameters
        ----------
        filename : str
            Name of the CSV file to save the metrics. The '.csv' extension will be 
            added if not present.
        
        delimiter : str, optional
            Delimiter to use in the CSV file. Default is ','.
        
        Returns
        -------
        None
        
        Notes
        -----
        This method can only be called after completing at least 1 generation.
        If no generation is completed, a RuntimeError is raised.
        
        For multi-objective optimization problems, the `best_fitness` and `mean_fitness` 
        columns will contain multiple values separated by semicolons.
        
        Examples
        --------
        >>> import pygad
        >>> import numpy
        >>> 
        >>> def fitness_func(ga_instance, solution, solution_idx):
        ...     output = numpy.sum(solution * [4, -2, 3.5, 5, -11, -4.7])
        ...     fitness = 1.0 / (numpy.abs(output - 44) + 0.000001)
        ...     return fitness
        >>> 
        >>> ga_instance = pygad.GA(num_generations=10,
        ...                        num_parents_mating=4,
        ...                        sol_per_pop=8,
        ...                        num_genes=6,
        ...                        fitness_func=fitness_func)
        >>> 
        >>> ga_instance.run()
        >>> ga_instance.to_csv('ga_metrics.csv')
        """
        if self.run_metrics is None or len(self.run_metrics['generation']) == 0:
            self.logger.error("The to_csv() method can only be called after completing at least 1 generation.")
            raise RuntimeError("The to_csv() method can only be called after completing at least 1 generation.")
        
        # Add .csv extension if not present
        if not filename.endswith('.csv'):
            filename = filename + '.csv'
        
        # Prepare headers
        headers = ['generation', 'time_elapsed', 'best_fitness', 'mean_fitness', 'diversity']
        
        # Check if this is a multi-objective problem
        # If best_fitness is a list/tuple/ndarray, then it's multi-objective
        is_multi_objective = False
        if len(self.run_metrics['best_fitness']) > 0:
            first_best_fitness = self.run_metrics['best_fitness'][0]
            if type(first_best_fitness) in [list, tuple, numpy.ndarray]:
                is_multi_objective = True
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            
            # Write header
            writer.writerow(headers)
            
            # Write data rows
            num_rows = len(self.run_metrics['generation'])
            for i in range(num_rows):
                generation = self.run_metrics['generation'][i]
                time_elapsed = self.run_metrics['time_elapsed'][i]
                best_fitness = self.run_metrics['best_fitness'][i]
                mean_fitness = self.run_metrics['mean_fitness'][i]
                diversity = self.run_metrics['diversity'][i]
                
                # Format fitness values for multi-objective problems
                if is_multi_objective:
                    # Convert list/tuple/ndarray to semicolon-separated string
                    if type(best_fitness) in [list, tuple, numpy.ndarray]:
                        best_fitness_str = ';'.join(str(v) for v in best_fitness)
                    else:
                        best_fitness_str = str(best_fitness)
                    
                    if type(mean_fitness) in [list, tuple, numpy.ndarray]:
                        mean_fitness_str = ';'.join(str(v) for v in mean_fitness)
                    else:
                        mean_fitness_str = str(mean_fitness)
                else:
                    best_fitness_str = str(best_fitness)
                    mean_fitness_str = str(mean_fitness)
                
                # Format diversity (handle NaN)
                if numpy.isnan(diversity):
                    diversity_str = 'NaN'
                else:
                    diversity_str = str(diversity)
                
                writer.writerow([
                    generation,
                    time_elapsed,
                    best_fitness_str,
                    mean_fitness_str,
                    diversity_str
                ])
        
        self.logger.info(f"Run metrics saved to {filename}")

    def plot_metrics(self,
                     title="PyGAD - Run Metrics",
                     font_size=12,
                     figsize=(12, 8),
                     plot_type="plot",
                     colors=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                     labels=["Best Fitness", "Mean Fitness", "Time Elapsed (s)", "Diversity"],
                     save_dir=None,
                     show=True):
        """
        Creates, shows, and returns a figure with 4 subplots showing the run metrics 
        recorded during the genetic algorithm evolution.
        
        This method visualizes the metrics recorded by the run metrics recorder, including:
        - Best fitness over generations
        - Mean fitness over generations
        - Time elapsed per generation
        - Population diversity (gene variance) over generations
        
        Parameters
        ----------
        title : str, optional
            Main title of the figure. Default is "PyGAD - Run Metrics".
        
        font_size : int, optional
            Font size for labels and titles. Default is 12.
        
        figsize : tuple, optional
            Figure size as (width, height) in inches. Default is (12, 8).
        
        plot_type : str, optional
            Type of plot. Can be "plot", "scatter", or "bar". Default is "plot".
        
        colors : list, optional
            List of 4 colors for the 4 subplots. Default is:
            ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] (blue, orange, green, red).
        
        labels : list, optional
            List of 4 labels for the 4 subplots' y-axes. Default is:
            ["Best Fitness", "Mean Fitness", "Time Elapsed (s)", "Diversity"].
        
        save_dir : str, optional
            Directory path to save the figure. If None, the figure is not saved.
        
        show : bool, optional
            Whether to show the figure. Default is True.
        
        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object containing all subplots.
        
        Notes
        -----
        This method can only be called after completing at least 1 generation.
        If no generation is completed, a RuntimeError is raised.
        
        For multi-objective optimization problems, only the first objective is plotted 
        for best fitness and mean fitness. Use separate plots for other objectives.
        
        Examples
        --------
        >>> import pygad
        >>> import numpy
        >>> 
        >>> def fitness_func(ga_instance, solution, solution_idx):
        ...     output = numpy.sum(solution * [4, -2, 3.5, 5, -11, -4.7])
        ...     fitness = 1.0 / (numpy.abs(output - 44) + 0.000001)
        ...     return fitness
        >>> 
        >>> ga_instance = pygad.GA(num_generations=10,
        ...                        num_parents_mating=4,
        ...                        sol_per_pop=8,
        ...                        num_genes=6,
        ...                        fitness_func=fitness_func)
        >>> 
        >>> ga_instance.run()
        >>> fig = ga_instance.plot_metrics()
        """
        from pygad.visualize.plot import get_matplotlib
        import matplotlib
        
        if self.run_metrics is None or len(self.run_metrics['generation']) == 0:
            self.logger.error("The plot_metrics() method can only be called after completing at least 1 generation.")
            raise RuntimeError("The plot_metrics() method can only be called after completing at least 1 generation.")
        
        matplt = get_matplotlib()
        
        generations = numpy.array(self.run_metrics['generation'])
        best_fitness = numpy.array(self.run_metrics['best_fitness'])
        mean_fitness = numpy.array(self.run_metrics['mean_fitness'])
        time_elapsed = numpy.array(self.run_metrics['time_elapsed'])
        diversity = numpy.array(self.run_metrics['diversity'])
        
        is_multi_objective = False
        if len(best_fitness) > 0:
            if type(self.run_metrics['best_fitness'][0]) in [list, tuple, numpy.ndarray]:
                is_multi_objective = True
                best_fitness = numpy.array([bf[0] for bf in self.run_metrics['best_fitness']])
                if type(self.run_metrics['mean_fitness'][0]) in [list, tuple, numpy.ndarray]:
                    mean_fitness = numpy.array([mf[0] for mf in self.run_metrics['mean_fitness']])
        
        fig, axs = matplt.subplots(2, 2, figsize=figsize)
        fig.suptitle(title, fontsize=font_size + 2, fontweight='bold')
        
        axs = axs.flatten()
        
        metrics_data = [
            (best_fitness, labels[0], colors[0], "Best Fitness Over Generations"),
            (mean_fitness, labels[1], colors[1], "Mean Fitness Over Generations"),
            (time_elapsed, labels[2], colors[2], "Time Elapsed Per Generation"),
            (diversity, labels[3], colors[3], "Population Diversity Over Generations")
        ]
        
        for idx, (data, ylabel, color, subplot_title) in enumerate(metrics_data):
            ax = axs[idx]
            
            if plot_type == "plot":
                ax.plot(generations, data, color=color, linewidth=2, marker='o', markersize=4)
            elif plot_type == "scatter":
                ax.scatter(generations, data, color=color, s=30, alpha=0.7)
            elif plot_type == "bar":
                ax.bar(generations, data, color=color, alpha=0.7)
            
            ax.set_title(subplot_title, fontsize=font_size)
            ax.set_xlabel("Generation", fontsize=font_size - 2)
            ax.set_ylabel(ylabel, fontsize=font_size - 2)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=font_size - 3)
        
        matplt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if is_multi_objective:
            fig.text(0.5, 0.01, 
                    "Note: Only the first objective is shown for fitness metrics (multi-objective problem).",
                    ha='center', fontsize=font_size - 2, style='italic')
        
        if save_dir is not None:
            fig.savefig(fname=save_dir, bbox_inches='tight', dpi=150)
        
        if show:
            matplt.show()
        
        return fig

def load(filename):
    """
    Reads a saved instance of the genetic algorithm:
        -filename: Name of the file to read the instance. No extension is needed.
    Returns the genetic algorithm instance.
    """

    try:
        with open(filename + ".pkl", 'rb') as file:
            ga_in = cloudpickle.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Error reading the file {filename}. Please check your inputs.")
    except:
        # raise BaseException("Error loading the file. If the file already exists, please reload all the functions previously used (e.g. fitness function).")
        raise BaseException("Error loading the file.")
    return ga_in
