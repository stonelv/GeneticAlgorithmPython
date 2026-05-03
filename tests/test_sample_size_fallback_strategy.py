import pygad
import pytest
import random
import numpy


def fitness_func(ga, solution, idx):
    return random.random()


def impossible_constraint(x, v):
    return [val for val in v if val > 100]


def narrow_constraint_exact_5(x, v):
    return [val for val in v if val == 5]


def constraint_odd_numbers(x, v):
    return [val for val in v if val % 2 == 1]


class TestSampleSizeFallbackStrategy:
    
    def test_default_fallback_strategy_is_keep(self):
        ga_instance = pygad.GA(num_generations=1,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=10,
                               num_genes=5,
                               gene_type=int,
                               init_range_low=0,
                               init_range_high=10,
                               random_seed=123,
                               save_solutions=True,
                               suppress_warnings=True)
        
        assert ga_instance.sample_size_fallback_strategy == 'keep'
        assert ga_instance.max_sample_size == 10000

    def test_gene_constraint_error_exception_class(self):
        try:
            raise pygad.GeneConstraintError(
                gene_index=0,
                gene_value=5,
                solution=[1, 2, 3, 4, 5],
                sample_size_used=10,
                generations_completed=3,
                details='No values >= 100 found in range [0, 10]'
            )
        except pygad.GeneConstraintError as e:
            assert e.gene_index == 0
            assert e.gene_value == 5
            assert e.solution == [1, 2, 3, 4, 5]
            assert e.sample_size_used == 10
            assert e.generations_completed == 3
            assert 'No values >= 100' in e.details

    def test_duplicate_gene_error_exception_class(self):
        try:
            raise pygad.DuplicateGeneError(
                gene_index=2,
                gene_value=10,
                solution=[1, 2, 10, 10, 20],
                sample_size_used=50,
                generations_completed=5,
                details='Value 10 already appears at index 3'
            )
        except pygad.DuplicateGeneError as e:
            assert e.gene_index == 2
            assert e.gene_value == 10
            assert e.solution == [1, 2, 10, 10, 20]
            assert e.sample_size_used == 50
            assert e.generations_completed == 5
            assert 'Value 10' in e.details

    def test_invalid_fallback_strategy_raises_value_error(self):
        with pytest.raises(ValueError) as excinfo:
            pygad.GA(num_generations=1,
                    num_parents_mating=5,
                    fitness_func=fitness_func,
                    sol_per_pop=10,
                    num_genes=5,
                    gene_type=int,
                    init_range_low=0,
                    init_range_high=10,
                    sample_size_fallback_strategy='invalid_strategy',
                    random_seed=123,
                    save_solutions=True,
                    suppress_warnings=True)
        
        assert 'sample_size_fallback_strategy' in str(excinfo.value)

    def test_keep_strategy_no_exception(self):
        gene_constraint = [impossible_constraint, None, None, None, None]
        
        ga_instance = pygad.GA(num_generations=1,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=10,
                               num_genes=5,
                               gene_type=int,
                               init_range_low=0,
                               init_range_high=10,
                               random_mutation_min_val=0,
                               random_mutation_max_val=10,
                               mutation_by_replacement=True,
                               allow_duplicate_genes=True,
                               gene_constraint=gene_constraint,
                               sample_size_fallback_strategy='keep',
                               sample_size=5,
                               random_seed=123,
                               save_solutions=True,
                               suppress_warnings=True)
        ga_instance.run()
        
        assert ga_instance is not None

    def test_raise_strategy_raises_gene_constraint_error(self):
        gene_constraint = [impossible_constraint, None, None, None, None]
        
        with pytest.raises(pygad.GeneConstraintError) as excinfo:
            pygad.GA(num_generations=1,
                    num_parents_mating=5,
                    fitness_func=fitness_func,
                    sol_per_pop=10,
                    num_genes=5,
                    gene_type=int,
                    init_range_low=0,
                    init_range_high=10,
                    random_mutation_min_val=0,
                    random_mutation_max_val=10,
                    mutation_by_replacement=True,
                    allow_duplicate_genes=True,
                    gene_constraint=gene_constraint,
                    sample_size_fallback_strategy='raise',
                    sample_size=5,
                    random_seed=123,
                    save_solutions=True,
                    suppress_warnings=True)
        
        e = excinfo.value
        assert hasattr(e, 'gene_index')
        assert hasattr(e, 'gene_value')
        assert hasattr(e, 'solution')
        assert hasattr(e, 'sample_size_used')
        assert hasattr(e, 'generations_completed')
        assert hasattr(e, 'details')

    def test_expand_sample_strategy_finds_valid_values(self):
        gene_constraint = [constraint_odd_numbers, None, None, None, None]
        
        ga_instance = pygad.GA(num_generations=1,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=5,
                               num_genes=5,
                               gene_type=int,
                               init_range_low=0,
                               init_range_high=100,
                               random_mutation_min_val=0,
                               random_mutation_max_val=100,
                               mutation_by_replacement=True,
                               allow_duplicate_genes=True,
                               gene_constraint=gene_constraint,
                               sample_size_fallback_strategy='expand_sample',
                               sample_size=2,
                               max_sample_size=100,
                               random_seed=123,
                               save_solutions=True,
                               suppress_warnings=True)
        ga_instance.run()
        
        initial_population = ga_instance.initial_population
        for val in initial_population[:, 0]:
            assert val % 2 == 1, f"Value {val} is not an odd number"

    def test_switch_to_discrete_strategy_uses_gene_space(self):
        gene_constraint = [impossible_constraint, None, None, None, None]
        
        gene_space = [
            [1, 2, 3, 4, 5],
            [10, 20, 30, 40, 50],
            [100, 200, 300],
            [0.1, 0.2, 0.3],
            [1, 2, 3]
        ]
        
        ga_instance = pygad.GA(num_generations=1,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=5,
                               num_genes=5,
                               gene_space=gene_space,
                               gene_type=int,
                               init_range_low=0,
                               init_range_high=10,
                               random_mutation_min_val=0,
                               random_mutation_max_val=10,
                               mutation_by_replacement=True,
                               allow_duplicate_genes=True,
                               gene_constraint=gene_constraint,
                               sample_size_fallback_strategy='switch_to_discrete',
                               sample_size=5,
                               random_seed=123,
                               save_solutions=True,
                               suppress_warnings=True)
        ga_instance.run()
        
        initial_population = ga_instance.initial_population
        for val in initial_population[:, 0]:
            assert val in [1, 2, 3, 4, 5], f"Value {val} not in gene_space[0]"

    def test_supported_fallback_strategies_list(self):
        from pygad.utils.validation import Validation
        assert 'keep' in Validation.supported_fallback_strategies
        assert 'expand_sample' in Validation.supported_fallback_strategies
        assert 'switch_to_discrete' in Validation.supported_fallback_strategies
        assert 'raise' in Validation.supported_fallback_strategies

    def test_case_insensitive_fallback_strategy(self):
        ga_instance = pygad.GA(num_generations=1,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=10,
                               num_genes=5,
                               gene_type=int,
                               init_range_low=0,
                               init_range_high=10,
                               sample_size_fallback_strategy='RAISE',
                               random_seed=123,
                               save_solutions=True,
                               suppress_warnings=True)
        assert ga_instance.sample_size_fallback_strategy == 'raise'

    def test_max_sample_size_parameter(self):
        ga_instance = pygad.GA(num_generations=1,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=10,
                               num_genes=5,
                               gene_type=int,
                               init_range_low=0,
                               init_range_high=10,
                               sample_size_fallback_strategy='expand_sample',
                               max_sample_size=500,
                               random_seed=123,
                               save_solutions=True,
                               suppress_warnings=True)
        assert ga_instance.max_sample_size == 500
