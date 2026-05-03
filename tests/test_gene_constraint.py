import pygad
import random
import numpy

num_generations = 1

initial_population = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

def population_gene_constraint(gene_space=None,
                               gene_type=float,
                               num_genes=10,
                               mutation_by_replacement=False,
                               random_mutation_min_val=-1,
                               random_mutation_max_val=1,
                               init_range_low=-4,
                               init_range_high=4,
                               random_seed=123,
                               crossover_type='single_point',
                               initial_population=None,
                               parent_selection_type='sss',
                               multi_objective=False,
                               gene_constraint=None,
                               allow_duplicate_genes=True):

    def fitness_func_no_batch_single(ga, solution, idx):
        return random.random()

    def fitness_func_no_batch_multi(ga, solution, idx):
        return [random.random(), random.random()]

    if multi_objective == True:
        fitness_func = fitness_func_no_batch_multi
    else:
        fitness_func = fitness_func_no_batch_single

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           gene_type=gene_type,
                           initial_population=initial_population,
                           parent_selection_type=parent_selection_type,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           random_mutation_min_val=random_mutation_min_val,
                           random_mutation_max_val=random_mutation_max_val,
                           allow_duplicate_genes=allow_duplicate_genes,
                           mutation_by_replacement=mutation_by_replacement,
                           random_seed=random_seed,
                           crossover_type=crossover_type,
                           gene_constraint=gene_constraint,
                           save_solutions=True,
                           suppress_warnings=False)

    ga_instance.run()

    return ga_instance

#### Single-Objective
def test_initial_population_int_by_replacement():
    gene_constraint=[lambda x,v: [val for val in v if val>=8],
                     lambda x,v: [val for val in v if val>=8],
                     lambda x,v: [val for val in v if 5>=val>=1],
                     lambda x,v: [val for val in v if 5>val>3],
                     lambda x,v: [val for val in v if val<2]]
    ga_instance = population_gene_constraint(gene_constraint=gene_constraint,
                                             init_range_low=0,
                                             init_range_high=10,
                                             random_mutation_min_val=0,
                                             random_mutation_max_val=10,
                                             num_genes=5,
                                             gene_type=int,
                                             mutation_by_replacement=True)
    initial_population = ga_instance.initial_population

    assert numpy.all(initial_population[:, 0] >= 8), "Not all values in column 0 are >= 8"
    assert numpy.all(initial_population[:, 1] >= 8), "Not all values in column 1 are >= 8"
    assert numpy.all(initial_population[:, 2] >= 1), "Not all values in column 2 are >= 1"
    assert numpy.all((initial_population[:, 2] >= 1) & (initial_population[:, 2] <= 5)), "Not all values in column 2 between 1 and 5 (inclusive)"
    assert numpy.all(initial_population[:, 3] == 4), "Not all values in column 3 between 3 and 5 (exclusive)"
    assert numpy.all(initial_population[:, 4] < 2), "Not all values in column 4 < 2"
    print("All constraints are met")

def test_initial_population_int_by_replacement_no_duplicates():
    gene_constraint=[lambda x,v: [val for val in v if val>=5],
                     lambda x,v: [val for val in v if val>=5],
                     lambda x,v: [val for val in v if val>=5],
                     lambda x,v: [val for val in v if val>=5],
                     lambda x,v: [val for val in v if val>=5]]
    ga_instance = population_gene_constraint(gene_constraint=gene_constraint,
                                             init_range_low=1,
                                             init_range_high=10,
                                             random_mutation_min_val=1,
                                             random_mutation_max_val=10,
                                             gene_type=int,
                                             num_genes=5,
                                             mutation_by_replacement=True,
                                             allow_duplicate_genes=False)

    num_duplicates = 0
    for idx, solution in enumerate(ga_instance.solutions):
        num = len(solution) - len(set(solution))
        if num != 0:
            print(solution, idx)
        num_duplicates += num

    assert num_duplicates == 0

    initial_population = ga_instance.initial_population
    # print(initial_population)

    assert numpy.all(initial_population[:, 0] >= 5), "Not all values in column 0 >= 5"
    assert numpy.all(initial_population[:, 1] >= 5), "Not all values in column 1 >= 5"
    assert numpy.all(initial_population[:, 2] >= 5), "Not all values in column 2 >= 5"
    assert numpy.all(initial_population[:, 3] >= 5), "Not all values in column 3 >= 5"
    assert numpy.all(initial_population[:, 4] >= 5), "Not all values in column 4 >= 5"
    print("All constraints are met")

def test_initial_population_int_by_replacement_no_duplicates2():
    gene_constraint=[lambda x,v: [val for val in v if val>=98],
                     lambda x,v: [val for val in v if val>=98],
                     lambda x,v: [val for val in v if 20<val<40],
                     lambda x,v: [val for val in v if val<40],
                     lambda x,v: [val for val in v if val<50],
                     lambda x,v: [val for val in v if val<100]]
    ga_instance = population_gene_constraint(gene_constraint=gene_constraint,
                                             random_mutation_min_val=1,
                                             random_mutation_max_val=100,
                                             init_range_low=1,
                                             init_range_high=100,
                                             gene_type=int,
                                             num_genes=6,
                                             crossover_type=None,
                                             mutation_by_replacement=True,
                                             allow_duplicate_genes=False)

    num_duplicates = 0
    for idx, solution in enumerate(ga_instance.solutions):
        num = len(solution) - len(set(solution))
        if num != 0:
            print(solution, idx)
        num_duplicates += num

    assert num_duplicates == 0

    initial_population = ga_instance.initial_population
    # print(initial_population)

    assert numpy.all(initial_population[:, 0] >= 98), "Not all values in column 0 are >= 98"
    assert numpy.all(initial_population[:, 1] >= 98), "Not all values in column 1 are >= 98"
    assert numpy.all((initial_population[:, 2] > 20) & (initial_population[:, 2] < 40)), "Not all values in column 2 between 20 and 40 (exclusive)"
    assert numpy.all(initial_population[:, 3] < 40), "Not all values in column 3 < 40"
    assert numpy.all(initial_population[:, 4] < 50), "Not all values in column 4 < 50"
    assert numpy.all(initial_population[:, 5] < 100), "Not all values in column 4 < 100"
    print("All constraints are met")

def test_initial_population_float_by_replacement_no_duplicates():
    gene_constraint=[lambda x,v: [val for val in v if val>=5],
                     lambda x,v: [val for val in v if val>=5],
                     lambda x,v: [val for val in v if val>=5],
                     lambda x,v: [val for val in v if val>=5],
                     lambda x,v: [val for val in v if val>=5]]
    ga_instance = population_gene_constraint(gene_constraint=gene_constraint,
                                             init_range_low=1,
                                             init_range_high=10,
                                             gene_type=[float, 1],
                                             num_genes=5,
                                             crossover_type=None,
                                             mutation_by_replacement=False,
                                             allow_duplicate_genes=False)

    num_duplicates = 0
    for idx, solution in enumerate(ga_instance.solutions):
        num = len(solution) - len(set(solution))
        if num != 0:
            print(solution, idx)
        num_duplicates += num

    assert num_duplicates == 0

    initial_population = ga_instance.initial_population
    # print(initial_population)

    assert numpy.all(initial_population[:, 0] >= 5), "Not all values in column 0 >= 5"
    assert numpy.all(initial_population[:, 1] >= 5), "Not all values in column 1 >= 5"
    assert numpy.all(initial_population[:, 2] >= 5), "Not all values in column 2 >= 5"
    assert numpy.all(initial_population[:, 3] >= 5), "Not all values in column 3 >= 5"
    assert numpy.all(initial_population[:, 4] >= 5), "Not all values in column 4 >= 5"
    print("All constraints are met")

def test_initial_population_float_by_replacement_no_duplicates2():
    gene_constraint=[lambda x,v: [val for val in v if val>=1],
                     lambda x,v: [val for val in v if val>=1],
                     lambda x,v: [val for val in v if val>=1],
                     lambda x,v: [val for val in v if val>=1],
                     lambda x,v: [val for val in v if val>=1]]
    ga_instance = population_gene_constraint(gene_constraint=gene_constraint,
                                             init_range_low=1,
                                             init_range_high=2,
                                             gene_type=[float, 1],
                                             num_genes=5,
                                             crossover_type=None,
                                             mutation_by_replacement=False,
                                             allow_duplicate_genes=False)

    num_duplicates = 0
    for idx, solution in enumerate(ga_instance.solutions):
        num = len(solution) - len(set(solution))
        if num != 0:
            print(solution, idx)
        num_duplicates += num

    assert num_duplicates == 0

    initial_population = ga_instance.initial_population
    # print(initial_population)

    assert numpy.all(initial_population[:, 0] >= 1), "Not all values in column 0 >= 1"
    assert numpy.all(initial_population[:, 1] >= 1), "Not all values in column 1 >= 1"
    assert numpy.all(initial_population[:, 2] >= 1), "Not all values in column 2 >= 1"
    assert numpy.all(initial_population[:, 3] >= 1), "Not all values in column 3 >= 1"
    assert numpy.all(initial_population[:, 4] >= 1), "Not all values in column 4 >= 1"
    print("All constraints are met")

def test_initial_population_float_by_replacement_no_duplicates_None_constraints():
    gene_constraint=[lambda x,v: [val for val in v if val>=1],
                     None,
                     lambda x,v: [val for val in v if val>=1],
                     None,
                     lambda x,v: [val for val in v if val>=1]]
    ga_instance = population_gene_constraint(gene_constraint=gene_constraint,
                                             init_range_low=1,
                                             init_range_high=2,
                                             gene_type=[float, 1],
                                             num_genes=5,
                                             crossover_type=None,
                                             mutation_by_replacement=False,
                                             allow_duplicate_genes=False)

    num_duplicates = 0
    for idx, solution in enumerate(ga_instance.solutions):
        num = len(solution) - len(set(solution))
        if num != 0:
            print(solution, idx)
        num_duplicates += num

    assert num_duplicates == 0

    initial_population = ga_instance.initial_population
    # print(initial_population)

    assert numpy.all(initial_population[:, 0] >= 1), "Not all values in column 0 >= 1"
    assert numpy.all(initial_population[:, 1] >= 1), "Not all values in column 1 >= 1"
    assert numpy.all(initial_population[:, 2] >= 1), "Not all values in column 2 >= 1"
    assert numpy.all(initial_population[:, 3] >= 1), "Not all values in column 3 >= 1"
    assert numpy.all(initial_population[:, 4] >= 1), "Not all values in column 4 >= 1"
    print("All constraints are met")

def test_sample_size_fallback_strategy_keep():
    def impossible_constraint(x, v):
        return [val for val in v if val > 100]
    
    gene_constraint=[impossible_constraint, None, None, None, None]
    
    ga_instance = population_gene_constraint(gene_constraint=gene_constraint,
                                             init_range_low=0,
                                             init_range_high=10,
                                             random_mutation_min_val=0,
                                             random_mutation_max_val=10,
                                             num_genes=5,
                                             gene_type=int,
                                             mutation_by_replacement=True,
                                             allow_duplicate_genes=True)
    
    print("keep strategy works - no exception raised")

def test_sample_size_fallback_strategy_raise_gene_constraint_error():
    def impossible_constraint(x, v):
        return [val for val in v if val > 100]
    
    gene_constraint=[impossible_constraint, None, None, None, None]
    
    def fitness_func(ga, solution, idx):
        return random.random()
    
    try:
        ga_instance = pygad.GA(num_generations=num_generations,
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
                               sample_size_fallback_strategy="raise",
                               sample_size=10,
                               random_seed=123,
                               save_solutions=True,
                               suppress_warnings=False)
        
        ga_instance.run()
        assert False, "GeneConstraintError should have been raised"
    except pygad.GeneConstraintError as e:
        print(f"GeneConstraintError raised as expected: {e}")
        assert hasattr(e, 'gene_index'), "GeneConstraintError should have gene_index attribute"
        assert hasattr(e, 'gene_value'), "GeneConstraintError should have gene_value attribute"
        assert hasattr(e, 'solution'), "GeneConstraintError should have solution attribute"
        assert hasattr(e, 'sample_size_used'), "GeneConstraintError should have sample_size_used attribute"
        assert hasattr(e, 'generations_completed'), "GeneConstraintError should have generations_completed attribute"
        assert hasattr(e, 'details'), "GeneConstraintError should have details attribute"
        print("All GeneConstraintError attributes are present")

def test_sample_size_fallback_strategy_expand_sample():
    def narrow_constraint(x, v):
        return [val for val in v if val == 5]
    
    gene_constraint=[narrow_constraint, None, None, None, None]
    
    def fitness_func(ga, solution, idx):
        return random.random()
    
    ga_instance = pygad.GA(num_generations=num_generations,
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
                           sample_size_fallback_strategy="expand_sample",
                           sample_size=5,
                           max_sample_size=20,
                           random_seed=123,
                           save_solutions=True,
                           suppress_warnings=False)
    
    ga_instance.run()
    
    initial_population = ga_instance.initial_population
    assert numpy.all(initial_population[:, 0] == 5), "Not all values in column 0 are 5"
    print("expand_sample strategy works")

def test_sample_size_fallback_strategy_switch_to_discrete():
    def impossible_range_constraint(x, v):
        return [val for val in v if val > 100]
    
    gene_constraint=[impossible_range_constraint, None, None, None, None]
    
    gene_space = [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [100, 200, 300],
        [0.1, 0.2, 0.3],
        [1, 2, 3]
    ]
    
    def fitness_func(ga, solution, idx):
        return random.random()
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
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
                           sample_size_fallback_strategy="switch_to_discrete",
                           sample_size=5,
                           random_seed=123,
                           save_solutions=True,
                           suppress_warnings=False)
    
    ga_instance.run()
    
    initial_population = ga_instance.initial_population
    for val in initial_population[:, 0]:
        assert val in [1, 2, 3, 4, 5], f"Value {val} not in gene_space[0]"
    print("switch_to_discrete strategy works")

def test_invalid_fallback_strategy():
    def fitness_func(ga, solution, idx):
        return random.random()
    
    try:
        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=5,
                               fitness_func=fitness_func,
                               sol_per_pop=10,
                               num_genes=5,
                               gene_type=int,
                               init_range_low=0,
                               init_range_high=10,
                               sample_size_fallback_strategy="invalid_strategy",
                               random_seed=123,
                               save_solutions=True,
                               suppress_warnings=False)
        assert False, "ValueError should have been raised"
    except ValueError as e:
        print(f"ValueError raised as expected: {e}")
        assert "sample_size_fallback_strategy" in str(e), "Error message should mention sample_size_fallback_strategy"

def test_gene_constraint_error_exception_class():
    try:
        raise pygad.GeneConstraintError(
            gene_index=0,
            gene_value=5,
            solution=[1, 2, 3, 4, 5],
            sample_size_used=10,
            generations_completed=3,
            details="No values >= 100 found in range [0, 10]"
        )
    except pygad.GeneConstraintError as e:
        assert e.gene_index == 0, f"Expected gene_index=0, got {e.gene_index}"
        assert e.gene_value == 5, f"Expected gene_value=5, got {e.gene_value}"
        assert e.solution == [1, 2, 3, 4, 5], f"Expected solution=[1,2,3,4,5], got {e.solution}"
        assert e.sample_size_used == 10, f"Expected sample_size_used=10, got {e.sample_size_used}"
        assert e.generations_completed == 3, f"Expected generations_completed=3, got {e.generations_completed}"
        assert "No values >= 100" in e.details, f"Expected 'No values >= 100' in details, got {e.details}"
        print("GeneConstraintError class works correctly")

def test_duplicate_gene_error_exception_class():
    try:
        raise pygad.DuplicateGeneError(
            gene_index=2,
            gene_value=10,
            solution=[1, 2, 10, 10, 20],
            sample_size_used=50,
            generations_completed=5,
            details="Value 10 already appears at index 3"
        )
    except pygad.DuplicateGeneError as e:
        assert e.gene_index == 2, f"Expected gene_index=2, got {e.gene_index}"
        assert e.gene_value == 10, f"Expected gene_value=10, got {e.gene_value}"
        assert e.solution == [1, 2, 10, 10, 20], f"Expected solution=[1,2,10,10,20], got {e.solution}"
        assert e.sample_size_used == 50, f"Expected sample_size_used=50, got {e.sample_size_used}"
        assert e.generations_completed == 5, f"Expected generations_completed=5, got {e.generations_completed}"
        assert "Value 10" in e.details, f"Expected 'Value 10' in details, got {e.details}"
        print("DuplicateGeneError class works correctly")

def test_default_fallback_strategy_is_keep():
    def fitness_func(ga, solution, idx):
        return random.random()
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=10,
                           num_genes=5,
                           gene_type=int,
                           init_range_low=0,
                           init_range_high=10,
                           random_seed=123,
                           save_solutions=True,
                           suppress_warnings=False)
    
    assert ga_instance.sample_size_fallback_strategy == "keep", f"Expected default strategy 'keep', got {ga_instance.sample_size_fallback_strategy}"
    assert ga_instance.max_sample_size == 10000, f"Expected default max_sample_size=10000, got {ga_instance.max_sample_size}"
    print("Default fallback strategy is 'keep'")

if __name__ == "__main__":
    #### Single-objective
    print()
    test_initial_population_int_by_replacement()
    print()
    test_initial_population_int_by_replacement_no_duplicates()
    print()
    test_initial_population_int_by_replacement_no_duplicates2()
    print()
    test_initial_population_float_by_replacement_no_duplicates()
    print()
    test_initial_population_float_by_replacement_no_duplicates2()
    print()
    test_initial_population_float_by_replacement_no_duplicates_None_constraints()
    print()
    
    #### Fallback Strategy Tests
    print("=== Testing Fallback Strategies ===")
    print()
    test_default_fallback_strategy_is_keep()
    print()
    test_sample_size_fallback_strategy_keep()
    print()
    test_sample_size_fallback_strategy_raise_gene_constraint_error()
    print()
    test_sample_size_fallback_strategy_expand_sample()
    print()
    test_sample_size_fallback_strategy_switch_to_discrete()
    print()
    test_invalid_fallback_strategy()
    print()
    test_gene_constraint_error_exception_class()
    print()
    test_duplicate_gene_error_exception_class()
    print()
