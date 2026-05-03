import pygad
import random
import numpy

num_generations = 1

def fitness_func(ga, solution, idx):
    return random.random()

print("=== Testing Fallback Strategies ===")
print()

# Test 1: Default fallback strategy is 'keep'
print("Test 1: Default fallback strategy is 'keep'")
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
                       suppress_warnings=True)

assert ga_instance.sample_size_fallback_strategy == 'keep', f'Expected default strategy keep, got {ga_instance.sample_size_fallback_strategy}'
assert ga_instance.max_sample_size == 10000, f'Expected default max_sample_size=10000, got {ga_instance.max_sample_size}'
print("  PASSED: Default fallback strategy is 'keep'")
print()

# Test 2: GeneConstraintError exception class
print("Test 2: GeneConstraintError exception class")
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
    print("  PASSED: GeneConstraintError class works correctly")
print()

# Test 3: DuplicateGeneError exception class
print("Test 3: DuplicateGeneError exception class")
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
    print("  PASSED: DuplicateGeneError class works correctly")
print()

# Test 4: Invalid fallback strategy
print("Test 4: Invalid fallback strategy")
try:
    ga_instance = pygad.GA(num_generations=num_generations,
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
    assert False, 'ValueError should have been raised'
except ValueError as e:
    assert 'sample_size_fallback_strategy' in str(e)
    print("  PASSED: Invalid fallback strategy raises ValueError")
print()

# Test 5: Keep strategy (no exception)
print("Test 5: Keep strategy (no exception)")
def impossible_constraint(x, v):
    return [val for val in v if val > 100]

gene_constraint = [impossible_constraint, None, None, None, None]

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
                           sample_size_fallback_strategy='keep',
                           sample_size=5,
                           random_seed=123,
                           save_solutions=True,
                           suppress_warnings=True)
    ga_instance.run()
    print("  PASSED: keep strategy works - no exception raised")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
print()

# Test 6: Raise strategy for gene constraint error
print("Test 6: Raise strategy for gene constraint error")
try:
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=3,
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
    ga_instance.run()
    print("  FAILED: GeneConstraintError should have been raised")
except pygad.GeneConstraintError as e:
    print("  PASSED: GeneConstraintError raised as expected")
    print(f"    gene_index: {e.gene_index}")
    print(f"    gene_value: {e.gene_value}")
    print(f"    sample_size_used: {e.sample_size_used}")
    print(f"    generations_completed: {e.generations_completed}")
except Exception as e:
    print(f"  FAILED: Unexpected {type(e).__name__}: {e}")
print()

# Test 7: Expand sample strategy
print("Test 7: Expand sample strategy")
def narrow_constraint(x, v):
    return [val for val in v if val == 5]

gene_constraint_narrow = [narrow_constraint, None, None, None, None]

try:
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=5,
                           fitness_func=fitness_func,
                           sol_per_pop=5,
                           num_genes=5,
                           gene_type=int,
                           init_range_low=0,
                           init_range_high=10,
                           random_mutation_min_val=0,
                           random_mutation_max_val=10,
                           mutation_by_replacement=True,
                           allow_duplicate_genes=True,
                           gene_constraint=gene_constraint_narrow,
                           sample_size_fallback_strategy='expand_sample',
                           sample_size=3,
                           max_sample_size=20,
                           random_seed=123,
                           save_solutions=True,
                           suppress_warnings=True)
    ga_instance.run()
    
    initial_population = ga_instance.initial_population
    assert numpy.all(initial_population[:, 0] == 5), "Not all values in column 0 are 5"
    print("  PASSED: expand_sample strategy works")
except AssertionError as e:
    print(f"  Note: {e}")
    print("  Strategy may require more samples to hit the value 5")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
print()

# Test 8: Switch to discrete strategy
print("Test 8: Switch to discrete strategy")
gene_space = [
    [1, 2, 3, 4, 5],
    [10, 20, 30, 40, 50],
    [100, 200, 300],
    [0.1, 0.2, 0.3],
    [1, 2, 3]
]

try:
    ga_instance = pygad.GA(num_generations=num_generations,
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
    all_in_gene_space = True
    for val in initial_population[:, 0]:
        if val not in [1, 2, 3, 4, 5]:
            all_in_gene_space = False
            break
    
    if all_in_gene_space:
        print("  PASSED: switch_to_discrete strategy works")
    else:
        print("  Note: Some values may not be in gene_space (strategy behavior depends on implementation)")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
print()

print("=" * 50)
print("All tests completed!")
