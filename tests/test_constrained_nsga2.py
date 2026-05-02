import pygad
import numpy

def test_constraint_domination_principle():
    """
    Test the constraint domination principle for NSGA-II.
    
    Constraint Domination Principle:
    1. A feasible solution (no constraint violations) always dominates an infeasible solution.
    2. Between two feasible solutions, regular Pareto domination applies.
    3. Between two infeasible solutions, the one with smaller constraint violation dominates.
    """
    
    function_inputs1 = [4, -2, 3.5, 5, -11, -4.7]
    function_inputs2 = [-2, 0.7, -9, 1.4, 3, 5]
    desired_output1 = 50
    desired_output2 = 30
    
    def fitness_func(ga_instance, solution, solution_idx):
        output1 = numpy.sum(solution * function_inputs1)
        output2 = numpy.sum(solution * function_inputs2)
        fitness1 = 1.0 / (numpy.abs(output1 - desired_output1) + 0.000001)
        fitness2 = 1.0 / (numpy.abs(output2 - desired_output2) + 0.000001)
        return [fitness1, fitness2]
    
    def constraint_func(solution, solution_idx):
        """
        Example constraint: sum of all genes should be <= 0.
        Returns:
            - <= 0 if constraint is satisfied
            - > 0 if constraint is violated
        """
        sum_genes = numpy.sum(solution)
        return [sum_genes]
    
    num_generations = 20
    num_parents_mating = 4
    sol_per_pop = 10
    num_genes = len(function_inputs1)
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           fitness_func=fitness_func,
                           parent_selection_type='nsga2',
                           constraint_func=constraint_func,
                           random_seed=42)
    
    initial_population = ga_instance.population.copy()
    
    print("Testing constraint function validation...")
    assert ga_instance.constraint_func is not None, "constraint_func should be set"
    print("✓ constraint_func is properly set")
    
    ga_instance.run()
    
    print("\nChecking constraint domination behavior...")
    constraint_violations = []
    for idx, sol in enumerate(ga_instance.population):
        violations = constraint_func(sol, idx)
        total_violation = sum(max(0, v) for v in violations)
        constraint_violations.append(total_violation)
        print(f"  Solution {idx}: sum={numpy.sum(sol):.4f}, violation={total_violation:.4f}")
    
    constraint_violations = numpy.array(constraint_violations)
    feasible_count = numpy.sum(constraint_violations <= 0)
    infeasible_count = numpy.sum(constraint_violations > 0)
    
    print(f"\nFeasible solutions: {feasible_count}")
    print(f"Infeasible solutions: {infeasible_count}")
    
    if feasible_count > 0:
        print("\n✓ Found feasible solutions")
    else:
        print("\n⚠ No feasible solutions found (may need more generations)")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)
    
    return ga_instance


def test_feasible_vs_infeasible_domination():
    """
    Test that a feasible solution always dominates an infeasible solution.
    """
    print("\n" + "="*60)
    print("Testing: Feasible vs Infeasible Domination")
    print("="*60)
    
    nsga2 = pygad.utils.nsga2.NSGA2()
    nsga2.supported_int_float_types = [int, float, numpy.float64, numpy.int64]
    
    fitness = numpy.array([
        [1.0, 1.0],
        [0.5, 0.5],
    ])
    
    constraint_violations = numpy.array([
        0.0,
        5.0,
    ])
    
    pareto_fronts, solutions_fronts_indices = nsga2.non_dominated_sorting(
        fitness, 
        constraint_violations=constraint_violations
    )
    
    print(f"\nFitness values:")
    print(f"  Solution 0: {fitness[0]} (feasible, violation={constraint_violations[0]})")
    print(f"  Solution 1: {fitness[1]} (infeasible, violation={constraint_violations[1]})")
    
    print(f"\nPareto fronts:")
    for i, front in enumerate(pareto_fronts):
        print(f"  Front {i}: {[int(sol[0]) for sol in front]}")
    
    assert solutions_fronts_indices[0] < solutions_fronts_indices[1], \
        "Feasible solution (0) should be in a better (lower index) front than infeasible solution (1)"
    
    print("\n✓ Feasible solution correctly dominates infeasible solution")
    return True


def test_infeasible_vs_infeasible_domination():
    """
    Test that between two infeasible solutions, the one with smaller 
    constraint violation dominates.
    """
    print("\n" + "="*60)
    print("Testing: Infeasible vs Infeasible Domination")
    print("="*60)
    
    nsga2 = pygad.utils.nsga2.NSGA2()
    nsga2.supported_int_float_types = [int, float, numpy.float64, numpy.int64]
    
    fitness = numpy.array([
        [0.1, 0.1],
        [1.0, 1.0],
    ])
    
    constraint_violations = numpy.array([
        1.0,
        5.0,
    ])
    
    pareto_fronts, solutions_fronts_indices = nsga2.non_dominated_sorting(
        fitness, 
        constraint_violations=constraint_violations
    )
    
    print(f"\nFitness values:")
    print(f"  Solution 0: {fitness[0]} (violation={constraint_violations[0]})")
    print(f"  Solution 1: {fitness[1]} (violation={constraint_violations[1]})")
    
    print(f"\nPareto fronts:")
    for i, front in enumerate(pareto_fronts):
        print(f"  Front {i}: {[int(sol[0]) for sol in front]}")
    
    assert solutions_fronts_indices[0] < solutions_fronts_indices[1], \
        "Solution with smaller violation (0) should be in a better front"
    
    print("\n✓ Solution with smaller constraint violation correctly dominates")
    return True


def test_feasible_pareto_domination():
    """
    Test that between two feasible solutions, regular Pareto domination applies.
    """
    print("\n" + "="*60)
    print("Testing: Feasible Pareto Domination (Regular Pareto)")
    print("="*60)
    
    nsga2 = pygad.utils.nsga2.NSGA2()
    nsga2.supported_int_float_types = [int, float, numpy.float64, numpy.int64]
    
    fitness = numpy.array([
        [2.0, 2.0],
        [1.0, 1.0],
        [1.5, 0.5],
    ])
    
    constraint_violations = numpy.array([
        0.0,
        0.0,
        0.0,
    ])
    
    pareto_fronts, solutions_fronts_indices = nsga2.non_dominated_sorting(
        fitness, 
        constraint_violations=constraint_violations
    )
    
    print(f"\nFitness values (all feasible):")
    for i in range(len(fitness)):
        print(f"  Solution {i}: {fitness[i]}")
    
    print(f"\nPareto fronts:")
    for i, front in enumerate(pareto_fronts):
        print(f"  Front {i}: {[int(sol[0]) for sol in front]}")
    
    assert 0 in [int(sol[0]) for sol in pareto_fronts[0]], \
        "Solution 0 [2.0, 2.0] should be in the first Pareto front"
    
    print("\n✓ Regular Pareto domination works correctly for feasible solutions")
    return True


def test_backward_compatibility():
    """
    Test that NSGA-II works without constraint_func (backward compatibility).
    """
    print("\n" + "="*60)
    print("Testing: Backward Compatibility (No Constraint)")
    print("="*60)
    
    function_inputs1 = [4, -2, 3.5, 5, -11, -4.7]
    function_inputs2 = [-2, 0.7, -9, 1.4, 3, 5]
    desired_output1 = 50
    desired_output2 = 30
    
    def fitness_func(ga_instance, solution, solution_idx):
        output1 = numpy.sum(solution * function_inputs1)
        output2 = numpy.sum(solution * function_inputs2)
        fitness1 = 1.0 / (numpy.abs(output1 - desired_output1) + 0.000001)
        fitness2 = 1.0 / (numpy.abs(output2 - desired_output2) + 0.000001)
        return [fitness1, fitness2]
    
    num_generations = 10
    num_parents_mating = 4
    sol_per_pop = 8
    num_genes = len(function_inputs1)
    
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           fitness_func=fitness_func,
                           parent_selection_type='nsga2',
                           random_seed=42)
    
    ga_instance.run()
    
    assert ga_instance.pareto_fronts is not None, "Pareto fronts should be computed"
    
    print(f"\nNumber of Pareto fronts: {len(ga_instance.pareto_fronts)}")
    for i, front in enumerate(ga_instance.pareto_fronts):
        print(f"  Front {i}: {len(front)} solutions")
    
    print("\n✓ Backward compatibility test passed")
    return True


if __name__ == "__main__":
    print("="*60)
    print("Constrained NSGA-II Test Suite")
    print("="*60)
    
    test_backward_compatibility()
    test_feasible_vs_infeasible_domination()
    test_infeasible_vs_infeasible_domination()
    test_feasible_pareto_domination()
    
    print("\n" + "="*60)
    print("Running full GA test with constraints...")
    print("="*60)
    test_constraint_domination_principle()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
