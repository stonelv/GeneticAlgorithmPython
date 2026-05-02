import numpy
import pygad

class NSGA2:

    def __init__(self):
        pass

    def get_non_dominated_set(self, curr_solutions, constraint_violations=None):
        """
        Get the set of non-dominated solutions from the current set of solutions.
        If constraint_violations is provided, constraint domination is used instead of regular Pareto domination.
    
        Parameters
        ----------
        curr_solutions : TYPE
            The set of solutions to find its non-dominated set.
        constraint_violations : numpy.ndarray, optional
            An array where constraint_violations[i] is the total constraint violation for solution i.
            A value of 0 means the solution is feasible (satisfies all constraints).
            A value > 0 means the solution is infeasible (violates some constraints).
            If None, regular Pareto domination is used.
    
        Returns
        -------
        dominated_set : TYPE
            A set of the dominated solutions.
        non_dominated_set : TYPE
            A set of the non-dominated set.
    
        """
        dominated_set = []
        non_dominated_set = []
        for idx1, sol1 in enumerate(curr_solutions):
            is_not_dominated = True
            for idx2, sol2 in enumerate(curr_solutions):
                if idx1 == idx2:
                    continue

                if constraint_violations is not None:
                    sol1_cv = constraint_violations[sol1[0]]
                    sol2_cv = constraint_violations[sol2[0]]
                    
                    sol1_feasible = sol1_cv <= 0.0
                    sol2_feasible = sol2_cv <= 0.0
                    
                    if sol1_feasible and not sol2_feasible:
                        continue
                    elif not sol1_feasible and sol2_feasible:
                        is_not_dominated = False
                        dominated_set.append(sol1)
                        break
                    elif not sol1_feasible and not sol2_feasible:
                        if sol2_cv < sol1_cv:
                            is_not_dominated = False
                            dominated_set.append(sol1)
                            break
                        else:
                            continue

                two_solutions = numpy.array(list(zip(sol1[1], sol2[1])))

                gr_eq = two_solutions[:, 1] >= two_solutions[:, 0]
                gr = two_solutions[:, 1] > two_solutions[:, 0]

                if gr_eq.all() and gr.any():
                    is_not_dominated = False
                    dominated_set.append(sol1)
                    break
                else:
                    pass
    
            if is_not_dominated:
                non_dominated_set.append(sol1)
    
        return dominated_set, non_dominated_set

    def _get_constraint_violations(self):
        """
        Helper method to get constraint violations if constraint_func is defined.
        
        Returns
        -------
        constraint_violations : numpy.ndarray or None
            Total constraint violations for each solution, or None if no constraint_func.
        """
        if hasattr(self, 'constraint_func') and self.constraint_func is not None:
            if hasattr(self, 'population') and self.population is not None:
                return self.calculate_constraint_violations(self.constraint_func, self.population)[0]
        return None

    def non_dominated_sorting(self, fitness, constraint_violations=None):
        """
        Apply non-dominant sorting over the fitness to create the pareto fronts based on non-dominated sorting of the solutions.
        If constraint_violations is provided or constraint_func is defined, constraint domination is used.
    
        Parameters
        ----------
        fitness : TYPE
            An array of the population fitness across all objective function.
        constraint_violations : numpy.ndarray, optional
            An array where constraint_violations[i] is the total constraint violation for solution i.
            If None and self.constraint_func is defined, constraint violations will be calculated automatically.
    
        Returns
        -------
        pareto_fronts : TYPE
            An array of the pareto fronts.

        """

        if type(fitness[0]) in [list, tuple, numpy.ndarray]:
            pass
        elif type(fitness[0]) in self.supported_int_float_types:
            raise TypeError('Non-dominated sorting is only applied when optimizing multi-objective problems.\n\nBut a single-objective optimization problem found as the fitness function returns a single numeric value.\n\nTo use multi-objective optimization, consider returning an iterable of any of these data types:\n1)list\n2)tuple\n3)numpy.ndarray')
        else:
            raise TypeError(f'Non-dominated sorting is only applied when optimizing multi-objective problems. \n\nTo use multi-objective optimization, consider returning an iterable of any of these data types:\n1)list\n2)tuple\n3)numpy.ndarray\n\nBut the data type {type(fitness[0])} found.')

        if constraint_violations is None:
            constraint_violations = self._get_constraint_violations()

        pareto_fronts = []
    
        remaining_set = fitness.copy()
    
        remaining_set = list(zip(range(0, fitness.shape[0]), remaining_set))
    
        solutions_fronts_indices = [-1]*len(remaining_set)
        solutions_fronts_indices = numpy.array(solutions_fronts_indices)
    
        front_index = -1
        while len(remaining_set) > 0:
            front_index += 1

            remaining_set, pareto_front = self.get_non_dominated_set(curr_solutions=remaining_set,
                                                                       constraint_violations=constraint_violations)
            pareto_front = numpy.array(pareto_front, dtype=object)
            pareto_fronts.append(pareto_front)
    
            solutions_indices = pareto_front[:, 0].astype(int)
            solutions_fronts_indices[solutions_indices] = front_index

        return pareto_fronts, solutions_fronts_indices

    def crowding_distance(self, pareto_front, fitness):
        """
        Calculate the crowding distance for all solutions in the current pareto front.
    
        Parameters
        ----------
        pareto_front : TYPE
            The set of solutions in the current pareto front.
        fitness : TYPE
            The fitness of the current population.
    
        Returns
        -------
        obj_crowding_dist_list : TYPE
            A nested list of the values for all objectives alongside their crowding distance.
        crowding_dist_sum : TYPE
            A list of the sum of crowding distances across all objectives for each solution.
        crowding_dist_front_sorted_indices : TYPE
            The indices of the solutions (relative to the current front) sorted by the crowding distance.
        crowding_dist_pop_sorted_indices : TYPE
            The indices of the solutions (relative to the population) sorted by the crowding distance.
        """
    
        pareto_front_no_indices = numpy.array([pareto_front[:, 1][idx] for idx in range(pareto_front.shape[0])])
    
        if pareto_front_no_indices.shape[0] == 1:
            return numpy.array([]), numpy.array([]), numpy.array([0]), pareto_front[:, 0].astype(int)
    
        obj_crowding_dist_list = []
        for obj_idx in range(pareto_front_no_indices.shape[1]):
            obj = pareto_front_no_indices[:, obj_idx]
            obj = list(zip(range(len(obj)), obj, [0]*len(obj)))
            obj = [list(element) for element in obj]
            obj_sorted = sorted(obj, key=lambda x: x[1])
        
            obj_min_val = min(fitness[:, obj_idx])
            obj_max_val = max(fitness[:, obj_idx])
            denominator = obj_max_val - obj_min_val
            if denominator == 0:
                denominator = 0.0000001
    
            inf_val = float('inf')
            obj_sorted[0][2] = inf_val
            obj_sorted[-1][2] = inf_val
    
            if len(obj_sorted) <= 2:
                obj_crowding_dist_list.append(obj_sorted)
                break
    
            for idx in range(1, len(obj_sorted)-1):
                crowding_dist = obj_sorted[idx+1][1] - obj_sorted[idx-1][1]
                crowding_dist = crowding_dist / denominator
                obj_sorted[idx][2] = crowding_dist
        
            obj_sorted = sorted(obj_sorted, key=lambda x: x[0])
            obj_crowding_dist_list.append(obj_sorted)
    
        obj_crowding_dist_list = numpy.array(obj_crowding_dist_list)
        crowding_dist = numpy.array([obj_crowding_dist_list[idx, :, 2] for idx in range(len(obj_crowding_dist_list))])
        crowding_dist_sum = numpy.sum(crowding_dist, axis=0)
    
        crowding_dist_sum = numpy.array(list(zip(obj_crowding_dist_list[0, :, 0], crowding_dist_sum)))
        crowding_dist_sum = sorted(crowding_dist_sum, key=lambda x: x[1], reverse=True)
    
        crowding_dist_front_sorted_indices = numpy.array(crowding_dist_sum)[:, 0]
        crowding_dist_front_sorted_indices = crowding_dist_front_sorted_indices.astype(int)
        crowding_dist_pop_sorted_indices = pareto_front[:, 0]
        crowding_dist_pop_sorted_indices = crowding_dist_pop_sorted_indices[crowding_dist_front_sorted_indices]
        crowding_dist_pop_sorted_indices = crowding_dist_pop_sorted_indices.astype(int)
    
        return obj_crowding_dist_list, crowding_dist_sum, crowding_dist_front_sorted_indices, crowding_dist_pop_sorted_indices

    def sort_solutions_nsga2(self,
                             fitness,
                             find_best_solution=False,
                             constraint_violations=None):
        """
        Sort the solutions based on the fitness.
        The sorting procedure differs based on whether the problem is single-objective or multi-objective optimization.
        If it is multi-objective, then non-dominated sorting and crowding distance are applied.
        At first, non-dominated sorting is applied to classify the solutions into pareto fronts.
        Then the solutions inside each front are sorted using crowded distance.
        The solutions inside pareto front X always come before those in front X+1.
    
        Parameters
        ----------
        fitness: The fitness of the entire population.
        find_best_solution: Whether the method is called only to find the best solution or as part of the PyGAD lifecycle. This is to decide whether the pareto_fronts instance attribute is edited or not.
        constraint_violations : numpy.ndarray, optional
            An array where constraint_violations[i] is the total constraint violation for solution i.
            If None and self.constraint_func is defined, constraint violations will be calculated automatically.

        Returns
        -------
        solutions_sorted : TYPE
            The indices of the sorted solutions.
    
        """
        if type(fitness[0]) in [list, tuple, numpy.ndarray]:
            if constraint_violations is None:
                constraint_violations = self._get_constraint_violations()
                
            solutions_sorted = []
            pareto_fronts, solutions_fronts_indices = self.non_dominated_sorting(fitness,
                                                                                    constraint_violations=constraint_violations)
            if find_best_solution:
                pass
            else:
                self.pareto_fronts = pareto_fronts.copy()
            for pareto_front in pareto_fronts:
                _, _, _, crowding_dist_pop_sorted_indices = self.crowding_distance(pareto_front=pareto_front.copy(),
                                                                                   fitness=fitness)
                crowding_dist_pop_sorted_indices = list(crowding_dist_pop_sorted_indices)
                solutions_sorted.extend(crowding_dist_pop_sorted_indices)
        elif type(fitness[0]) in pygad.GA.supported_int_float_types:
            solutions_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
            solutions_sorted.reverse()
        else:
            raise TypeError(f'Each element in the fitness array must be of a number of an iterable (list, tuple, numpy.ndarray). But the type {type(fitness[0])} found')

        return solutions_sorted

    def calculate_constraint_violations(self, constraint_func, solutions):
        """
        Calculate the constraint violations for each solution.
        
        Parameters
        ----------
        constraint_func : callable
            A function that takes a solution and returns a list/array of constraint violations.
            Each violation value should be:
            - <= 0 if the constraint is satisfied
            - > 0 if the constraint is violated
            The function signature is: constraint_func(solution, solution_idx)
        solutions : numpy.ndarray
            The population of solutions.
            
        Returns
        -------
        total_violations : numpy.ndarray
            An array of total constraint violation for each solution (sum of all violations).
            A value of 0 means feasible, > 0 means infeasible.
        per_constraint_violations : numpy.ndarray
            A 2D array of constraint violations for each constraint of each solution.
        """
        num_solutions = solutions.shape[0]
        per_constraint_violations = []
        
        for idx, sol in enumerate(solutions):
            violations = constraint_func(sol, idx)
            if violations is None:
                violations = []
            if not isinstance(violations, (list, tuple, numpy.ndarray)):
                violations = [violations]
            per_constraint_violations.append(violations)
        
        per_constraint_violations = numpy.array(per_constraint_violations)
        
        positive_violations = numpy.maximum(per_constraint_violations, 0.0)
        total_violations = numpy.sum(positive_violations, axis=1)
        
        return total_violations, per_constraint_violations
