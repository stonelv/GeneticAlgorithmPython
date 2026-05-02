import pygad
import numpy

"""
Example demonstrating the Run Metrics Recorder feature in PyGAD.

This example shows how to:
1. Record metrics during the genetic algorithm run() method
2. Access the recorded metrics directly
3. Export metrics to a CSV file using to_csv()
4. Visualize metrics using plot_metrics()

The metrics recorded for each generation include:
- generation: Generation number
- time_elapsed: Time taken for this generation (seconds)
- best_fitness: Best fitness value in the population
- mean_fitness: Average fitness value of the population
- diversity: Population diversity (average gene variance)
"""

def fitness_func(ga_instance, solution, solution_idx):
    """
    Simple fitness function that sums the product of solution with predefined inputs.
    This is a classic optimization problem: finding weights that produce a desired output.
    """
    function_inputs = [4, -2, 3.5, 5, -11, -4.7]
    desired_output = 44
    
    output = numpy.sum(solution * function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness

def on_generation(ga_instance):
    """
    Callback function called after each generation.
    Demonstrates how to access metrics during the run.
    """
    if ga_instance.generations_completed % 10 == 0:
        print(f"Generation: {ga_instance.generations_completed}")
        if ga_instance.run_metrics and len(ga_instance.run_metrics['generation']) > 0:
            latest_idx = len(ga_instance.run_metrics['generation']) - 1
            print(f"  Best Fitness: {ga_instance.run_metrics['best_fitness'][latest_idx]:.6f}")
            print(f"  Mean Fitness: {ga_instance.run_metrics['mean_fitness'][latest_idx]:.6f}")
            print(f"  Diversity: {ga_instance.run_metrics['diversity'][latest_idx]:.6f}")

# Configure the genetic algorithm
num_generations = 50
num_parents_mating = 4
sol_per_pop = 20
num_genes = 6

print("=" * 60)
print("PyGAD Run Metrics Recorder Example")
print("=" * 60)
print(f"\nConfiguration:")
print(f"  - Number of generations: {num_generations}")
print(f"  - Population size: {sol_per_pop}")
print(f"  - Number of genes: {num_genes}")
print(f"  - Parents mating: {num_parents_mating}")

# Create the GA instance
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       fitness_func=fitness_func,
                       on_generation=on_generation,
                       random_seed=42)

print("\n" + "=" * 60)
print("Running the Genetic Algorithm...")
print("=" * 60)

# Run the GA - metrics are automatically recorded during run()
ga_instance.run()

print("\n" + "=" * 60)
print("Accessing Recorded Metrics")
print("=" * 60)

# Check if metrics were recorded
if ga_instance.run_metrics is not None:
    num_metrics = len(ga_instance.run_metrics['generation'])
    print(f"\nTotal generations recorded: {num_metrics}")
    print(f"\nMetrics keys available: {list(ga_instance.run_metrics.keys())}")
    
    # Display first few generations
    print("\nFirst 5 generations:")
    print(f"{'Generation':<12} {'Time (s)':<12} {'Best Fitness':<15} {'Mean Fitness':<15} {'Diversity':<15}")
    print("-" * 70)
    for i in range(min(5, num_metrics)):
        gen = ga_instance.run_metrics['generation'][i]
        t = ga_instance.run_metrics['time_elapsed'][i]
        best = ga_instance.run_metrics['best_fitness'][i]
        mean = ga_instance.run_metrics['mean_fitness'][i]
        div = ga_instance.run_metrics['diversity'][i]
        print(f"{gen:<12} {t:<12.6f} {best:<15.6f} {mean:<15.6f} {div:<15.6f}")
    
    # Display last few generations
    print("\nLast 5 generations:")
    print(f"{'Generation':<12} {'Time (s)':<12} {'Best Fitness':<15} {'Mean Fitness':<15} {'Diversity':<15}")
    print("-" * 70)
    start_idx = max(0, num_metrics - 5)
    for i in range(start_idx, num_metrics):
        gen = ga_instance.run_metrics['generation'][i]
        t = ga_instance.run_metrics['time_elapsed'][i]
        best = ga_instance.run_metrics['best_fitness'][i]
        mean = ga_instance.run_metrics['mean_fitness'][i]
        div = ga_instance.run_metrics['diversity'][i]
        print(f"{gen:<12} {t:<12.6f} {best:<15.6f} {mean:<15.6f} {div:<15.6f}")
    
    # Calculate summary statistics
    total_time = sum(ga_instance.run_metrics['time_elapsed'])
    avg_time = total_time / num_metrics
    max_diversity = max(ga_instance.run_metrics['diversity'])
    min_diversity = min(ga_instance.run_metrics['diversity'])
    
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"\nTotal execution time: {total_time:.4f} seconds")
    print(f"Average time per generation: {avg_time:.4f} seconds")
    print(f"Maximum diversity: {max_diversity:.6f}")
    print(f"Minimum diversity: {min_diversity:.6f}")
    print(f"Final best fitness: {ga_instance.run_metrics['best_fitness'][-1]:.6f}")
    print(f"Final mean fitness: {ga_instance.run_metrics['mean_fitness'][-1]:.6f}")

print("\n" + "=" * 60)
print("Exporting Metrics to CSV")
print("=" * 60)

# Export to CSV
try:
    ga_instance.to_csv('ga_run_metrics.csv')
    print("\nMetrics successfully exported to 'ga_run_metrics.csv'")
    print("\nCSV file contains the following columns:")
    print("  - generation: Generation number")
    print("  - time_elapsed: Time taken for the generation (seconds)")
    print("  - best_fitness: Best fitness value")
    print("  - mean_fitness: Average fitness value")
    print("  - diversity: Population diversity (gene variance)")
except Exception as e:
    print(f"Error exporting to CSV: {e}")

print("\n" + "=" * 60)
print("Visualizing Metrics")
print("=" * 60)

# Plot metrics
print("\nGenerating metrics visualization...")
print("The plot will show 4 subplots:")
print("  1. Best Fitness Over Generations")
print("  2. Mean Fitness Over Generations")
print("  3. Time Elapsed Per Generation")
print("  4. Population Diversity Over Generations")

try:
    # Create the plot
    fig = ga_instance.plot_metrics(
        title="Genetic Algorithm Run Metrics",
        save_dir='ga_run_metrics.png'
    )
    print("\nPlot saved as 'ga_run_metrics.png'")
except Exception as e:
    print(f"Error generating plot: {e}")

# Get the best solution
print("\n" + "=" * 60)
print("Best Solution")
print("=" * 60)

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"\nParameters of the best solution: {solution}")
print(f"Fitness value of the best solution: {solution_fitness}")

if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

print("\n" + "=" * 60)
print("Example Complete")
print("=" * 60)
print("\nFiles created:")
print("  - ga_run_metrics.csv: CSV file with all recorded metrics")
print("  - ga_run_metrics.png: Visualization of the metrics")
print("\nTips:")
print("  - Metrics are automatically recorded during run()")
print("  - Access raw data via ga_instance.run_metrics dictionary")
print("  - Use to_csv() for spreadsheet analysis")
print("  - Use plot_metrics() for visual analysis")
