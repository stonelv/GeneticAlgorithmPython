"""
Test suite for the Run Metrics Recorder feature in PyGAD.

This test suite verifies:
1. run_metrics data structure and recording
2. to_csv() method functionality
3. plot_metrics() method functionality (API hookup)
4. Full integration: run() -> to_csv() -> plot_metrics()

Run with pytest:
    pytest tests/test_run_metrics.py -v
"""

import pygad
import numpy
import os
import tempfile
import matplotlib

# Use Agg backend for headless testing (no GUI needed)
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Global constants for testing
NUM_GENERATIONS = 5
NUM_PARENTS_MATING = 2
SOL_PER_POP = 4
NUM_GENES = 6
RANDOM_SEED = 42


def fitness_func(ga_instance, solution, solution_idx):
    """Simple fitness function for testing."""
    function_inputs = [4, -2, 3.5, 5, -11, -4.7]
    desired_output = 44
    output = numpy.sum(solution * function_inputs)
    fitness = 1.0 / (numpy.abs(output - desired_output) + 0.000001)
    return fitness


def fitness_func_multi(ga_instance, solution, solution_idx):
    """Multi-objective fitness function for testing."""
    return [numpy.sum(solution**2), numpy.sum(solution)]


class TestRunMetricsStructure:
    """Tests for the run_metrics data structure and recording."""

    def setup_method(self):
        """Create a fresh GA instance for each test."""
        self.ga_instance = pygad.GA(
            num_generations=NUM_GENERATIONS,
            num_parents_mating=NUM_PARENTS_MATING,
            sol_per_pop=SOL_PER_POP,
            num_genes=NUM_GENES,
            fitness_func=fitness_func,
            random_seed=RANDOM_SEED,
            suppress_warnings=True
        )

    def test_run_metrics_none_before_run(self):
        """run_metrics should be None before calling run()."""
        assert self.ga_instance.run_metrics is None

    def test_run_metrics_exists_after_run(self):
        """run_metrics should be a dict after calling run()."""
        self.ga_instance.run()
        assert self.ga_instance.run_metrics is not None
        assert isinstance(self.ga_instance.run_metrics, dict)

    def test_run_metrics_keys(self):
        """run_metrics should have all expected keys."""
        self.ga_instance.run()
        expected_keys = ['generation', 'time_elapsed', 'best_fitness', 'mean_fitness', 'diversity']
        for key in expected_keys:
            assert key in self.ga_instance.run_metrics, f"Missing key: {key}"

    def test_run_metrics_count_matches_num_generations(self):
        """The number of recorded metrics should equal num_generations."""
        self.ga_instance.run()
        num_recorded = len(self.ga_instance.run_metrics['generation'])
        assert num_recorded == NUM_GENERATIONS, \
            f"Expected {NUM_GENERATIONS} records, got {num_recorded}"

    def test_run_metrics_generation_numbers(self):
        """Generation numbers should be 0-based and sequential."""
        self.ga_instance.run()
        generations = self.ga_instance.run_metrics['generation']
        expected = list(range(NUM_GENERATIONS))
        assert generations == expected, \
            f"Expected generations {expected}, got {generations}"

    def test_run_metrics_time_elapsed_valid(self):
        """time_elapsed should be non-negative numbers."""
        self.ga_instance.run()
        time_elapsed = self.ga_instance.run_metrics['time_elapsed']
        
        assert len(time_elapsed) == NUM_GENERATIONS
        
        for i, t in enumerate(time_elapsed):
            assert isinstance(t, (int, float, numpy.floating)), \
                f"time_elapsed[{i}] should be a number, got {type(t)}"
            assert t >= 0, f"time_elapsed[{i}] should be non-negative, got {t}"

    def test_run_metrics_best_fitness_valid(self):
        """best_fitness should contain valid numbers."""
        self.ga_instance.run()
        best_fitness = self.ga_instance.run_metrics['best_fitness']
        
        assert len(best_fitness) == NUM_GENERATIONS
        
        for i, bf in enumerate(best_fitness):
            assert isinstance(bf, (int, float, numpy.floating)), \
                f"best_fitness[{i}] should be a number, got {type(bf)}"

    def test_run_metrics_mean_fitness_valid(self):
        """mean_fitness should contain valid numbers."""
        self.ga_instance.run()
        mean_fitness = self.ga_instance.run_metrics['mean_fitness']
        
        assert len(mean_fitness) == NUM_GENERATIONS
        
        for i, mf in enumerate(mean_fitness):
            assert isinstance(mf, (int, float, numpy.floating)), \
                f"mean_fitness[{i}] should be a number, got {type(mf)}"

    def test_run_metrics_diversity_valid(self):
        """diversity should contain valid numbers (or NaN)."""
        self.ga_instance.run()
        diversity = self.ga_instance.run_metrics['diversity']
        
        assert len(diversity) == NUM_GENERATIONS
        
        for i, d in enumerate(diversity):
            is_valid = isinstance(d, (int, float, numpy.floating)) or numpy.isnan(d)
            assert is_valid, f"diversity[{i}] should be a number or NaN, got {type(d)}"

    def test_run_metrics_best_fitness_matches_best_solutions_fitness(self):
        """run_metrics['best_fitness'] should match best_solutions_fitness."""
        self.ga_instance.run()
        rm_best_fitness = self.ga_instance.run_metrics['best_fitness']
        bs_fitness = self.ga_instance.best_solutions_fitness
        
        for i in range(NUM_GENERATIONS):
            assert abs(rm_best_fitness[i] - bs_fitness[i]) < 1e-10, \
                f"Mismatch at index {i}: run_metrics={rm_best_fitness[i]}, best_solutions_fitness={bs_fitness[i]}"

    def test_run_metrics_multiple_runs(self):
        """Multiple calls to run() should append to metrics."""
        self.ga_instance.run()
        first_count = len(self.ga_instance.run_metrics['generation'])
        
        self.ga_instance.run()
        second_count = len(self.ga_instance.run_metrics['generation'])
        
        assert second_count == first_count + NUM_GENERATIONS, \
            f"Expected {first_count + NUM_GENERATIONS} records after second run, got {second_count}"


class TestRunMetricsMultiObjective:
    """Tests for multi-objective optimization."""

    def test_run_metrics_multi_objective(self):
        """run_metrics should work with multi-objective fitness."""
        ga_instance = pygad.GA(
            num_generations=3,
            num_parents_mating=2,
            sol_per_pop=5,
            num_genes=3,
            fitness_func=fitness_func_multi,
            parent_selection_type="nsga2",
            suppress_warnings=True
        )
        ga_instance.run()
        
        best_fitness = ga_instance.run_metrics['best_fitness']
        mean_fitness = ga_instance.run_metrics['mean_fitness']
        
        # In multi-objective, fitness should be lists/tuples/arrays
        for i, bf in enumerate(best_fitness):
            assert type(bf) in [list, tuple, numpy.ndarray], \
                f"best_fitness[{i}] should be iterable for multi-objective, got {type(bf)}"
        
        for i, mf in enumerate(mean_fitness):
            assert type(mf) in [list, tuple, numpy.ndarray], \
                f"mean_fitness[{i}] should be iterable for multi-objective, got {type(mf)}"


class TestToCSV:
    """Tests for the to_csv() method."""

    def setup_method(self):
        """Create a fresh GA instance for each test."""
        self.ga_instance = pygad.GA(
            num_generations=NUM_GENERATIONS,
            num_parents_mating=NUM_PARENTS_MATING,
            sol_per_pop=SOL_PER_POP,
            num_genes=NUM_GENES,
            fitness_func=fitness_func,
            random_seed=RANDOM_SEED,
            suppress_warnings=True
        )

    def test_to_csv_before_run_raises_error(self):
        """to_csv() should raise RuntimeError before run()."""
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            self.ga_instance.to_csv(csv_path)
            assert False, "to_csv() should have raised RuntimeError"
        except RuntimeError:
            pass
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_to_csv_creates_file(self):
        """to_csv() should create a CSV file with correct content."""
        self.ga_instance.run()
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            self.ga_instance.to_csv(csv_path)
            
            # Verify file exists
            assert os.path.exists(csv_path), f"CSV file not created: {csv_path}"
            
            # Verify content
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            # Header + NUM_GENERATIONS data lines
            assert len(lines) == NUM_GENERATIONS + 1, \
                f"Expected {NUM_GENERATIONS + 1} lines, got {len(lines)}"
            
            # Verify header
            header = lines[0].strip().split(',')
            expected_header = ['generation', 'time_elapsed', 'best_fitness', 'mean_fitness', 'diversity']
            assert header == expected_header, f"Expected header {expected_header}, got {header}"
            
            # Verify data lines
            for i in range(1, len(lines)):
                row = lines[i].strip().split(',')
                assert len(row) == 5, f"Row {i} should have 5 columns, got {len(row)}"
                
                # Generation should be an integer
                gen_idx = int(row[0])
                assert gen_idx == i - 1, f"Expected generation {i-1}, got {gen_idx}"
                
                # Time should be a float
                float(row[1])
                
                # Fitness values should be numbers
                float(row[2])
                float(row[3])
                
                # Diversity should be a number or 'NaN'
                if row[4] != 'NaN':
                    float(row[4])
            
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_to_csv_custom_delimiter(self):
        """to_csv() should support custom delimiters."""
        self.ga_instance.run()
        
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            self.ga_instance.to_csv(csv_path, delimiter=';')
            
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            # Header should use semicolon delimiter
            header = lines[0].strip().split(';')
            expected_header = ['generation', 'time_elapsed', 'best_fitness', 'mean_fitness', 'diversity']
            assert header == expected_header
            
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_to_csv_adds_csv_extension(self):
        """to_csv() should add .csv extension if missing."""
        self.ga_instance.run()
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            base_path = f.name
        
        csv_path = base_path + '.csv'
        
        try:
            # Call without .csv extension
            self.ga_instance.to_csv(base_path)
            
            # File should exist with .csv extension
            assert os.path.exists(csv_path), f"CSV file not found at {csv_path}"
            
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)
            if os.path.exists(base_path):
                os.remove(base_path)


class TestPlotMetrics:
    """Tests for the plot_metrics() method."""

    def setup_method(self):
        """Create a fresh GA instance for each test."""
        self.ga_instance = pygad.GA(
            num_generations=NUM_GENERATIONS,
            num_parents_mating=NUM_PARENTS_MATING,
            sol_per_pop=SOL_PER_POP,
            num_genes=NUM_GENES,
            fitness_func=fitness_func,
            random_seed=RANDOM_SEED,
            suppress_warnings=True
        )

    def test_plot_metrics_method_exists(self):
        """plot_metrics should be a callable method of GA instance."""
        assert hasattr(self.ga_instance, 'plot_metrics'), \
            "GA instance should have plot_metrics method"
        assert callable(self.ga_instance.plot_metrics), \
            "plot_metrics should be callable"

    def test_plot_metrics_before_run_raises_error(self):
        """plot_metrics() should raise RuntimeError before run()."""
        try:
            self.ga_instance.plot_metrics(show=False)
            assert False, "plot_metrics() should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_plot_metrics_returns_figure(self):
        """plot_metrics() should return a matplotlib Figure."""
        self.ga_instance.run()
        
        fig = self.ga_instance.plot_metrics(show=False)
        
        assert isinstance(fig, matplotlib.figure.Figure), \
            f"plot_metrics should return matplotlib Figure, got {type(fig)}"
        
        plt.close(fig)

    def test_plot_metrics_all_parameters(self):
        """plot_metrics() should accept all parameters."""
        self.ga_instance.run()
        
        # Test different plot types
        for p_type in ["plot", "scatter", "bar"]:
            fig = self.ga_instance.plot_metrics(
                plot_type=p_type,
                title=f"Test Title - {p_type}",
                xlabel=None,  # This is actually handled internally
                ylabel=None,
                font_size=14,
                figsize=(10, 6),
                colors=["red", "blue", "green", "purple"],
                labels=["A", "B", "C", "D"],
                show=False
            )
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)

    def test_plot_metrics_save_dir(self):
        """plot_metrics() should save figure when save_dir is specified."""
        self.ga_instance.run()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            png_path = f.name
        
        try:
            fig = self.ga_instance.plot_metrics(save_dir=png_path, show=False)
            
            # Verify file was created
            assert os.path.exists(png_path), f"PNG file not created: {png_path}"
            
            plt.close(fig)
            
        finally:
            if os.path.exists(png_path):
                os.remove(png_path)


class TestFullIntegration:
    """Full integration tests for the run metrics recorder feature."""

    def test_full_workflow_single_objective(self):
        """Test complete workflow: run() -> to_csv() -> plot_metrics()"""
        ga_instance = pygad.GA(
            num_generations=3,
            num_parents_mating=2,
            sol_per_pop=5,
            num_genes=4,
            fitness_func=fitness_func,
            random_seed=123,
            suppress_warnings=True
        )
        
        # Step 1: Run the GA
        ga_instance.run()
        
        # Verify metrics structure
        assert ga_instance.run_metrics is not None
        assert len(ga_instance.run_metrics['generation']) == 3
        
        # Step 2: Export to CSV
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            ga_instance.to_csv(csv_path)
            assert os.path.exists(csv_path)
            
            # Step 3: Plot metrics
            fig = ga_instance.plot_metrics(show=False)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)
            
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_full_workflow_multi_objective(self):
        """Test complete workflow with multi-objective optimization."""
        ga_instance = pygad.GA(
            num_generations=3,
            num_parents_mating=2,
            sol_per_pop=5,
            num_genes=3,
            fitness_func=fitness_func_multi,
            parent_selection_type="nsga2",
            random_seed=123,
            suppress_warnings=True
        )
        
        # Step 1: Run the GA
        ga_instance.run()
        
        # Verify metrics structure
        assert ga_instance.run_metrics is not None
        assert len(ga_instance.run_metrics['generation']) == 3
        
        # Step 2: Export to CSV
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            ga_instance.to_csv(csv_path)
            assert os.path.exists(csv_path)
            
            # Step 3: Plot metrics (should work with multi-objective)
            fig = ga_instance.plot_metrics(show=False)
            assert isinstance(fig, matplotlib.figure.Figure)
            plt.close(fig)
            
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


if __name__ == "__main__":
    # Run tests when this file is executed directly
    import pytest
    import sys
    
    # Get the directory of this file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run pytest on this file
    sys.exit(pytest.main([
        os.path.join(test_dir, 'test_run_metrics.py'),
        '-v',
        '--tb=short'
    ]))
