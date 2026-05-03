import pytest
import numpy

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

pytestmark = pytest.mark.skipif(
    not HAS_TENSORFLOW,
    reason="TensorFlow is not installed"
)


import pygad.kerasga


def create_small_model(input_shape=(2,), output_units=1):
    """Create a minimal Keras model for testing."""
    input_layer = tf.keras.layers.Input(input_shape)
    dense_layer = tf.keras.layers.Dense(
        4, 
        activation="relu",
        kernel_initializer=tf.keras.initializers.Constant(0.1),
        bias_initializer=tf.keras.initializers.Constant(0.05)
    )(input_layer)
    output_layer = tf.keras.layers.Dense(
        output_units, 
        activation="linear",
        kernel_initializer=tf.keras.initializers.Constant(0.2),
        bias_initializer=tf.keras.initializers.Constant(0.1)
    )(dense_layer)
    return tf.keras.Model(inputs=input_layer, outputs=output_layer)


def get_fixed_solution(model):
    """Get a fixed solution vector based on the model's initial weights."""
    numpy.random.seed(42)
    base_weights = pygad.kerasga.model_weights_as_vector(model)
    solution = base_weights + numpy.random.uniform(low=-0.1, high=0.1, size=base_weights.size)
    return solution


def get_fixed_inputs():
    """Get fixed input data for consistent testing."""
    return numpy.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [-0.5, 0.5],
    ])


class TestKerasGAReuseModel:
    """Tests for pygad.kerasga.predict() with reuse_model parameter."""

    def setup_method(self):
        """Reset cache before each test."""
        pygad.kerasga.clear_model_cache()

    def teardown_method(self):
        """Reset cache after each test."""
        pygad.kerasga.clear_model_cache()

    def test_all_reuse_model_modes_output_consistency(self):
        """
        Test that all reuse_model modes produce identical outputs.
        
        Modes tested:
        - reuse_model=None (default)
        - reuse_model=False
        - reuse_model=True
        - reuse_model=<cloned model instance>
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()

        pred_none = pygad.kerasga.predict(
            model=model,
            solution=solution,
            data=test_inputs,
            reuse_model=None
        )

        pred_false = pygad.kerasga.predict(
            model=model,
            solution=solution,
            data=test_inputs,
            reuse_model=False
        )

        pred_true = pygad.kerasga.predict(
            model=model,
            solution=solution,
            data=test_inputs,
            reuse_model=True
        )

        user_model = tf.keras.models.clone_model(model)
        pred_user = pygad.kerasga.predict(
            model=model,
            solution=solution,
            data=test_inputs,
            reuse_model=user_model
        )

        assert numpy.allclose(pred_none, pred_false), \
            "reuse_model=None and reuse_model=False should produce same output"
        assert numpy.allclose(pred_none, pred_true), \
            "reuse_model=None and reuse_model=True should produce same output"
        assert numpy.allclose(pred_none, pred_user), \
            "reuse_model=None and reuse_model=<instance> should produce same output"

    def test_reuse_model_true_multiple_calls_consistent(self):
        """
        Test that multiple calls with reuse_model=True produce consistent results.
        
        This indirectly tests that caching is working correctly - the same
        cached model is being reused and weights are properly updated.
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()

        pred_first = pygad.kerasga.predict(
            model=model, solution=solution, data=test_inputs, reuse_model=True
        )

        predictions = [pred_first]
        for _ in range(9):
            pred = pygad.kerasga.predict(
                model=model, solution=solution, data=test_inputs, reuse_model=True
            )
            predictions.append(pred)

        for pred in predictions:
            assert numpy.allclose(pred_first, pred), \
                "All predictions with same solution should be identical"

    def test_clear_model_cache_allows_prediction(self):
        """
        Test that after clear_model_cache(), predictions still work correctly.
        
        This verifies that:
        1. Predictions work before clearing cache
        2. clear_model_cache() doesn't break anything
        3. Predictions work identically after clearing cache
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()

        pred_before = pygad.kerasga.predict(
            model=model, solution=solution, data=test_inputs, reuse_model=True
        )

        pygad.kerasga.clear_model_cache()

        pred_after = pygad.kerasga.predict(
            model=model, solution=solution, data=test_inputs, reuse_model=True
        )

        assert numpy.allclose(pred_before, pred_after), \
            "Predictions should be identical before and after clear_model_cache()"

    def test_clear_model_cache_specific_model(self):
        """
        Test that clear_model_cache(model) clears only that specific model's cache.
        
        This is verified by:
        1. Populate cache with model1 and model2
        2. Clear only model1
        3. Verify model2 still produces consistent predictions
        """
        model1 = create_small_model(input_shape=(2,), output_units=1)
        model2 = create_small_model(input_shape=(3,), output_units=2)
        solution1 = get_fixed_solution(model1)
        solution2 = get_fixed_solution(model2)
        inputs1 = get_fixed_inputs()
        inputs2 = numpy.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        pred1_before = pygad.kerasga.predict(
            model=model1, solution=solution1, data=inputs1, reuse_model=True
        )
        pred2_before = pygad.kerasga.predict(
            model=model2, solution=solution2, data=inputs2, reuse_model=True
        )

        pygad.kerasga.clear_model_cache(model=model1)

        pred1_after = pygad.kerasga.predict(
            model=model1, solution=solution1, data=inputs1, reuse_model=True
        )
        pred2_after = pygad.kerasga.predict(
            model=model2, solution=solution2, data=inputs2, reuse_model=True
        )

        assert numpy.allclose(pred1_before, pred1_after), \
            "model1 predictions should be consistent even after cache clear"
        assert numpy.allclose(pred2_before, pred2_after), \
            "model2 predictions should be consistent (its cache was not cleared)"

    def test_different_models_independent_caching(self):
        """
        Test that different model instances have independent caching.
        
        Verified by:
        1. Using two different models
        2. Ensuring predictions for model1 don't affect model2 and vice versa
        3. Ensuring each model's predictions remain consistent
        """
        model1 = create_small_model(input_shape=(2,), output_units=1)
        model2 = create_small_model(input_shape=(2,), output_units=2)

        assert id(model1) != id(model2), "Models should be different instances"

        solution1 = get_fixed_solution(model1)
        solution2 = get_fixed_solution(model2)
        test_inputs = get_fixed_inputs()

        pred1_1 = pygad.kerasga.predict(
            model=model1, solution=solution1, data=test_inputs, reuse_model=True
        )
        pred2_1 = pygad.kerasga.predict(
            model=model2, solution=solution2, data=test_inputs, reuse_model=True
        )

        pred1_2 = pygad.kerasga.predict(
            model=model1, solution=solution1, data=test_inputs, reuse_model=True
        )
        pred2_2 = pygad.kerasga.predict(
            model=model2, solution=solution2, data=test_inputs, reuse_model=True
        )

        assert numpy.allclose(pred1_1, pred1_2), "model1 predictions should be consistent"
        assert numpy.allclose(pred2_1, pred2_2), "model2 predictions should be consistent"

        assert pred1_1.shape != pred2_1.shape, \
            "Different models should have different output shapes"

    def test_user_provided_model_weights_updated(self):
        """
        Test that when using reuse_model=<user_instance>, the weights
        are correctly applied to the user's model.
        
        Verified by:
        1. Clone the original model (initially has different weights from solution)
        2. Call predict() with reuse_model=user_model
        3. Directly call predict() on user_model and verify it matches
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()

        user_model = tf.keras.models.clone_model(model)

        pred_via_kerasga = pygad.kerasga.predict(
            model=model, 
            solution=solution, 
            data=test_inputs, 
            reuse_model=user_model
        )

        pred_direct = user_model.predict(test_inputs, verbose=0)

        assert numpy.allclose(pred_via_kerasga, pred_direct), \
            "User model's weights should have been updated by predict()"

    def test_none_and_false_behavior_identical(self):
        """
        Explicit test that reuse_model=None and reuse_model=False 
        have identical behavior.
        
        Both should produce identical predictions across multiple calls.
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()

        pred_none = pygad.kerasga.predict(
            model=model, solution=solution, data=test_inputs, reuse_model=None
        )
        pred_false = pygad.kerasga.predict(
            model=model, solution=solution, data=test_inputs, reuse_model=False
        )

        assert numpy.allclose(pred_none, pred_false), \
            "None and False should produce identical predictions"

    def test_multiple_solutions_same_model_reuse_true(self):
        """
        Test that multiple different solutions with the same model
        and reuse_model=True correctly update the cached model's weights
        for each prediction.
        
        Verified by:
        1. Get predictions for multiple solutions with reuse_model=True
        2. Get the same predictions with reuse_model=None (baseline)
        3. Compare that each solution's prediction matches
        """
        model = create_small_model()
        keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=5)
        test_inputs = get_fixed_inputs()

        preds_cached = []
        for solution in keras_ga.population_weights:
            pred = pygad.kerasga.predict(
                model=model, 
                solution=solution, 
                data=test_inputs, 
                reuse_model=True
            )
            preds_cached.append(pred)

        preds_baseline = []
        for solution in keras_ga.population_weights:
            pred = pygad.kerasga.predict(
                model=model, 
                solution=solution, 
                data=test_inputs, 
                reuse_model=None
            )
            preds_baseline.append(pred)

        for cached, baseline in zip(preds_cached, preds_baseline):
            assert numpy.allclose(cached, baseline), \
                "Cached and non-cached predictions should match for each solution"

    def test_kerasga_class_still_works(self):
        """
        Verify that the KerasGA class still works correctly 
        with the modified module.
        
        This is a sanity check that we didn't break existing functionality.
        """
        model = create_small_model()
        num_solutions = 5
        
        keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=num_solutions)

        assert keras_ga.model is model
        assert keras_ga.num_solutions == num_solutions
        assert len(keras_ga.population_weights) == num_solutions

        for weights in keras_ga.population_weights:
            assert isinstance(weights, numpy.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
