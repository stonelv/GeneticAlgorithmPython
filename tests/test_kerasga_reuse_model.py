import pytest
import numpy

try:
    import tensorflow.keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

pytestmark = pytest.mark.skipif(
    not HAS_TENSORFLOW,
    reason="TensorFlow is not installed"
)


import pygad.kerasga
from pygad.kerasga import kerasga


def create_small_model(input_shape=(2,), output_units=1):
    """Create a minimal Keras model for testing."""
    input_layer = tensorflow.keras.layers.Input(input_shape)
    dense_layer = tensorflow.keras.layers.Dense(
        4, 
        activation="relu",
        kernel_initializer=tensorflow.keras.initializers.Constant(0.1),
        bias_initializer=tensorflow.keras.initializers.Constant(0.05)
    )(input_layer)
    output_layer = tensorflow.keras.layers.Dense(
        output_units, 
        activation="linear",
        kernel_initializer=tensorflow.keras.initializers.Constant(0.2),
        bias_initializer=tensorflow.keras.initializers.Constant(0.1)
    )(dense_layer)
    return tensorflow.keras.Model(inputs=input_layer, outputs=output_layer)


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
        assert len(kerasga._model_cache) == 0

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

        user_model = tensorflow.keras.models.clone_model(model)
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

    def test_reuse_model_true_cache_does_not_grow(self):
        """
        Test that multiple calls with reuse_model=True do not grow the cache.
        
        The cache should contain at most 1 entry per unique model id.
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()

        assert len(kerasga._model_cache) == 0, "Cache should be empty initially"

        pygad.kerasga.predict(model=model, solution=solution, data=test_inputs, reuse_model=True)
        cache_size_after_1 = len(kerasga._model_cache)
        assert cache_size_after_1 == 1, "Cache should have 1 entry after first call"

        for i in range(10):
            pygad.kerasga.predict(model=model, solution=solution, data=test_inputs, reuse_model=True)
        
        cache_size_after_many = len(kerasga._model_cache)
        assert cache_size_after_many == cache_size_after_1, \
            f"Cache size should not grow. Was {cache_size_after_1}, now {cache_size_after_many}"
        
        assert id(model) in kerasga._model_cache, \
            "Cache should be keyed by model's id()"

    def test_clear_model_cache_clears_all(self):
        """
        Test that clear_model_cache() with no arguments clears all cached models.
        """
        model1 = create_small_model(input_shape=(2,), output_units=1)
        model2 = create_small_model(input_shape=(3,), output_units=2)
        solution1 = get_fixed_solution(model1)
        solution2 = get_fixed_solution(model2)
        inputs1 = get_fixed_inputs()
        inputs2 = numpy.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        pygad.kerasga.predict(model=model1, solution=solution1, data=inputs1, reuse_model=True)
        pygad.kerasga.predict(model=model2, solution=solution2, data=inputs2, reuse_model=True)
        
        assert len(kerasga._model_cache) == 2, "Should have 2 cached models"

        pygad.kerasga.clear_model_cache()
        
        assert len(kerasga._model_cache) == 0, "Cache should be empty after clear_model_cache()"

    def test_clear_model_cache_specific_model(self):
        """
        Test that clear_model_cache(model) clears only that specific model's cache.
        """
        model1 = create_small_model(input_shape=(2,), output_units=1)
        model2 = create_small_model(input_shape=(3,), output_units=2)
        solution1 = get_fixed_solution(model1)
        solution2 = get_fixed_solution(model2)
        inputs1 = get_fixed_inputs()
        inputs2 = numpy.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        pygad.kerasga.predict(model=model1, solution=solution1, data=inputs1, reuse_model=True)
        pygad.kerasga.predict(model=model2, solution=solution2, data=inputs2, reuse_model=True)
        
        assert len(kerasga._model_cache) == 2

        pygad.kerasga.clear_model_cache(model=model1)
        
        assert len(kerasga._model_cache) == 1, "Should have 1 cached model remaining"
        assert id(model1) not in kerasga._model_cache, "model1 should be removed from cache"
        assert id(model2) in kerasga._model_cache, "model2 should remain in cache"

    def test_cache_cleared_can_predict_again(self):
        """
        Test that after clearing cache, predict() with reuse_model=True works again.
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()

        pred1 = pygad.kerasga.predict(model=model, solution=solution, data=test_inputs, reuse_model=True)
        assert len(kerasga._model_cache) == 1

        pygad.kerasga.clear_model_cache()
        assert len(kerasga._model_cache) == 0

        pred2 = pygad.kerasga.predict(model=model, solution=solution, data=test_inputs, reuse_model=True)
        
        assert len(kerasga._model_cache) == 1, "Cache should be repopulated"
        assert numpy.allclose(pred1, pred2), "Predictions should be identical before and after cache clear"

    def test_different_models_have_separate_cache(self):
        """
        Test that different model instances do not share cache entries.
        
        Each model should have its own cached clone based on id(model).
        """
        model1 = create_small_model(input_shape=(2,), output_units=1)
        model2 = create_small_model(input_shape=(2,), output_units=1)
        
        assert id(model1) != id(model2), "Models should be different instances"
        
        solution1 = get_fixed_solution(model1)
        solution2 = get_fixed_solution(model2)
        test_inputs = get_fixed_inputs()

        pred1_first = pygad.kerasga.predict(model=model1, solution=solution1, data=test_inputs, reuse_model=True)
        assert len(kerasga._model_cache) == 1
        assert id(model1) in kerasga._model_cache
        assert id(model2) not in kerasga._model_cache

        pred2_first = pygad.kerasga.predict(model=model2, solution=solution2, data=test_inputs, reuse_model=True)
        assert len(kerasga._model_cache) == 2
        assert id(model1) in kerasga._model_cache
        assert id(model2) in kerasga._model_cache

        pred1_second = pygad.kerasga.predict(model=model1, solution=solution1, data=test_inputs, reuse_model=True)
        pred2_second = pygad.kerasga.predict(model=model2, solution=solution2, data=test_inputs, reuse_model=True)

        assert numpy.allclose(pred1_first, pred1_second), "model1 predictions should be consistent"
        assert numpy.allclose(pred2_first, pred2_second), "model2 predictions should be consistent"

        assert len(kerasga._model_cache) == 2, "Should still have exactly 2 cache entries"

    def test_user_provided_model_not_in_internal_cache(self):
        """
        Test that when user provides their own reuse_model instance,
        it is NOT stored in the internal _model_cache.
        
        The internal cache is only for reuse_model=True mode.
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()
        user_model = tensorflow.keras.models.clone_model(model)

        assert len(kerasga._model_cache) == 0

        pygad.kerasga.predict(model=model, solution=solution, data=test_inputs, reuse_model=user_model)

        assert len(kerasga._model_cache) == 0, \
            "User-provided model should not be added to internal cache"

    def test_solution_weights_applied_to_user_model(self):
        """
        Test that when using reuse_model=<user_instance>, the weights
        are correctly applied to the user's model.
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()
        
        user_model = tensorflow.keras.models.clone_model(model)
        
        pred_via_user_model = pygad.kerasga.predict(
            model=model, 
            solution=solution, 
            data=test_inputs, 
            reuse_model=user_model
        )
        
        pred_direct = user_model.predict(test_inputs, verbose=0)
        
        assert numpy.allclose(pred_via_user_model, pred_direct), \
            "User model's weights should have been updated by predict()"

    def test_none_and_false_are_equivalent(self):
        """
        Explicit test that reuse_model=None and reuse_model=False 
        have identical behavior (no caching, always clone).
        """
        model = create_small_model()
        solution = get_fixed_solution(model)
        test_inputs = get_fixed_inputs()

        assert len(kerasga._model_cache) == 0

        for _ in range(5):
            pygad.kerasga.predict(model=model, solution=solution, data=test_inputs, reuse_model=None)
            pygad.kerasga.predict(model=model, solution=solution, data=test_inputs, reuse_model=False)

        assert len(kerasga._model_cache) == 0, \
            "reuse_model=None and False should never populate the cache"

    def test_multiple_solutions_same_model_reuse_true(self):
        """
        Test that multiple different solutions with the same model
        and reuse_model=True correctly update the cached model's weights
        for each prediction.
        """
        model = create_small_model()
        keras_ga = pygad.kerasga.KerasGA(model=model, num_solutions=5)
        test_inputs = get_fixed_inputs()

        predictions = []
        for solution in keras_ga.population_weights:
            pred = pygad.kerasga.predict(
                model=model, 
                solution=solution, 
                data=test_inputs, 
                reuse_model=True
            )
            predictions.append(pred)

        assert len(kerasga._model_cache) == 1, \
            "Should only have 1 cached model entry for all solutions"

        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                if not numpy.allclose(predictions[i], predictions[j]):
                    pass
        
        pred_without_cache = pygad.kerasga.predict(
            model=model, 
            solution=keras_ga.population_weights[0], 
            data=test_inputs, 
            reuse_model=None
        )
        
        assert numpy.allclose(predictions[0], pred_without_cache), \
            "First solution prediction should match between cached and non-cached modes"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
