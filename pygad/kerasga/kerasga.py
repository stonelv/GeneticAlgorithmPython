import copy
import numpy
import tensorflow.keras

_model_cache = {}


def model_weights_as_vector(model):
    """
    Reshapes the Keras model weight as a vector.

    Parameters
    ----------
    model : TYPE
        The Keras model.

    Returns
    -------
    TYPE
        The weights as a 1D vector.

    """
    weights_vector = []

    for layer in model.layers: # model.get_weights():
        if layer.trainable:
            layer_weights = layer.get_weights()
            for l_weights in layer_weights:
                vector = numpy.reshape(l_weights, (l_weights.size))
                weights_vector.extend(vector)

    return numpy.array(weights_vector)

def model_weights_as_matrix(model, weights_vector):
    """
    Reshapes the PyGAD 1D solution as a Keras weight matrix.

    Parameters
    ----------
    model : TYPE
        The Keras model.
    weights_vector : TYPE
        The PyGAD solution as a 1D vector.

    Returns
    -------
    weights_matrix : TYPE
        The Keras weights as a matrix.

    """
    weights_matrix = []

    start = 0
    for layer_idx, layer in enumerate(model.layers): # model.get_weights():
    # for w_matrix in model.get_weights():
        layer_weights = layer.get_weights()
        if layer.trainable:
            for l_weights in layer_weights:
                layer_weights_shape = l_weights.shape
                layer_weights_size = l_weights.size
        
                layer_weights_vector = weights_vector[start:start + layer_weights_size]
                layer_weights_matrix = numpy.reshape(layer_weights_vector, (layer_weights_shape))
                weights_matrix.append(layer_weights_matrix)
        
                start = start + layer_weights_size
        else:
            for l_weights in layer_weights:
                weights_matrix.append(l_weights)

    return weights_matrix

def predict(model, 
            solution, 
            data, 
            batch_size=None,
            verbose=0,
            steps=None,
            reuse_model=None):
    """
    Use the PyGAD's solution to make predictions using the Keras model.

    Parameters
    ----------
    model : TYPE
        The Keras model.
    solution : TYPE
        A single PyGAD solution as 1D vector.
    data : TYPE
        The data or a generator.
    batch_size : TYPE, optional
        The batch size (i.e. number of samples per step or batch). The default is None. Check documentation of the Keras Model.predict() method for more information.
    verbose : TYPE, optional
        Verbosity mode. The default is 0. Check documentation of the Keras Model.predict() method for more information.
    steps : TYPE, optional
        The total number of steps (batches of samples). The default is None. Check documentation of the Keras Model.predict() method for more information.
    reuse_model : None, bool, or tensorflow.keras.Model, optional
        Controls model cloning behavior to reduce overhead:
        - None or False (default): Always clone the model using clone_model() for each prediction.
          This preserves the original behavior but may be slower for frequent predictions.
        - True: Automatically cache and reuse a single cloned model instance. The model is cloned
          only once (on first call) and subsequent calls only update its weights. This is much
          faster but uses a module-level cache keyed by the original model's id().
        - A Keras Model instance: Directly reuse the provided model instance. Only its weights
          are updated. This gives the user full control over the cached model's lifecycle.

    Returns
    -------
    predictions : TYPE
        The Keras model predictions.

    """
    global _model_cache
    
    solution_weights = model_weights_as_matrix(model=model,
                                               weights_vector=solution)
    
    if reuse_model is None or reuse_model is False:
        _model = tensorflow.keras.models.clone_model(model)
        _model.set_weights(solution_weights)
    elif reuse_model is True:
        model_id = id(model)
        if model_id not in _model_cache:
            _model_cache[model_id] = tensorflow.keras.models.clone_model(model)
        _model = _model_cache[model_id]
        _model.set_weights(solution_weights)
    else:
        _model = reuse_model
        _model.set_weights(solution_weights)
    
    predictions = _model.predict(x=data,
                                 batch_size=batch_size,
                                 verbose=verbose,
                                 steps=steps)

    return predictions


def clear_model_cache(model=None):
    """
    Clear the cached model instances.

    Parameters
    ----------
    model : tensorflow.keras.Model, optional
        If provided, only clear the cache entry for this specific model.
        If None (default), clear all cached models.

    This is useful when:
    - You're done using predict() with reuse_model=True and want to free memory
    - The original model's architecture has changed and you need a fresh clone
    """
    global _model_cache
    if model is not None:
        model_id = id(model)
        if model_id in _model_cache:
            del _model_cache[model_id]
    else:
        _model_cache.clear()

class KerasGA:

    def __init__(self, model, num_solutions):

        """
        Creates an instance of the KerasGA class to build a population of model parameters.

        model: A Keras model class.
        num_solutions: Number of solutions in the population. Each solution has different model parameters.
        """
        
        self.model = model

        self.num_solutions = num_solutions

        # A list holding references to all the solutions (i.e. networks) used in the population.
        self.population_weights = self.create_population()

    def create_population(self):

        """
        Creates the initial population of the genetic algorithm as a list of networks' weights (i.e. solutions). Each element in the list holds a different weights of the Keras model.

        The method returns a list holding the weights of all solutions.
        """

        model_weights_vector = model_weights_as_vector(model=self.model)

        net_population_weights = []
        net_population_weights.append(model_weights_vector)

        for idx in range(self.num_solutions-1):

            net_weights = copy.deepcopy(model_weights_vector)
            net_weights = numpy.array(net_weights) + numpy.random.uniform(low=-1.0, high=1.0, size=model_weights_vector.size)

            # Appending the weights to the population.
            net_population_weights.append(net_weights)

        return net_population_weights
