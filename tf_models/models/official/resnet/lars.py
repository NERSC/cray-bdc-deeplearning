from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops
from tensorflow.python.ops import variables

class GDMomentumLARSOptimizer(optimizer.Optimizer):
  """Optimizer that implements the gradient descent algorithm.
  """

  def __init__(self, learning_rate, 
               momentum_coeff=0.9,
               weight_decay_coeff=0.0005,
	       lars_coeff=0.001, 
               name="GDMomentumLARS"):
    """Construct a new gradient descent optimizer.
    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to "GradientDescent".
    """
    super(GDMomentumLARSOptimizer, self).__init__(False, name)
    self._learning_rate = learning_rate 
    self._momentum_coeff = momentum_coeff
    self._weight_decay_coeff = weight_decay_coeff
    self._lars_coeff = lars_coeff
   
    # Tensors for learning rate and momentum.  Created in _prepare.
    self._learning_rate_tensor = None
    self._momentum_coeff_tensor = None
    self._weight_decay_tensor = None
    self._lars_tensor = None

  def _create_slots(self, var_list):

    for v in var_list: 
      self._zeros_slot(v, "momentum", self._name)

  def _prepare(self):

    self._learning_rate_tensor = ops.convert_to_tensor(self._learning_rate,
                                                       name="learning_rate")
    self._momentum_coeff_tensor = ops.convert_to_tensor(self._momentum_coeff, 
							name="momentum_coeff")
    self._weight_decay_tensor = ops.convert_to_tensor(self._weight_decay_coeff,
                                                	name="weight_decay_coeff")
    self._lars_tensor = ops.convert_to_tensor(self._lars_coeff,
                                                 	name="lars_coeff")

  def _apply_dense(self, grad, var):

    mom = self.get_slot(var, "momentum")
    lr = math_ops.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    mom_coeff = math_ops.cast(self._momentum_coeff_tensor, var.dtype.base_dtype)
    wd_coeff = math_ops.cast(self._weight_decay_tensor, var.dtype.base_dtype)   
    lars_coeff = math_ops.cast(self._lars_tensor, var.dtype.base_dtype)

    lr_local = lars_coeff * tf.norm(var, ord='euclidean') / (tf.norm(grad, ord='euclidean') + wd_coeff * tf.norm(var, ord='euclidean'))

    mom_t = mom.assign(mom_coeff * mom + lr * lr_local * (grad + wd_coeff * var))
    var_t = var.assign(var - mom_t)

    return tf.group(*[var_t, mom_t]) 

