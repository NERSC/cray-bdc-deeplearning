# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order
#CRAY ADDED
#import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()
#DONE CRAY ADDED

from official.resnet import resnet_model
from official.utils.arg_parsers import parsers
from official.utils.export import export
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers
import time
from datetime import datetime



# CRAY ADDED
import ml_comm as mc
import math
from tensorflow.python.ops import math_ops
from lars import GDMomentumLARSOptimizer
#

from tensorflow.python.client import device_lib
#END CRAY

#CRAY ADDED
def get_available_gpus(rank_id,num_local_ranks):

#This function is intended to be used with dense GPU nodes 
#where we need to assign ranks to GPUs with proper affinity 
#and make prevent each rank's instance of tensorflow 
#from allocating multiple GPUs, which results in OOM or 
#other insane behaviors. Might need to do something similar for
#rank assignment per socket, hence the unused local_cpu_list

    gpu_id = str(rank_id%num_local_ranks)
    USE_GPU=True 
    #gpu_id = str([0,1,2,3,4,5,6,7])
    #os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
    gpu_name = '/device:GPU:'+gpu_id
    with tf.device(gpu_name):
        local_device_protos = device_lib.list_local_devices()
    
        local_gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
        local_cpu_list = [x.name for x in local_device_protos if x.device_type == 'CPU']
    assign_dev_list = []
    
    gpu_list=[]
    for i,gpu in enumerate(local_gpu_list):
        print(rank_id, ": checking ", gpu)
        dev_id=str(gpu).split(':')[2]
        print("dev_id = ", dev_id)
        if(str(gpu_id) in str(dev_id)):
            print("found device matching rank id: ", rank_id, " as ", dev_id)
            
            #print("setting up device list from: ", local_device_protos)
            #gpu[-1]=str(rank_id)
            assign_dev_list.append(gpu)
            gpu_list.append(dev_id)
        else:
            assign_dev_list.append(gpu_name)
            gpu_list.append(gpu_id)

            
    if(USE_GPU):

        print(rank_id, ": using gpu_list = ", gpu_list, " and assign_list = ", assign_dev_list," to set visible devices!")#, setting gpu_idx=",gidx)
        gpu_visible = str(gpu_list)
    
        print(rank_id, ": Setting visible device to -> ", gpu_visible)

        os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_visible)
            
            
    return assign_dev_list
#DONE CRAY

# CRAY ADDED
# since this script uses a monitored session, we need to create a hook to initialize
# variables after the session is generated
class BcastTensors(tf.train.SessionRunHook):

  def __init__(self):
    self.bcast = None
    

  def begin(self):
    if not self.bcast:
      new_vars   = mc.broadcast(tf.trainable_variables(),0)
      self.bcast = tf.group(*[tf.assign(v,new_vars[k]) for k,v in enumerate(tf.trainable_variables())])

  def after_create_session(self, session, coord):
     session.run(self.bcast)
     
     #global_step = tf.train.get_global_step()
     #epoch = tf.Print(global_step, [global_step], "Epoch: ")
     #print(session.run(epoch))

# END CRAY ADDED


# CRAY ADDED
# since this script uses a monitored session, we need to create a hook to average metrics at the end of epochs
class AverageTrainMetrics(tf.train.SessionRunHook):

  def __init__(self,loss, metrics, log_freq, batch_size, lr, epoch):
    self.log_freq = log_freq
    self.lr = lr
    self.epoch_true = 0
    self.epoch = epoch
    self.batch_size = batch_size * mc.get_nranks()
    self.both = [loss,metrics['accuracy'][0],lr,epoch]
    self.samps = 0.
    self.perf = 0.
    self.step = 0
    self.sums = 0
    self.start_time = None

  def begin(self):
    self.step = 0
    self.epoch_true += 1
    self.start_time = time.time()

  def before_run(self, run_context):
    self.step += 1
    return tf.train.SessionRunArgs(self.both)  # Asks for loss value.

  def after_run(self, run_context, run_values):
    if self.step % self.log_freq == 0:
      
      current_time = time.time()
      duration = current_time - self.start_time
      self.start_time = current_time
      loss_value = np.asarray([run_values.results[0]],dtype='float32')
      acc_value = np.asarray([run_values.results[1]],dtype='float32')
      #lr = np.asarray([run_values.results[2]],dtype='float32')
      #epoch = np.asarray([run_values.results[3]],dtype='float32')

      examples_per_sec = self.log_freq * self.batch_size / duration
      sec_per_batch = float(duration / self.log_freq)
      mc.average(loss_value)
      mc.average(acc_value)
      
      format_str = ('%s: step %d, loss = %.3f, acc = %.3f, (%.1f examples/sec; %.3f '
                    'sec/batch)')
      if (mc.get_rank() == 0):
          print ("available values = ", run_values)
          print (format_str % (datetime.now(), self.step, loss_value, acc_value,
                           examples_per_sec, sec_per_batch ))
          
      self.samps = self.samps + examples_per_sec
      self.perf  = self.perf + sec_per_batch
      self.sums  = self.sums + 1

  def end(self,session):

    lr = session.run(self.lr)
    #epoch = session.run(self.epoch)
  
    format_str = ('TRAIN Session ENDED at %s: step %d (%.1f examples/sec; %.3f '
                  'sec/batch), learning rate: %.5f')
    self.epoch_true = tf.train.global_step(session, self.epoch)/(self.step+1)
    
    if (mc.get_rank() == 0):
      print('Epoch: ', self.epoch_true)
      print('global_step: %s' % tf.train.global_step(session, self.epoch))
      print (format_str % (datetime.now(), self.step, self.samps/self.sums, self.perf/self.sums, lr))

class AverageEvalMetrics(tf.train.SessionRunHook):

  def __init__(self,loss,metrics,batch_size):
    self.batch_size = batch_size * mc.get_nranks()
    self.local_batch_size = batch_size
    self.both = [loss,metrics['accuracy'][0]]
    self.samps = 0
    self.perf = 0
    self.step = 0
    self.loss = np.asarray([0.],dtype='float32') 
    self.acc = np.asarray([0.],dtype='float32') 
    self.start_time = None

  def begin(self):
    self.step = 0
    self.start_time = time.time()

  def before_run(self, run_context):
    self.step += 1
    return tf.train.SessionRunArgs(self.both)  # Asks for loss value.

  def after_run(self, run_context, run_values):
  
    current_time = time.time()
    duration = current_time - self.start_time
    self.start_time = current_time
    
    loss_value = np.asarray([run_values.results[0]],dtype='float32')
    acc_value = np.asarray([run_values.results[1]],dtype='float32')
    examples_per_sec = self.batch_size / duration
    sec_per_batch = duration
          
    self.samps = self.samps + examples_per_sec
    self.perf  = self.perf + sec_per_batch
    self.loss  = self.loss + loss_value
    self.acc   = self.acc + acc_value

    if (mc.get_rank() == 0):
      print("Eval step {:9d}".format(self.step))

  def end(self,session):

    self.loss = self.loss / self.step
    self.acc  = self.acc / self.step
    
    mc.average(self.loss)
    mc.average(self.acc)
    
    format_str = ('EVAL Session ENDED at %s: step %d, loss = %.3f, accuracy = %.3f (%.1f examples/sec; %.3f '
                  'sec/batch)')


    if (mc.get_rank() == 0):
      print (format_str % (datetime.now(), self.step, self.loss, self.acc, self.samps/self.step, self.perf/self.step))
      

# END CRAY ADDED


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_parallel_calls=1,
                           examples_per_epoch=0, multi_gpu=False, numworkers=1):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    num_parallel_calls: The number of records that are processed in parallel.
      This can be optimized per data set but for generally homogeneous data
      sets, should be approximately the number of available CPU cores.
    examples_per_epoch: The number of examples in the current set that
      are processed each epoch. Note that this is only used for multi-GPU mode,
      and only to handle what will eventually be handled inside of Estimator.
    multi_gpu: Whether this is run multi-GPU. Note that this is only required
      currently to handle the batch leftovers (see below), and can be removed
      when that is handled directly by Estimator.

  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size)
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)

  # Currently, if we are using multiple GPUs, we can't pass in uneven batches.
  # (For example, if we have 4 GPUs, the number of examples in each batch
  # must be divisible by 4.) We already ensured this for the batch_size, but
  # we have to additionally ensure that any "leftover" examples-- the remainder
  # examples (total examples % batch_size) that get called a batch for the very
  # last batch of an epoch-- do not raise an error when we try to split them
  # over the GPUs. This will likely be handled by Estimator during replication
  # in the future, but for now, we just drop the leftovers here.
  if multi_gpu:
    total_examples = num_epochs * examples_per_epoch
    dataset = dataset.take(batch_size * (total_examples // batch_size))

  # Parse the raw records into images and labels
  dataset = dataset.map(lambda value: parse_record_fn(value, is_training),
                        num_parallel_calls=num_parallel_calls)

  dataset = dataset.batch(batch_size)

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path.
  #CRAY ADDED
  #dataset = dataset.prefetch(1)
  dataset = dataset.prefetch(batch_size)
  #CRAY DONE

  return dataset


def get_synth_input_fn(height, width, num_channels, num_classes):
  """Returns an input function that returns a dataset with zeroes.

  This is useful in debugging input pipeline performance, as it removes all
  elements of file reading and image preprocessing.

  Args:
    height: Integer height that will be used to create a fake image tensor.
    width: Integer width that will be used to create a fake image tensor.
    num_channels: Integer depth that will be used to create a fake image tensor.
    num_classes: Number of classes that should be represented in the fake labels
      tensor

  Returns:
    An input_fn that can be used in place of a real one to return a dataset
    that can be used for iteration.
  """
  def input_fn(is_training, data_dir, batch_size, *args):  # pylint: disable=unused-argument
    images = tf.zeros((batch_size, height, width, num_channels), tf.float32)
    labels = tf.zeros((batch_size, num_classes), tf.int32)
    return tf.data.Dataset.from_tensors((images, labels)).repeat()

  return input_fn


################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################
def learning_rate_with_decay(
    batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    batch_denom: this value will be used to scale the base learning rate.
      `0.1 * batch size` is divided by this number, such that when
      batch_denom == batch_size, the initial learning rate will be 0.1.
    num_images: total number of images that will be used for training.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.

  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = 0.1 * batch_size / batch_denom
  batches_per_epoch = num_images / batch_size

  # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
  boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
  vals = [initial_learning_rate * decay for decay in decay_rates]

  def learning_rate_fn(global_step):
    global_step = tf.cast(global_step, tf.int32)
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn

def learning_rate_warmup_poly_decay(
    batch_size, num_images, learning_rate_0, learning_rate_base, decay_epochs, warmup_epochs, mlcomm=0):

  eff_batch_size = batch_size
  if (mlcomm == 1):
    eff_batch_size = eff_batch_size * mc.get_nranks()

  batches_per_epoch = num_images / eff_batch_size
  decay_steps       = batches_per_epoch * decay_epochs
  warmup_steps      = batches_per_epoch * warmup_epochs
  
  lr_0 = tf.cast(learning_rate_0, tf.float32)
  lr_base = tf.cast(learning_rate_base, tf.float32)
  d_steps = tf.cast(decay_steps, tf.float32)
  w_steps = tf.cast(warmup_steps, tf.float32)


  def learning_rate_fn(global_step):

    global_step = tf.cast(global_step, tf.float32)
    #cstep = global_step*num_images/eff_batch_size
    epoch = (d_steps + w_steps)/(global_step+1)

    total_epochs = decay_epochs + warmup_epochs

    tf.Print(epoch, [epoch], "Epoch: ")
    #current_step = tf.Print(cstep, [cstep], "Current train steps so far: ")
    if (mlcomm == 1):
        current_lr = learning_rate_0*math.pow(1.0-(decay_steps-warmup_steps)/decay_steps,2)
        if (mc.get_rank()==0):
            print("Using Cray learning_rate_warmup_poly_decay(): ")
            print(" -> effective batch size: ", eff_batch_size)
            print(" -> batches per epoch: ", batches_per_epoch)

            print(" -> initial learning rate: ", learning_rate_0)
            print(" -> learning rate base: ", learning_rate_base)
            print(" -> starting global learning rate at first epoch: ", current_lr)
            print(" -> decay after ", decay_epochs, " epochs")
            print("     -> decay steps: ", decay_steps)
            print(" -> warmup with ", warmup_epochs, " epochs")
            print("     -> warmup steps: ", warmup_steps)
            print(" -> number workers: ", mc.get_nranks())
            #print(" -> Finished Epoch: ", tf.get_session_tensor(epoch), "/", total_epochs)

    #global_step = tf.cast(global_step, tf.float32)   
    def lr_warmup(): return (lr_0 + global_step * (lr_base - lr_0) / w_steps)
    def lr_poly(): return (lr_base * math_ops.pow((1 - (global_step - w_steps) / d_steps), 2))

    return tf.cond(tf.less(global_step, warmup_steps), lambda: lr_warmup(), lambda: lr_poly()) 
    
  return learning_rate_fn


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, version, loss_scale,
                    batch_size, log_freq,
                    loss_filter_fn=None, multi_gpu=False,
                    dtype=resnet_model.DEFAULT_DTYPE, mlcomm=0):
  """Shared functionality for different resnet model_fns.

  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.

  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    version: Integer representing which version of the ResNet network to use.
      See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    multi_gpu: If True, wrap the optimizer in a TowerOptimizer suitable for
      data-parallel distribution across multiple GPUs.
    dtype: the TensorFlow dtype to use for calculations.

  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Generate a summary node for the images
  tf.summary.image('images', features, max_outputs=6)

  features = tf.cast(features, dtype)

  model = model_class(resnet_size, data_format, version=version, dtype=dtype)

  logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

  # This acts as a no-op if the logits are already in fp32 (provided logits are
  # not a SparseTensor). If dtype is is low precision, logits must be cast to
  # fp32 for numerical stability.
  logits = tf.cast(logits, tf.float32)

  predictions = {
      'classes': tf.argmax(logits, axis=1),
      'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    # Return the predictions and the specification for serving a SavedModel
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

  # Calculate loss, which includes softmax cross entropy and L2 regularization.
  cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
       if loss_filter_fn(v.name)])
  tf.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    if (mlcomm == 1):
      #GDMomentumLARS(self, learning_rate, momentum_coeff=0.9, weight_decay_coeff=0.0005, lars_coeff=0.001, name="GDMomentumLARS")
      optimizer = GDMomentumLARSOptimizer(
            momentum_coeff=0.9, weight_decay_coeff=0.0005, 
            lars_coeff=0.001, learning_rate=learning_rate)

    else:
      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate,
          momentum=momentum)

    # If we are running multi-GPU, we need to wrap the optimizer.
    if multi_gpu:
      optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

    if loss_scale != 1:
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]

      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
     
    else:
      if (mlcomm == 1):
          # CRAY ADDED
          # we need to split out the minimize call below so we can modify gradients
          grads_and_vars = optimizer.compute_gradients(loss)

          grads     = mc.gradients([gv[0] for gv in grads_and_vars], 0)
          gs_and_vs = [(g,v) for (_,v), g in zip(grads_and_vars, grads)]

          minimize_op = optimizer.apply_gradients(gs_and_vs,
                                                  global_step=global_step)

      else:
          minimize_op = optimizer.minimize(loss, global_step)
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)

  else:
    train_op = None

  accuracy = tf.metrics.accuracy(
      tf.argmax(labels, axis=1), predictions['classes'])
  metrics = {'accuracy': accuracy}

  # Create a tensor named train_accuracy for logging purposes
  tf.identity(accuracy[1], name='train_accuracy')
  tf.summary.scalar('train_accuracy', accuracy[1])

  train_hooks = None
  eval_hooks  = None
  if (mlcomm == 1):
      if mode == tf.estimator.ModeKeys.TRAIN: 
          train_hooks = [BcastTensors(), AverageTrainMetrics(loss,metrics,log_freq,batch_size,lr=tf.cast(learning_rate,tf.float32),epoch=global_step)]
          eval_hooks  = None
      else:
          train_hooks = None
          eval_hooks  = [AverageEvalMetrics(loss,metrics,batch_size)]

  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      loss=loss,
      train_op=train_op,
      training_hooks=train_hooks, evaluation_hooks=eval_hooks,
      eval_metric_ops=metrics)


def validate_batch_size_for_multi_gpu(batch_size):
  """For multi-gpu, batch-size must be a multiple of the number of GPUs.

  Note that this should eventually be handled by replicate_model_fn
  directly. Multi-GPU support is currently experimental, however,
  so doing the work here until that feature is in place.

  Args:
    batch_size: the number of examples processed in each training batch.

  Raises:
    ValueError: if no GPUs are found, or selected batch_size is invalid.
  """
  from tensorflow.python.client import device_lib  # pylint: disable=g-import-not-at-top

  local_device_protos = device_lib.list_local_devices()
  num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
  if not num_gpus:
    raise ValueError('Multi-GPU mode was specified, but no GPUs '
                     'were found. To use CPU, run without --multi_gpu.')

  remainder = batch_size % num_gpus
  if remainder:
    err = ('When running with multiple GPUs, batch size '
           'must be a multiple of the number of available GPUs. '
           'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
          ).format(num_gpus, batch_size, batch_size - remainder)
    raise ValueError(err)


def resnet_main(flags, model_function, input_function, num_train_samps, num_eval_samps, shape=None):
  """Shared main loop for ResNet Models.

  Args:
    flags: FLAGS object that contains the params for running. See
      ResnetArgParser for created flags.
    model_function: the function that instantiates the Model and builds the
      ops for train/eval. This will be passed directly into the estimator.
    input_function: the function that processes the dataset and returns a
      dataset that the estimator can train on. This will be wrapped with
      all the relevant flags for running and passed to estimator.
    shape: list of ints representing the shape of the images used for training.
      This is only used if flags.export_dir is passed.
  """

  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if flags.multi_gpu:
    validate_batch_size_for_multi_gpu(flags.batch_size)

    # There are two steps required if using multi-GPU: (1) wrap the model_fn,
    # and (2) wrap the optimizer. The first happens here, and (2) happens
    # in the model_fn itself when the optimizer is defined.
    model_function = tf.contrib.estimator.replicate_model_fn(
        loss_reduction=tf.losses.Reduction.MEAN)

  # Create session config based on values of inter_op_parallelism_threads and
  # intra_op_parallelism_threads. Note that we default to having
  # allow_soft_placement = True, which is required for multi-GPU and not
  # harmful for other modes.


  myrank     = 0
  numworkers = 1
  if (flags.enable_ml_comm == 1):

      # initialize the Cray PE ML Plugin
      # config the thread team (correcting the number of epochs for the effectice batch size))
      #totsize = sum([reduce(lambda x, y: x*y, v.get_shape().as_list()) for v in tf.trainable_variables()])
     
      totsize = 25551401 #Specific size for resnet50-v2
      mc.init(2, 1, totsize, "tensorflow")
      myrank = mc.get_rank()
      numworkers = mc.get_nranks()
      if (myrank == 0):
          print("ResNet with {:9d} parameters".format(totsize))

      max_steps_train = int(math.ceil(flags.train_epochs *
                                      (num_train_samps + num_eval_samps) / (mc.get_nranks() * flags.batch_size)))
                    #(0,0,num_steps_before_going_nonblock, max_steps_train, verbose=1, how_often_to_print=100)
      mc.config_team(0, 0, max_steps_train, max_steps_train, 1, 100)

      flags.model_dir = flags.model_dir if mc.get_rank() == 0 else None
      flags.benchmark_log_dir = flags.benchmark_log_dir if mc.get_rank() == 0 else None
      flags.export_dir = flags.export_dir if mc.get_rank() == 0 else None
      

  else:
    rank_id = myrank 


  session_config = tf.ConfigProto( log_device_placement=False,
      inter_op_parallelism_threads=flags.inter_op_parallelism_threads,
      intra_op_parallelism_threads=flags.intra_op_parallelism_threads,
      allow_soft_placement=True
      )

    # Set up a RunConfig to save checkpoint and set session config.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=500,
                                                session_config=session_config)
  
  classifier = tf.estimator.Estimator(
      model_fn=model_function, model_dir=flags.model_dir, config=run_config,
      params={
          'resnet_size': flags.resnet_size,
          'data_format': flags.data_format,
          'batch_size': flags.batch_size,
          'multi_gpu': flags.multi_gpu,
          'train_epochs': flags.train_epochs,
          'version': flags.version,
          'loss_scale': flags.loss_scale,
          'dtype': flags.dtype,
          'mlcomm': flags.enable_ml_comm,
          'log_freq': flags.global_perf_log_freq,
          'weight_decay': flags.weight_decay,
          'init_lr': flags.init_lr,
          'base_lr': flags.base_lr,
          'warmup_epochs': flags.warmup_epochs,
          'log_freq': flags.global_perf_log_freq,

      })

  benchmark_logger = logger.config_benchmark_logger(flags.benchmark_log_dir)
  benchmark_logger.log_run_info('resnet')

  for _ in range(flags.train_epochs // flags.epochs_between_evals):
    train_hooks = hooks_helper.get_train_hooks(
        flags.hooks,
        batch_size=flags.batch_size,
        benchmark_log_dir=flags.benchmark_log_dir
	)
    if (myrank == 0):
        print('Starting a training cycle.')
   
    def input_fn_train():
      return input_function(True, flags.data_dir, flags.batch_size,
                            flags.epochs_between_evals,
                            flags.num_parallel_calls, flags.multi_gpu, numworkers, myrank)

    tsteps = math.ceil(float(flags.epochs_between_evals*num_train_samps) / (numworkers*flags.batch_size))
    classifier.train(input_fn=input_fn_train, steps=tsteps,
                     max_steps=flags.max_train_steps)

    if (myrank == 0):
        print('Starting to evaluate.')
        
    # Evaluate the model and print results
    def input_fn_eval():
      return input_function(False, flags.data_dir, flags.batch_size,
                            3, flags.num_parallel_calls, flags.multi_gpu, numworkers, myrank)

    # flags.max_train_steps is generally associated with testing and profiling.
    # As a result it is frequently called with synthetic data, which will
    # iterate forever. Passing steps=flags.max_train_steps allows the eval
    # (which is generally unimportant in those circumstances) to terminate.
    # Note that eval will run for max_train_steps each loop, regardless of the
    # global_step count.
    esteps = math.ceil(float(num_eval_samps) / (numworkers*flags.batch_size))
    eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                       steps=esteps)

    benchmark_logger.log_evaluation_result(eval_results)

    if model_helpers.past_stop_threshold(
        flags.stop_threshold, eval_results['accuracy']):
      break

  if flags.export_dir is not None:
    warn_on_multi_gpu_export(flags.multi_gpu)

    # Exports a saved model for the given classifier.
    input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
        shape, batch_size=flags.batch_size)
    classifier.export_savedmodel(flags.export_dir, input_receiver_fn)

  if (flags.enable_ml_comm == 1):
  	mc.finalize()


def warn_on_multi_gpu_export(multi_gpu=False):
  """For the time being, multi-GPU mode does not play nicely with exporting."""
  if multi_gpu:
    tf.logging.warning(
        'You are exporting a SavedModel while in multi-GPU mode. Note that '
        'the resulting SavedModel will require the same GPUs be available.'
        'If you wish to serve the SavedModel from a different device, '
        'try exporting the SavedModel with multi-GPU mode turned off.')


class ResnetArgParser(argparse.ArgumentParser):
  """Arguments for configuring and running a Resnet Model."""

  def __init__(self, resnet_size_choices=None):
    super(ResnetArgParser, self).__init__(parents=[
        parsers.BaseParser(),
        parsers.PerformanceParser(),
        parsers.ImageModelParser(),
        parsers.ExportParser(),
        parsers.BenchmarkParser(),
    ])

    self.add_argument(
        '--version', '-v', type=int, choices=[1, 2],
        default=resnet_model.DEFAULT_VERSION,
        help='Version of ResNet. (1 or 2) See README.md for details.'
    )

    self.add_argument(
        '--resnet_size', '-rs', type=int, default=50,
        choices=resnet_size_choices,
        help='[default: %(default)s] The size of the ResNet model to use.',
        metavar='<RS>' if resnet_size_choices is None else None
    )
    
    self.add_argument(
	    '--enable_ml_comm', '-mc', type=int, choices=[0,1],
	    default=1, help='[default: %(default)s] Whether to use Cray ML-Comm Distributed Training Plugin'
    )
    
    self.add_argument(
	    '--global_perf_log_freq', '-pf', type=int, default=50,
	    help='[default: %(default)s] Number of steps after which to report global (all process averages) training loss and performance'
    )



    self.add_argument(
        '--warmup_epochs', '-we', type=int, default=0,
        help='[default: %(default)s] Number of warmup epochs when using LARS'
    )


    self.add_argument(
        '--base_lr', '-blr', type=float, default=1.0,
        help='[default: %(default)s] Learning rate to start after warmup epochs finish when using LARS'
    )


    self.add_argument(
            '--init_lr', '-ilr', type=float, default=0.1,
            help='[default: %(default)s] Learning rate to start warmup with when using LARS'
    )


    self.add_argument(
            '--weight_decay', '-wd', type=float, default=1e-4,
            help='[default: %(default)s] Weight decay to use during training'
    )



  def parse_args(self, args=None, namespace=None):
    args = super(ResnetArgParser, self).parse_args(
        args=args, namespace=namespace)

    # handle coupling between dtype and loss_scale
    parsers.parse_dtype_info(args)

    return args
