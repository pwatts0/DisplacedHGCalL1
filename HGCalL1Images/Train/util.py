# -*- coding: utf-8 -*-

import tensorflow.keras.backend as K
import numpy as np
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

# For FLOP compute, based on https://github.com/kentaroy47/keras-Opcounter

# bunch of cal per layer
def count_linear(layers):
    MAC = layers.output_shape[1] * layers.input_shape[1]
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[1]
    else:
        ADD = 0
    return MAC*2 + ADD

def count_conv2d(layers, log = False):
    if log:
        print(layers.get_config())
    # number of conv operations = input_h * input_w / stride = output^2
    numshifts = int(layers.output_shape[1] * layers.output_shape[2])
    
    # MAC/convfilter = kernelsize^2 * InputChannels * OutputChannels
    MACperConv = layers.get_config()["kernel_size"][0] * layers.get_config()["kernel_size"][1] * layers.input_shape[3] * layers.output_shape[3]
    
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[3]
    else:
        ADD = 0
        
    return MACperConv * numshifts * 2 + ADD

def profile(model, log = False):
    # make lists
    layer_name = []
    layer_flops = []
    # TODO: relus
    inshape = []
    weights = []
    # run through models
    for layer in model.layers:
        if "act" in layer.get_config()["name"]:
          print ("Skipping ativation functions for now!")
           
        elif "dense" in layer.get_config()["name"] or "fc" in layer.get_config()["name"]:
            
            layer_flops.append(count_linear(layer))
            layer_name.append(layer.get_config()["name"])
            inshape.append(layer.input_shape)
            weights.append(int(np.sum([K.count_params(p) for p in (layer.trainable_weights)])))
        elif "conv" in layer.get_config()["name"] and "pad" not in layer.get_config()["name"] and "bn" not in layer.get_config()["name"] and "relu" not in layer.get_config()["name"] and "concat" not in layer.get_config()["name"]:
            layer_flops.append(count_conv2d(layer,log))
            layer_name.append(layer.get_config()["name"])
            inshape.append(layer.input_shape)
            weights.append(int(np.sum([K.count_params(p) for p in (layer.trainable_weights)])))
        elif "res" in layer.get_config()["name"] and "branch" in layer.get_config()["name"]:
            layer_flops.append(count_conv2d(layer,log))
            layer_name.append(layer.get_config()["name"])
            inshape.append(layer.input_shape)
            weights.append(int(np.sum([K.count_params(p) for p in (layer.trainable_weights)])))
            
    return layer_name, layer_flops, inshape, weights
    
def doOps(model):
  print("Counting number of OPS in model")
  model.summary()
  layer_name, layer_flops, inshape, weights = profile(model)
  for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
      print("layer:", name, shape, " MegaFLOPS:", flop/1e6, " MegaWeights:", weight/1e6)
  totalGFlops = sum(layer_flops)/1e9
  print("Total FLOPS[GFLOPS]:",totalGFlops )
  return totalGFlops    

# MISC

def print_model_sparsity(pruned_model):
  """Prints sparsity for the pruned layers in the model.
  Model Sparsity Summary
  --
  prune_lstm_1: (kernel, 0.5), (recurrent_kernel, 0.6)
  prune_dense_1: (kernel, 0.5)
  Args:
    pruned_model: keras model to summarize.
  Returns:
    None
  """
  def _get_sparsity(weights):
    return 1.0 - np.count_nonzero(weights) / float(weights.size)

  print("Model Sparsity Summary ({})".format(pruned_model.name))
  print("--")
  for layer in pruned_model.layers:
    if isinstance(layer, pruning_wrapper.PruneLowMagnitude):
      prunable_weights = layer.layer.get_prunable_weights()
      if prunable_weights:
        print("{}: {}".format(
            layer.name, ", ".join([
                "({}, {})".format(weight.name,
                                  str(_get_sparsity(K.get_value(weight))))
                for weight in prunable_weights
            ])))
  print("\n")
      