from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2D, QDense, Clip, QActivation
# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation}


from tensorflow.keras.layers import Layer
import tensorflow as tf

class Select8Layers(Layer):
    def __init__(self,**kwargs):
        super(Select8Layers, self).__init__(**kwargs)
        
    def compute_output_shape(self, input_shape): # B x eta x phi x layers
        shape = input_shape
        shape[-1] = 8
        return shape 
        
    def call(self, input):
        l0 =  input[...,0:1]
        l2 =  input[...,2:3]
        l4 =  input[...,4:5]
        l6 =  input[...,6:7]
        l8 =  input[...,8:9]
        l10 = input[...,10:11]
        l12 = input[...,12:13]
        l13 = input[...,13:14]
        
        return tf.concat([l0,l2,l4,l6,l8,l10,l12,l13],axis=-1)

global_layers_list['Select8Layers']=Select8Layers