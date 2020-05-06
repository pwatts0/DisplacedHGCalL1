from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras import QConv2D, QDense, Clip, QActivation
# Define custom layers here and add them to the global_layers_list dict (important!)
global_layers_list = {'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,'QDense': QDense, 'QConv2D': QConv2D, 'Clip': Clip, 'QActivation': QActivation}
