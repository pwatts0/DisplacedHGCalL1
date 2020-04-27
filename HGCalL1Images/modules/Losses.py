
# Define custom losses here and add them to the global_loss_list dict (important!)
global_loss_list = {}


def binary_cross_entropy_with_extras(y_true, y_pred):
    
    return keras.losses.binary_crossentropy(y_true, y_pred[:,0:1])

global_loss_list['binary_cross_entropy_with_extras']=binary_cross_entropy_with_extras