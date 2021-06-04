'''

This is the exact model used for the arxiv paper training!

'''

from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout #etc

from Losses import binary_cross_entropy_with_extras

# Tools for model compression and quantization
from qkeras.utils import model_quantize
from qkeras.estimate import print_qstats

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.sparsity import keras as sparsity
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule, pruning_wrapper

from util import doOps, print_model_sparsity
from qdictionaries import qDicts

def setWeights(model,fullModel='full_model/model.h5'):
  modelB = keras.models.load_model(fullModel,custom_objects={'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude})
  for layerA,layerB in zip(model.layers,modelB.layers):
    layerA.set_weights(layerB.get_weights())
    
def model_pruned(Inputs,dropoutrate=0.1,momentum=0.95,pruning_params = {'pruning_schedule': sparsity.ConstantSparsity(0.75, begin_step=2000, frequency=100)}):
    
    x = Inputs[0] #B x 30 x 128 x 3
    x = BatchNormalization(momentum=momentum)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x_col = Dense(4, activation='relu', name="pseudocolors")(x)
    x = Dropout(dropoutrate/5.)(x_col)
    
    #x = Conv2D(16, (4,4), padding='same', activation='relu')(x)
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x) #x = prune.prune_low_magnitude( (Conv2D(16, (3,3), padding='valid', activation='relu')),**pruning_params) (x)
    x = MaxPooling2D(pool_size=(2,2))(x) #15 x 64
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x) #7 x 32
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Dropout(dropoutrate)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(1,2))(x) #7 x 32
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(1,2))(x) #7 x 32
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Dropout(dropoutrate)(x)
    
    x = Flatten()(x)
    
    x = prune.prune_low_magnitude( (Dense(32, activation='relu')),**pruning_params) (x)
    x = Dense(1, activation='sigmoid')(x)
  
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions, name='model_pruned')
    
def my_model(Inputs,dropoutrate=0.1,momentum=0.95):
    
    x = Inputs[0] #B x 30 x 128 x 3
    x = BatchNormalization(momentum=momentum)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    x_col = Dense(4, activation='relu', name="pseudocolors")(x)
    x = Dropout(dropoutrate/5.)(x_col)
    
    #x = Conv2D(16, (4,4), padding='same', activation='relu')(x)
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x) #,strides=(2,2)
    x = MaxPooling2D(pool_size=(2,2))(x) #15 x 64
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x) #7 x 32
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Dropout(dropoutrate)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(1,2))(x) #7 x 32
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(1,2))(x) #7 x 32
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Dropout(dropoutrate)(x)
    
    x = Flatten()(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
  
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)

def my_model_complex(Inputs,dropoutrate=0.01):
    
    x = Inputs[0] #B x 30 x 128 x 3
    x = BatchNormalization(momentum=0.6)(x)
    x = Dense(48, activation='relu')(x)
    
    x = Conv2D(48, (7,7), padding='same', activation='relu', strides=(2,2))(x)
    
    x = BatchNormalization(momentum=0.6)(x)
    x = Dropout(dropoutrate)(x)
    
    x = Conv2D(48, (3,3), padding='valid', activation='relu', strides=(2,2))(x)
    
    
    x = Dropout(dropoutrate)(x)
    
    x = Conv2D(48, (3,3), padding='valid', activation='relu', strides=(2,2))(x)
    
    
    x = BatchNormalization(momentum=0.6)(x)
    x = Dropout(dropoutrate)(x)
    
    x = Flatten()(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
  
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)



Prune     = True
Quantize  = False
fullModel = 'full_model/model.h5'

additionalCallbacks = None

train=training_base(testrun=False,resumeSilently=False,renewtokens=False)

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot
    
    train.setModel(my_model)
    totalGFlops = doOps(train.keras_model)
    print('\n Total floating point ops per second = {} GFLOPS \n'.format(totalGFlops))
    
    if Prune:
      train.setModel(model_pruned)
      if fullModel:
        setWeights(train.keras_model,fullModel)
      print_model_sparsity(train.keras_model)
      additionalCallbacks = pruning_callbacks.UpdatePruningStep()
      
    elif Quantize:
      try:
        train.keras_model = keras.models.load_model(fullModel)
        transferWeights = True 
      except:  
        print("No pretrained model found! Building new model without pretrained weights")
        transferWeights = False 
      
      train.keras_model = model_quantize(train.keras_model, qDicts['4_bit'], 4, transfer_weights=transferWeights)  #currently dense2_binary', conv2d_binary', '4_bit'         
      print_qstats(train.keras_model)
    
    train.compileModel(learningrate=0.0001,
                   loss='binary_crossentropy')#,binary_cross_entropy_with_extras) 
                   
print(train.keras_model.summary())

model,history = train.trainModel(nepochs=30,
                                 batchsize=50,#50,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 additional_callbacks = additionalCallbacks )
train.change_learning_rate(0.0003)
model,history = train.trainModel(nepochs=30,
                                 batchsize=50,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1,
                                 additional_callbacks = additionalCallbacks)
                                 
