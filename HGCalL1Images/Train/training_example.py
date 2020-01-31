

from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization #etc

def my_model(Inputs,otheroption):
    
    x = Inputs[0] #this is the self.x list from the TrainData data structure
    x = BatchNormalization(momentum=0.9)(x)
    x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    x = Conv2D(8,(4,4),activation='relu', padding='same')(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Conv2D(8,(4,4),strides=(2,2),activation='relu', padding='valid')(x)
    x = Conv2D(4,(4,4),strides=(2,2),activation='relu', padding='valid')(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    
    # 3 prediction classes
    x = Dense(3, activation='softmax')(x)
    
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=False,renewtokens=False)

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model,otheroption=1)
    
    train.compileModel(learningrate=0.003,
                   loss='categorical_crossentropy') 
                   
print(train.keras_model.summary())


model,history = train.trainModel(nepochs=10, 
                                 batchsize=500,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)
                                 
print('Since the training is done, use the predict.py script to predict the model output on you test sample, e.g.: predict.py <training output>/KERAS_model.h5 <training output>/trainsamples.djcdc <path to data>/test.txt <output dir>')
