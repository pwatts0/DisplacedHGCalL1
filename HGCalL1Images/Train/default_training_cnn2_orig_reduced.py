'''

This is a model with 8 HGCal layers

'''

from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout #etc

from Losses import binary_cross_entropy_with_extras
from Layers import Select8Layers

def my_model(Inputs,dropoutrate=0.1,momentum=0.95):
    
    x = Inputs[0] #B x 30 x 128 x 14
    x = Select8Layers()(x)  #B x 30 x 128 x 8
    x = BatchNormalization(momentum=momentum)(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x_col = Dense(3, activation='relu', name="pseudocolors")(x)
    x = Dropout(dropoutrate/5.)(x_col)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x) 
    x = MaxPooling2D(pool_size=(2,2))(x) #15 x 64
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x) #7 x 32
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Dropout(dropoutrate)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(1,2))(x) #7 x 16
    x = BatchNormalization(momentum=momentum)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(1,2))(x) #7 x 8
    x = BatchNormalization(momentum=momentum)(x) # 7 x 8 x 16
    
    x = Dropout(dropoutrate)(x)
    
    x = Flatten()(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
  
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)



train=training_base(testrun=False,resumeSilently=False,renewtokens=False)

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model)
    
    
    train.compileModel(learningrate=0.0001,
                   loss='binary_crossentropy',#binary_cross_entropy_with_extras,
                   )#metrics=['accuracy']) 
                   
print(train.keras_model.summary())


model,history = train.trainModel(nepochs=1, 
                                 batchsize=50,#50,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)
train.change_learning_rate(0.0003)
model,history = train.trainModel(nepochs=30, 
                                 batchsize=500,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)
                                 
