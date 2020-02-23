

from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout #etc

def my_model(Inputs,otheroption):
    
    x = Inputs[0] #B x 30 x 128 x 3
    x = BatchNormalization(momentum=0.6)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(3, activation='relu')(x)
    
    x = Conv2D(16, (7,7), padding='same', activation='relu')(x)
    x = Conv2D(16, (3,3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x) #15 x 64
    
    x = Dropout(0.5)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x) #7 x 32
    
    
    x = Dropout(0.5)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x) #3 x 16
    
    
    x = Dropout(0.5)(x)
    
    x = Flatten()(x)
    
    x = Dense(32, activation='relu')(x)
    
    # 3 prediction classes
    x = Dense(1, activation='sigmoid')(x)
    
    #x = Concatenate()([x,x_col]) #colours just for pretty plot
    
    predictions = [x]
    return Model(inputs=Inputs, outputs=predictions)


train=training_base(testrun=False,resumeSilently=False,renewtokens=False)

if not train.modelSet(): # allows to resume a stopped/killed training. Only sets the model if it cannot be loaded from previous snapshot

    train.setModel(my_model,otheroption=1)
    
    train.compileModel(learningrate=0.003,
                   loss='binary_crossentropy',
                   metrics=['accuracy']) 
                   
print(train.keras_model.summary())


model,history = train.trainModel(nepochs=100, 
                                 batchsize=1000,
                                 checkperiod=20, # saves a checkpoint model every N epochs
                                 verbose=1)
                                 
