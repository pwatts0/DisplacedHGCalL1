

from DeepJetCore.training.training_base import training_base
import keras
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout #etc

from Losses import binary_cross_entropy_with_extras

def my_model(Inputs,dropoutrate=0.3):
    
    x = Inputs[0] #B x 30 x 128 x 3
    x = BatchNormalization(momentum=0.6)(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    #x_col = Dense(3, activation='relu')(x)
    #x=c_col
    
    x = Conv2D(16, (7,7), padding='same', activation='relu')(x)
    x = Conv2D(16, (3,3), padding='same', activation='relu')(x) #,strides=(2,2)
    x = MaxPooling2D(pool_size=(2,2))(x) #15 x 64
    
    
    x = BatchNormalization(momentum=0.6)(x)
    x = Dropout(dropoutrate)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x) #7 x 32
    
    
    x = Dropout(dropoutrate)(x)
    
    x = Conv2D(16, (3,3), padding='valid', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x) #3 x 16
    
    
    x = BatchNormalization(momentum=0.6)(x)
    x = Dropout(dropoutrate)(x)
    
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
    
    
    train.compileModel(learningrate=0.001,
                   loss='binary_crossentropy',#binary_cross_entropy_with_extras,
                   metrics=['accuracy']) 
                   
print(train.keras_model.summary())


model,history = train.trainModel(nepochs=2, 
                                 batchsize=50,
                                 checkperiod=1, # saves a checkpoint model every N epochs
                                 verbose=1)

model,history = train.trainModel(nepochs=10, 
                                 batchsize=300,
                                 checkperiod=2, # saves a checkpoint model every N epochs
                                 verbose=1)
                                 
