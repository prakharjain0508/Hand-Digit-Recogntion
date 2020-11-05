import numpy as np
import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
mode=0                                                      #mode=0: Fully-Connected Neural Network
                                                            #mode=1: Convolutional Neural Network

test_wrong_pred = [[] * 10 for i in range (0,10)]
test_class_count = [0] * 10
train_wrong_pred = [[] * 10 for i in range (0,10)]
train_class_count = [0] * 10
test_num_wrong_class_count = [[0] * 10 for i in range (0,10)]
train_num_wrong_class_count = [[0] * 10 for i in range (0,10)]
train_sum=0
test_sum=0

pred_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header= None)
pred_data.rename(columns = {64 : 'label'}, inplace = True)
pred = pred_data.drop(["label"], axis=1).values
pred1 = pred.reshape(pred.shape[0], 8, 8, 1)

dtr = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header= None)
dtr.rename(columns = {64 : 'label'}, inplace = True) 
predictors = dtr.drop(["label"], axis=1).values
predictors1 = predictors.reshape(predictors.shape[0], 8, 8, 1)

train_predictors = predictors[:3058]
train_predictors1 = predictors1[:3058]

target = to_categorical(dtr.label)
train_target = target[0:3058]
test_target = to_categorical(pred_data.label)

#print(predictors)
#print(target)
if(mode == 0):
    print("############### Fully Connected Neural Netwrok ###############\n")
    model = Sequential()
    
    model.add(Dense(800,activation='relu',input_shape=(64,)))
    
    model.add(Dense(800,activation='relu'))
    
    model.add(Dense(800,activation='relu'))
    
    model.add(Dense(10,activation='softmax'))
    
    early_stopping_monitor = EarlyStopping(patience=5)
    
    keras.optimizers.SGD(lr=0.009, momentum=0.95, nesterov=False)
    
    model.compile(optimizer='sgd',loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(predictors,target, validation_split=0.2, epochs = 100, callbacks = [early_stopping_monitor], verbose = 1)
    
    a = model.evaluate(pred, test_target,verbose=0)
    b = model.evaluate(train_predictors, train_target, verbose=0)
    
    print("\nGenerating predictions...\n")
    
    test_preds = model.predict_classes(pred, verbose=0)
    print(test_preds.shape)
    train_preds = model.predict_classes(train_predictors, verbose=0)
    
    def write_preds(test_preds, fname):
        pd.DataFrame({"ImageId": list(range(0,len(test_preds))), "Label": test_preds}).to_csv(fname, index=False, header=True)
    
    write_preds(test_preds, "Predicted_class_test_data.csv")
    write_preds(train_preds, 'Predicted_class_train_data.csv')
    
    print("Check the folder for .xls files for predicted test and train data classification\n")
    ######################### FOR TRAIN DATA ################################## 
    print("########### TRAIN RESULTS ###########")
    print('\nOverall Loss: ' + str(b[0]))
    print('Overall Accuracy: ' + str(b[1]))
    for i in range(0,3058):
        train_class_count[dtr.label[i]] = train_class_count[dtr.label[i]] + 1;
    for i in range(0,3058):
        if(dtr.label[i] != train_preds[i]):
            train_wrong_pred[dtr.label[i]].append(train_preds[i]);
    #print(train_wrong_pred)
    #print(train_class_count)
    for i in range (0,10):
        train_sum = train_sum + len(train_wrong_pred[i])
        
    print("\nClass Accuracy: ")
    for i in range(0,10):
        print("Class " + str(i) + " : ",end="")
        print((train_class_count[i]-len(train_wrong_pred[i]))/train_class_count[i])
    print('\n')
    
    for i in range(0,10):
        for j in range(0,10):
            train_num_wrong_class_count[i][j] = train_wrong_pred[i].count(j)
    
    print("Confusion Matrix:")
    for i in range(0,10):
        for j in range(0,10):
            if(i==j):
                print(train_class_count[i]-len(train_wrong_pred[i]), end="   ")
                continue
            if(j==9):
                print(train_num_wrong_class_count[i][j])
            else:
                print(train_num_wrong_class_count[i][j], end="   ")
    print('\n')
    ########################## FOR TEST DATA ################################## 
    print('########## TEST RESULTS ############')
    print('\nOverall Loss: ' + str(a[0]))
    print('Overall Accuracy: ' + str(a[1]))
    for i in range(0,1797):
        test_class_count[pred_data.label[i]] = test_class_count[pred_data.label[i]] + 1;
    for i in range(0,1797):
        if(pred_data.label[i] != test_preds[i]):
            test_wrong_pred[pred_data.label[i]].append(test_preds[i]);
    #print(test_wrong_pred)
    #print(test_class_count)
    for i in range (0,10):
        test_sum = test_sum + len(test_wrong_pred[i])
        
    print("\nClass Accuracy: ")
    for i in range(0,10):
        print("Class " + str(i) + " : ",end="")
        print((test_class_count[i]-len(test_wrong_pred[i]))/test_class_count[i])
    print('\n')
    
    for i in range(0,10):
        for j in range(0,10):
            test_num_wrong_class_count[i][j] = test_wrong_pred[i].count(j)
    
    print("Confusion Matrix:")
    for i in range(0,10):
        for j in range(0,10):
            if(i==j):
                print(test_class_count[i]-len(test_wrong_pred[i]), end="   ")
                continue
            if(j==9):
                print(test_num_wrong_class_count[i][j])
            else:
                print(test_num_wrong_class_count[i][j], end="   ")
    print('\n')

###################################### FOR CNNs #####################################
if(mode == 1): 
    print("############### Convolutional Neural Netwrok ###############\n")
    keras.optimizers.SGD(lr=0.05, momentum=0.75, nesterov=False)
    early_stopping_monitor = EarlyStopping(patience=5)
    def larger_model():
    	# create model
    	model = Sequential()
    	model.add(Conv2D(64, kernel_size = (2, 2), strides=(1,1), input_shape=(8,8,1), activation='relu'))
    	model.add(MaxPooling2D(pool_size=(2, 2), strides = (2,2)))
    	model.add(Dropout(0.3))
    	model.add(Flatten())
    	model.add(Dense(500, activation='relu'))
    	model.add(Dense(300, activation='relu'))
    	model.add(Dense(10, activation='softmax'))
    	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    	return model

    model = larger_model()
    model.fit(predictors1, target, validation_split=0.2, epochs=100, callbacks = [early_stopping_monitor], verbose=1)
    
    a = model.evaluate(pred1, test_target,verbose=0)
    b = model.evaluate(train_predictors1, train_target, verbose=0)
    
    print("Generating predictions...\n")
    
    test_preds = model.predict_classes(pred1, verbose=0)
    train_preds = model.predict_classes(train_predictors1, verbose=0)
    
    def write_preds(test_preds, fname):
        pd.DataFrame({"ImageId": list(range(0,len(test_preds))), "Label": test_preds}).to_csv(fname, index=False, header=True)
    
    write_preds(test_preds, "Predicted_class_test_data_CNN.csv")
    write_preds(train_preds, 'Predicted_class_train_data_CNN.csv')
    
    print("Check the folder for .xls files for predicted test and train data classification\n")
    ######################### FOR TRAIN DATA ################################## 
    print("########### TRAIN RESULTS ###########")
    print('\nOverall Loss: ' + str(b[0]))
    print('Overall Accuracy: ' + str(b[1]))
    for i in range(0,3058):
        train_class_count[dtr.label[i]] = train_class_count[dtr.label[i]] + 1;
    for i in range(0,3058):
        if(dtr.label[i] != train_preds[i]):
            train_wrong_pred[dtr.label[i]].append(train_preds[i]);
    #print(train_wrong_pred)
    #print(train_class_count)
    for i in range (0,10):
        train_sum = train_sum + len(train_wrong_pred[i])

    print("\nClass Accuracy: ")
    for i in range(0,10):
        print("Class " + str(i) + " : ",end="")
        print((train_class_count[i]-len(train_wrong_pred[i]))/train_class_count[i])
    print('\n')
    
    for i in range(0,10):
        for j in range(0,10):
            train_num_wrong_class_count[i][j] = train_wrong_pred[i].count(j)
    
    print("Confusion Matrix:")
    for i in range(0,10):
        for j in range(0,10):
            if(i==j):
                print(train_class_count[i]-len(train_wrong_pred[i]), end="   ")
                continue
            if(j==9):
                print(train_num_wrong_class_count[i][j])
            else:
                print(train_num_wrong_class_count[i][j], end="   ")
    print('\n')
    ########################## FOR TEST DATA ################################## 
    print('########## TEST RESULTS ############')
    print('\nOverall Loss: ' + str(a[0]))
    print('Overall Accuracy: ' + str(a[1]))
    for i in range(0,1797):
        test_class_count[pred_data.label[i]] = test_class_count[pred_data.label[i]] + 1;
    for i in range(0,1797):
        if(pred_data.label[i] != test_preds[i]):
            test_wrong_pred[pred_data.label[i]].append(test_preds[i]);
    #print(test_wrong_pred)
    #print(test_class_count)
    for i in range (0,10):
        test_sum = test_sum + len(test_wrong_pred[i])

    print("\nClass Accuracy: ")
    for i in range(0,10):
        print("Class " + str(i) + " : ",end="")
        print((test_class_count[i]-len(test_wrong_pred[i]))/test_class_count[i])
    print('\n')
     
    for i in range(0,10):
        for j in range(0,10):
            test_num_wrong_class_count[i][j] = test_wrong_pred[i].count(j)
    
    print("Confusion Matrix:")
    for i in range(0,10):
        for j in range(0,10):
            if(i==j):
                print(test_class_count[i]-len(test_wrong_pred[i]), end="   ")
                continue
            if(j==9):
                print(test_num_wrong_class_count[i][j])
            else:
                print(test_num_wrong_class_count[i][j], end="   ")
    print('\n')
