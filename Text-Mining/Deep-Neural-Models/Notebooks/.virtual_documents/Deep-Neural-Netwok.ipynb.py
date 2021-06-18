## required lib for the model


from tensorflow.keras.models import Sequential ## keras working over tensor flow
from tensorflow.keras.layers import Dense ## layers
import numpy as np



## creating the mode


model = Sequential() ## default parameter
model.add(Dense(10, activation='sigmoid',input_shape=(3,))) # 10 -> number of neuron on the hidden layer and input shape is shape of the input 3 neuron
model.add(Dense(1,activation='sigmoid')) ## output layer and number of neuron is '1'


model.summary() ## remember always there baise neuron on the input layer and hidden layer


## dummay data
input = np.array([
        [3,3,1],
        [4,3,1],
        [3,1,3],
        [2,3,1] 
])

# input shape always be set in the model is number of dim of instance
# in the above input data there are 4 instance and each instance have 3 dimension
# But this is not always be case of the input structure sometime you have [[2,3],[2,3],[2,3]]

output = np.array([0,0,1,1])


## training the Model
model  =Sequential()
model.add(Dense(10, ## hidden layer neuron
                activation='sigmoid', ## function input
               input_shape = (3,))) ## number of dim of instance
                                    ## Not the number of the instance
model.add(Dense(1,
               activation='sigmoid',)) 


## model compile
model.compile(loss="binary_crossentropy", # try to reduce the cost function to make reach near the 1 accuracy
             optimizer = 'sgd', # Sochastic gradient Decent
              metrics = ['accuracy'] # you can use it more
             )


## fiting the Model

model.fit(input[:-1],
          output[:-1],
          epochs=2, ## number of times to train the data
          batch_size=1, ## when data is large enough then need this
          validation_data=(input[-1:],output[-1:])
         )
## in the output accuracy should be one i dont why not give me one accuracy walla seacrch kary ghy!


## Required ib


from tensorflow.keras.models import Sequential ## keras working over tensor flow
from tensorflow.keras.layers import Dense ## layers
import numpy as np



## Dummy data
input  = np.random.rand(10000,3)
output = np.random.randint(2, size=10000) # always less then 2


input.shape


output.shape


input


output


## training the Model
model  =Sequential()
model.add(Dense(10, ## hidden layer neuron
                activation='sigmoid', ## function input
               input_shape = (3,))) ## number of dim of instance
                                    ## Not the number of the instance
model.add(Dense(1,
               activation='sigmoid',)) 


model.summary()


## model compile
model.compile(loss="binary_crossentropy", # try to reduce the cost function to make reach near the 1 accuracy
             optimizer = 'sgd', # Sochastic gradient Decent
              metrics = ['accuracy'] # you can use it more
             )


## fiting the Model

model.fit(input[:-3000],
          output[:-3000],
          epochs=2, ## number of times to train the data
          batch_size=64, ## when data is large enough then need this
          validation_data=(input[-3000:],output[-3000:])
         )
## in the output accuracy should be one i dont why not give me one accuracy walla seacrch kary ghy!


## EVALUATION



model.evaluate(input[-2000:],output[-2000:])


model.predict(input[-5:])



pred_y = model.predict_classes(input[-10:-5])


## check the in the data set
# whether is correct or not
y = output[-5:] #actuall value

pred_y ,y


from sklearn.metrics import precision_score,recall_score,f1_score 


precision_score(pred_y,y,average='micro')


recall_score(pred_y,y,average='micro')


f1_score(pred_y,y,average='micro')


# requried lib
from tensorflow.keras.models import Sequential ## keras working over tensor flow
from tensorflow.keras.layers import Dense ## layers
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("Datasets/DatasetUpdate.csv")
df["Label"] = [1 if x == True  else 0 for x in df["Label"]] # binary representation


df = df.drop(columns=['Unnamed: 0'])


df



x = df["Text"]
y = df.iloc[:,1]


x


y





df["Label"]


tfidf = TfidfVectorizer(max_features=200)

matrix_X = tfidf.fit_transform(x)


matrix_X


matrix_X.toarray()


from sklearn.model_selection import train_test_split



train_X , test_X ,train_y,test_y = train_test_split(matrix_X ,y, shuffle=True ,train_size=0.8 )


model  =Sequential()
model.add(Dense(10, ## hidden layer neuron
                activation='sigmoid', ## function input
               input_shape = (200,))) ## number of dim of instance
                                    ## Not the number of the instance
model.add(Dense(1,
               activation='sigmoid',)) 


model.compile(loss="binary_crossentropy", # try to reduce the cost function to make reach near the 1 accuracy
             optimizer = 'sgd', # Sochastic gradient Decent
              metrics = ['accuracy'] # you can use it more
             )


model.summary


train_X.shape



# ## you will see the error here because he will take array
train_X = train_X.toarray()
test_X = test_X.toarray()
model.fit(train_X,
          train_y,
          epochs=3, ## number of times to train the data
          batch_size=20, ## when data is large enough then need this
          validation_data=(test_X,test_y)
         )
## in the output accuracy should be one i dont why not give me one accuracy walla seacrch kary ghy!


from sklearn.metrics import accuracy_score,f1_score,recall_score


model.evaluate(test_X,test_y)
# pred_y = model.predict_classes()
pred_y = np.argmax(model.predict(test_X[:10]), axis=-1)

model.predict([test_X[:10]])

y =  test_y[:10]


print("Preceson Score : ",precision_score(pred_y , y , average='macro'))
print("Recall Score : ",recall_score(pred_y,y,average="macro",zero_division=True))
print("F1 Score : ",f1_score(pred_y, y ,average='macro'))








# requried lib
from tensorflow.keras.models import Sequential ## keras working over tensor flow
from tensorflow.keras.layers import Dense ,Dropout ## layers
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("Datasets/DatasetUpdate.csv")
df["Label"] = [1 if x == True  else 0 for x in df["Label"]] # binary representation

df = df.drop(columns=['Unnamed: 0'])
df


x = df["Text"]
y = df.iloc[:,1]


tfidf = TfidfVectorizer(max_features=200)

matrix_X = tfidf.fit_transform(x)


from sklearn.model_selection import train_test_split


train_X , test_X ,train_y,test_y = train_test_split(matrix_X ,y, shuffle=True ,train_size=0.8 )


model1 = Sequential()

model1.add(Dense(100,activation='relu',input_shape= (200,)))

model1.add(Dropout(0.15)) ## block some information
model1.add(Dense(80,activation='relu'))

model1.add(Dense(50,activation='relu'))
model1.add(Dropout(0.10)) ## will block some informatin
model1.add(Dense(30,activation='relu'))
model1.add(Dense(1,activation='sigmoid'))




model1.summary()


# model1.compile(loss="binary_crossentropy", # try to reduce the cost function to make reach near the 1 accuracy
#              optimizer = 'sgd', # Sochastic gradient Decent
#               metrics = ['accuracy'] # you can use it more
#              )

# chnage with optimmizer function 'rmsprop'
model1.compile(loss="binary_crossentropy", # try to reduce the cost function to make reach near the 1 accuracy
             optimizer = 'rmsprop', # Sochastic gradient Decent
              metrics = ['accuracy'] # you can use it more
             )



# ## you will see the error here because he will take array
train_X = train_X.toarray()
test_X = test_X.toarray()
model1.fit(train_X,
          train_y,
          epochs=10, ## number of times to train the data
          batch_size=64, ## when data is large enough then need this
          validation_data=(test_X,test_y)
         )
## in the output accuracy should be one i dont why not give me one accuracy walla seacrch kary ghy!


model1.evaluate(test_X,test_y)
# pred_y = model.predict_classes()
pred_y = np.argmax(model.predict(test_X[:10]), axis=-1)

model1.predict([test_X[:10]])

y =  test_y[:10]


print("Preceson Score : ",precision_score(pred_y , y , average='macro'))
print("Recall Score : ",recall_score(pred_y,y,average="macro",zero_division=True))
print("F1 Score : ",f1_score(pred_y, y ,average='macro'))



