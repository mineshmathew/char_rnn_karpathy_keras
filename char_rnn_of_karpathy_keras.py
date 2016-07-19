
# coding: utf-8

# In[1]:


####

#minesh.mathew@gmail.com
#modified version of text generation example in keras; trained in a many-to-many fashion using a time distributed dense layer

####
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM,TimeDistributedDense,SimpleRNN
from keras.utils.data_utils import get_file
import numpy as np
from time import sleep
import random
import sys

##uncomment below if you want to use nietzches writings as the corpus

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
#text = open(path).read().lower()
text = open('./textdatasets/tinyshakesepare.txt').read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# split the corpus into sequences of length=maxlen
#input is a sequence of 40 chars and target is also a sequence of 40 chars shifted by one position
#for eg: if you maxlen=3 and the text corpus is abcdefghi, your input ---> target pairs will be
# [a,b,c] --> [b,c,d], [b,c,d]--->[c,d,e]....and so on
maxlen = 40
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen+1, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i+1:i +1+ maxlen])
    #if i<10 :
       # print (text[i: i + maxlen])
        #print(text[i+1:i +1+ maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences),maxlen, len(chars)), dtype=np.bool) # y is also a sequence , or  a seq of 1 hot vectors
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1

for i, sentence in enumerate(next_chars):
    for t, char in enumerate(sentence):
        y[i, t, char_indices[char]] = 1
    

print ('vetorization completed')




# In[2]:

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
#model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))  # original one
model.add(LSTM(512, input_dim=len(chars),return_sequences=True)) #minesh witout specifying the input_length
model.add(LSTM(512, return_sequences=True)) #- original
model.add(Dropout(0.2))
model.add(TimeDistributedDense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print ('model is made')

# train the model, output generated text after each iteration


# In[9]:

print (model.summary())


# In[ ]:

for iteration in range(1, 6):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history=model.fit(X, y, batch_size=128, nb_epoch=1,verbose=0)
    sleep(0.1) # https://github.com/fchollet/keras/issues/2110
    
    # saving models at the following iterations -- uncomment it if you want tos save weights and load it later
    #if iteration==1 or iteration==3 or iteration==5 or iteration==10 or iteration==20 or iteration==30 or iteration==50 or iteration==60 :
    #    model.save_weights('Karpathy_LSTM_weights_'+str(iteration)+'.h5', overwrite=True)
    #start_index = random.randint(0, len(text) - maxlen - 1)

    #sys.stdout.flush()
    print ('loss is')
    print (history.history['loss'][0])
    print (history)
    print()    

    


# #### testing
# now you use the trained model to generat text.
# the  output shown in this notebook is for a model which is trained only for 1 iteration

# In[6]:

seed_string="brutus:"
print ("seed string -->", seed_string)
print ('The generated text is')
sys.stdout.write(seed_string),
#x=np.zeros((1, len(seed_string), len(chars)))
for i in range(320):
    x=np.zeros((1, len(seed_string), len(chars)))
    for t, char in enumerate(seed_string):
        x[0, t, char_indices[char]] = 1.
    preds = model.predict(x, verbose=0)[0]
    #print (np.argmax(preds[7]))
    next_index=np.argmax(preds[len(seed_string)-1])
    
    
    #next_index=np.argmax(preds[len(seed_string)-11])
    #print (preds.shape)
    #print (preds)
    #next_index = sample(preds, 1) #diversity is 1
    next_char = indices_char[next_index]
    seed_string = seed_string + next_char
    
    #print (seed_string)
    #print ('##############')
    #if i==40:
    #    print ('####')
    sys.stdout.write(next_char)

sys.stdout.flush()    



# In[ ]:



