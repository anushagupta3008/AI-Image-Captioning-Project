#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils import Sequence
from keras.layers import Input, Dense, Dropout, Embedding, LSTM
from keras.layers.merge import add


# In[2]:


model = load_model('model_29.h5')


# In[3]:


model_temp = ResNet50(weights='imagenet',input_shape=(224,224,3))


# In[4]:


model_resnet = Model(model_temp.input,model_temp.layers[-2].output)


# In[7]:


def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    #Normalization
    img = preprocess_input(img)
    return img


# In[15]:


#Pass through trained resnet to get mX2048 image vector
def encode_image(img):
    img = preprocess_img(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape((-1,))
    #print(feature_vector.shape)
    return feature_vector.reshape(1,-1)


# In[16]:



# In[17]:



# In[20]:


max_len = 35
vocab_size = 1848


# In[24]:


with open('word_2_idx.pkl', 'rb') as w2i:
    word_to_idx = pickle.load(w2i)
with open('idx_2_word.pkl', 'rb') as i2w:
    idx_to_word = pickle.load(i2w)


# In[25]:


word_to_idx['the']


# In[21]:


def predict_caption(photo):
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')

        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #Word with max probability
        word = idx_to_word[ypred]
        in_text += ' ' + word

        if word == 'endseq':
            break

    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption

def caption_this_image(img):
    enc = encode_image(img)
    caption = predict_caption(enc)

    return caption


# In[27]:



# In[ ]:




