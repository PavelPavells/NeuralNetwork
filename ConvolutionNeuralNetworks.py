#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import numpy as np


# In[5]:


fashion_mnist = tf.keras.datasets.fashion_mnist


# In[6]:


(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()


# In[7]:


class_names = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]


# In[9]:


print(train_labels[9])


# In[10]:


class_names[5]


# In[12]:


plt.imshow(train_images[9])


# In[16]:


#normalize
train_images = train_images / 255.0
test_images = test_images / 255.0


# In[17]:


train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)


# In[28]:


model = tf.keras.Sequential()


# In[29]:


model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape(28,28,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2,padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2,padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# In[30]:


model.summary()


# In[31]:


model.compile(loss='soarse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[32]:


model.fit(train_images,train_labels,batch_size=64,epochs=2)


# In[33]:


model.evaluate(test_images, test_labels)


# In[34]:


predictions = model.predict(test_images)


# In[35]:


predictions[0]


# In[36]:


np.argmax(predictions[0])


# In[39]:


class_names[9]


# In[40]:


(train_images,train_labels),(orig_test_images,test_labels) = fashion_mnist.load_data()


# In[41]:


plt.imshow(orig_test_images[0])


# In[44]:


def what_is_this_image(test_image):
    test_image = test_image / 255.0
    predictions = model.predict(test_image.reshape(-1,28,28,1))
    index = np.argmax(prediction)
    print("Ok we think it is a " + class_names[index])
    imshow(test_image)


# In[45]:


what_is_this_image(orig_test_images[45])

