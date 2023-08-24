import pandas as pd
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers , models
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import models



data = []
lables = []
i = 0
for item in glob.glob("fire_dataset/*/*"):
    i+=1
    img = cv2.imread(item)
    r_img = cv2.resize(img,(32,32)).flatten()
    data.append(r_img)
    lable = item.split("\\")[-2]
    lables.append(lable)
    
    if i%100==0:
        print("[Info]: {}/1000 processed".format(i))

le = LabelEncoder()

lables = le.fit_transform(lables) #convert lables to integer enciding
lables = to_categorical(lables) #convert integer encoding to hot encoding
data = np.array(data) / 255.0
X_train,X_test,y_train,y_test = train_test_split(data,lables,test_size=0.3)

net = models.Sequential([
    layers.Dense(300,activation="relu",input_dim=3072),
    layers.Dense(40,activation="relu"),
    layers.Dense(2,activation="softmax"),
])

print(net.summary()) 
net.compile(optimizer="sgd",loss="binary_crossentropy",metrics=["accuracy"])

H = net.fit(X_train,y_train,batch_size=32,epochs=60,validation_data=(X_test,y_test))

net.save("mlp_fire.h5")

plt.style.use("ggplot")
plt.plot(H.history['accuracy'],label="train")
plt.plot(H.history['val_accuracy'],label="test")
plt.plot(H.history["loss"],label="train loss")
plt.plot(H.history["val_loss"],label="test loss")
plt.legend()
plt.xlabel("epocha")
plt.ylabel("accuracy")
plt.title("File/None Fire Datasets")
plt.show()

loss,acc = net.evaluate(X_test,y_test)
print("loss: {:.2f}, acc: {:.2f}".format(loss,acc*100))