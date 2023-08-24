#Impelementation with KNN Classification

from sklearn.neighbors import KNeighborsClassifier 
import pandas as pd
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump,load



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


X_train,X_test,y_train,y_test = train_test_split(data,lables,test_size=0.3)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
out = clf.score(X_test,y_test)

print("Accuaracy : {:.2f} %".format(out * 100))

data = np.array(data)

dump(clf,"clf_knn.joblib")