import cv2
import numpy as np
import glob
from tensorflow.keras import models


net = models.load_model("mlp_fire.h5")
labels = ["fire","not fire"]
for item in glob.glob("test_fire/*"):
    img = cv2.imread(item)
    r_img = cv2.resize(img,(32,32)).flatten()
    r_img = r_img/255
    out = net.predict(np.array([r_img]))[0]
    pred = np.argmax(out)
    print(labels[pred])
    cv2.putText(img,"{}:{:.2f}".format(labels[pred],out[pred]*100),(10,30),cv2.FONT_HERSHEY_SIMPLEX
                ,1,(0,255,0), 2)

    cv2.imshow("my Image",img)

    if cv2.waitKey(0) == ord("q"):
        break


cv2.destroyAllWindows()