import glob
import cv2
import joblib

clf = joblib.load("clf_knn.joblib")


for item in glob.glob("test_fire/*"):
    img = cv2.imread(item)
    r_img = cv2.resize(img,(32,32)).flatten()
    print(r_img)
    out = clf.predict([r_img])[0]
    cv2.putText(img,out,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),3)
    cv2.imshow("my Image",img)

    if cv2.waitKey(0) == ord("q"):
        break

cv2.destroyAllWindows()
