import cv2

pathofeye='/home/apollo/pan/cv2/data/haarcascades/haarcascade_eye.xml'
pathoffront='/home/apollo/pan/cv2/data/haarcascades/haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(pathoffront)

image = cv2.imread(r"xx.jpeg")

size = image.shape
h, w = size[0], size[1]


print (h,w)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2,
minNeighbors=5, minSize=(30, 30),)
for (x, y, width, height) in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    im2 = cv2.resize(image, (int(w*0.55), int(h*0.55)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Face", im2)
    #cv2.waitKey(9000) & 0xFF == ord('q')


cv2.waitKey(0)