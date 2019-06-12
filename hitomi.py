#!/usr/bin/python
import sys, cv2
if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(sys.argv[1])
    haar_scale    = 1.1
    min_neighbors = 2
    min_size      = (10, 10)

    cv2.namedWindow("Live",   1)

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        image  = cv2.resize(frame,(320, 240))
        gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.equalizeHist(gray1)
        rects = cascade.detectMultiScale(gray2, scaleFactor=haar_scale, minNeighbors=min_neighbors, minSize=min_size)
        if rects is not None:
            for (x, y, w, h) in rects:
                cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)

        cv2.imshow("Live", image)

        if cv2.waitKey(10) >= 0:
            break

    capture.release()
    cv2.destroyAllWindows()


