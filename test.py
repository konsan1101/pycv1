#!/usr/bin/python

import os
import sys
import numpy as np
import cv2
import time
import threading
from   multiprocessing import Process, Queue, Pipe, Manager, Lock

print(os.path.dirname(__file__))
print(os.path.basename(__file__))
print(sys.version_info)
print(cv2.__version__)

class fpsWithTick(object):
    def __init__(self):
        self._count     = 0
        self._oldCount  = 0
        self._freq      = 1000 / cv2.getTickFrequency()
        self._startTime = cv2.getTickCount()
    def get(self):
        nowTime         = cv2.getTickCount()
        diffTime        = (nowTime - self._startTime) * self._freq
        self._startTime = nowTime
        fps             = (self._count - self._oldCount) / (diffTime / 1000.0)
        self._oldCount  = self._count
        self._count     += 1
        fpsRounded      = round(fps, 2)
        return fpsRounded



def view_image(cn_r,cn_s):
    print("image init")

    proc_width  = 320
    proc_height = 180
    view_width  = 480
    view_height = 270
    
    blue_img = np.zeros((135,240,3), np.uint8)
    cv2.rectangle(blue_img,(0,0),(240,135),(255,0,0),cv2.cv.CV_FILLED)
    cv2.putText(blue_img, "No Data !", (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))

    casname = cn_r.get()
    cascade = cv2.CascadeClassifier(casname)
    haar_scale    = 1.1
    min_neighbors = 2
    min_size      = (10, 10)
    
    print("image start")

    fps_class = fpsWithTick()
    while True:
        if not cn_r.empty():
            image = cn_r.get()
            if image is None:
                print("image None")
                break

            #print("image run")

            proc_img  = cv2.resize(image,(proc_width, proc_height))
            proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.equalizeHist(proc_gray)
            view_img  = cv2.resize(image,(view_width, view_height))
        
            rects = cascade.detectMultiScale(proc_gray, scaleFactor=haar_scale, minNeighbors=min_neighbors, minSize=min_size)
            if rects is not None:
                for (hit_x, hit_y, hit_w, hit_h) in rects:
                    lx  = int(hit_x * view_width / proc_width)
                    ly  = int(hit_y * view_height / proc_height)
                    lw  = int(hit_w * view_width / proc_width)
                    lh  = int(hit_h * view_height / proc_height)
                    lxw = lx + lw
                    lyh = ly + lh
                    cv2.rectangle(view_img, (lx,ly), (lxw,lyh), (0,0,255), 2)

            cv2.putText(view_img, casname, (40,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255))
            str_fps = str(fps_class.get())
            cv2.putText(view_img, str_fps, (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))
            cn_s.put( view_img.copy() )

        time.sleep(0.1)

    print("image end")



if __name__ == '__main__':
    print("main init")
    
    dev = "0"
    cas = "face.xml"
    if len(sys.argv)==2 and sys.argv[1].isdigit():
        dev = sys.argv[1]
    elif len(sys.argv)==2:
        cas = sys.argv[1]
    elif len(sys.argv)==3 and sys.argv[2].isdigit():
        cas = sys.argv[1]
        dev = sys.argv[2]
    elif len(sys.argv)==3 and sys.argv[1].isdigit():
        cas = sys.argv[2]
        dev = sys.argv[1]
    elif len(sys.argv)==3:
        cas = sys.argv[1]
        dev = sys.argv[2]

    live_width     = 240
    live_height    = 135
    image_width    = 480
    image_height   = 270
    
    blue_img = np.zeros((live_height,live_width,3), np.uint8)
    cv2.rectangle(blue_img,(0,0),(live_width,live_height),(255,0,0),cv2.cv.CV_FILLED)
    cv2.putText(blue_img, "No Data !", (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
    image_img = cv2.resize(blue_img, (image_width, image_height))

    cv2.namedWindow("Live",   1)
    cv2.imshow(     "Live",   blue_img)
    cv2.moveWindow( "Live",   0,   25)
    cv2.namedWindow("Image",  1)
    cv2.imshow(     "Image",  image_img)
    cv2.moveWindow( "Image",  240, 25)

    capture = None
    if dev.isdigit():
        capture = cv2.VideoCapture(int(dev))
        capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  640)
        capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.cv.CV_CAP_PROP_FPS, 15)
    else:
        capture = cv2.VideoCapture(dev)

    print("main start")

    image_proc = None
    fps_class = fpsWithTick()
    while True:
        ret, frame = capture.read()
        if not ret:
            cv2.imshow("Live", blue_img )
            print("capture error")
            time.sleep(10.0)
            break

        else:
        
            #print("main run")

            frame_height, frame_width = frame.shape[:2]
            frame_height2 = frame_width * image_height / image_width
            if frame_height2 == frame_height:
                image_img = cv2.resize(frame, (image_width, image_height))
            else:
                h=(frame_height-frame_height2) / 2
                image_img  = cv2.resize(frame[h:h+frame_height2, 0:frame_width], (image_width, image_height))
            live_img = cv2.resize(image_img, (live_width, live_height))
            
            str_src = str(frame_width) + "x" + str(frame_height)
            str_fps = str(fps_class.get())
            cv2.putText(live_img, str_src, (40,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255))
            cv2.putText(live_img, str_fps, (40,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
            #cv2.imshow("Live", live_img)
            
            if image_proc is None:
                image_s = Queue()
                image_r = Queue()
                image_proc = threading.Thread(target=view_image, args=(image_s,image_r,))
                image_proc.daemon = True
                image_beat = time.time()
                image_s.put(cas)
                image_s.put( image_img.copy() )
                image_proc.start()
            if image_r.empty():
                cv2.line(live_img, (0, 1), (240, 1), (0,0,255), 3)
            else:
                cv2.line(live_img, (0, 1), (240, 1), (255,0,0), 3)
                image_beat = time.time()
                img = image_r.get()
                cv2.imshow("Image", img )
                image_s.put( image_img.copy() )
            if (time.time() - image_beat) > 10:
                print("image 10s")
                break

            cv2.imshow("Live", live_img )

        if cv2.waitKey(10) >= 0:
            break
    
        time.sleep(0.01)



    print("main terminate")

    if image_proc is not None:
        image_s.put(None)
        time.sleep(3)
        image_s.close()
        image_r.close()

    capture.release()
    cv2.destroyAllWindows()

    print("main Bye!")


