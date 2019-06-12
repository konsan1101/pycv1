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



def view_thresh(cn_r,cn_s):
    print("thresh init")
    
    proc_width  = 480
    proc_height = 270
    view_width  = 240
    view_height = 135
    
    blue_img = np.zeros((135,240,3), np.uint8)
    cv2.rectangle(blue_img,(0,0),(240,135),(255,0,0),cv2.cv.CV_FILLED)
    cv2.putText(blue_img, "No Data !", (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
    
    print("thresh start")
    
    fps_class = fpsWithTick()
    while True:
        if not cn_r.empty():
            image  = cn_r.get()
            if image is None:
                print("thresh None")
                break
            
            #print("thresh run")
            
            image2_img  = image.copy()
            #image2_hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #image2_chan = cv2.split(image2_hsv)
            #image2_chan[2] = cv2.equalizeHist(image2_chan[2])
            #image2_hsv  = cv2.merge(image2_chan)
            #image2_img  = cv2.cvtColor(image2_hsv, cv2.COLOR_HSV2BGR)
            
            gray   = cv2.cvtColor(image2_img, cv2.COLOR_BGR2GRAY)
            proc_gray   = cv2.resize(gray, (proc_width, proc_height))
            proc_gray   = cv2.equalizeHist(proc_gray)
            proc_thresh = cv2.blur(proc_gray, (5,5), 0)
            _, proc_thresh = cv2.threshold(proc_thresh, 112, 255, cv2.THRESH_BINARY)
            proc_thresh = cv2.bitwise_not(proc_thresh)
            
            proc_mask   = cv2.cvtColor(proc_thresh, cv2.COLOR_GRAY2RGB)
            proc_img    = cv2.resize(image2_img, (proc_width, proc_height))
            proc_img    = cv2.bitwise_and(proc_img, proc_mask)
            
            cnts0 = cv2.findContours(proc_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cnts0.sort(key=cv2.contourArea, reverse=True)
            cnts1 = [cnt for cnt in cnts0 if cv2.contourArea(cnt) > 500]
            for i, c in enumerate(cnts1):
                cv2.drawContours(proc_img, [c], -1, (255, 0, 0), 2)
                
                al = cv2.arcLength(c, True) * 0.01
                c2 = cv2.approxPolyDP(c, al, True)
                cv2.drawContours(proc_img, [c2], -1, (0, 255, 0), 2)
                
                arc_area = cv2.contourArea(c2)
                x,y,w,h  = cv2.boundingRect(c2)
                hit_area = w * h
                if (arc_area/hit_area) > 0.8:
                    cv2.rectangle(proc_img, (x,y), (x+w,y+h), (0,0,255), 2)
        
            view_img   = cv2.resize(proc_img, (view_width, view_height))
            cv2.putText(view_img, "THRESH", (40,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255))
            str_fps = str(fps_class.get())
            cv2.putText(view_img, str_fps, (40,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
            cn_s.put( view_img.copy() )
            
            thresh = cv2.resize(proc_img, (480, 270))
            cn_s.put( thresh.copy() )
        
        time.sleep(0.1)

    print("thresh end")



def view_canny(cn_r,cn_s):
    print("canny init")
    
    proc_width  = 480
    proc_height = 270
    view_width  = 240
    view_height = 135
    
    blue_img = np.zeros((135,240,3), np.uint8)
    cv2.rectangle(blue_img,(0,0),(240,135),(255,0,0),cv2.cv.CV_FILLED)
    cv2.putText(blue_img, "No Data !", (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
    
    print("canny start")
    
    fps_class = fpsWithTick()
    while True:
        if not cn_r.empty():
            image  = cn_r.get()
            if image is None:
                print("canny None")
                break
            
            #print("canny run")
            
            image2_img  = image.copy()
            #image2_hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            #image2_chan = cv2.split(image2_hsv)
            #image2_chan[2] = cv2.equalizeHist(image2_chan[2])
            #image2_hsv  = cv2.merge(image2_chan)
            #image2_img  = cv2.cvtColor(image2_hsv, cv2.COLOR_HSV2BGR)
            
            gray   = cv2.cvtColor(image2_img, cv2.COLOR_BGR2GRAY)
            proc_gray  = cv2.resize(gray, (proc_width, proc_height))
            proc_gray  = cv2.equalizeHist(proc_gray)
            proc_gray  = cv2.blur(proc_gray, (5,5), 0)
            #_, proc_thresh = cv2.threshold(proc_gray, 128, 255, cv2.THRESH_BINARY)
            #proc_thresh = cv2.bitwise_not(proc_thresh)
            #proc_canny = cv2.Canny(proc_thresh, threshold1=80, threshold2=110)
            proc_canny = cv2.Canny(proc_gray, threshold1=80, threshold2=110)
            
            proc_over  = cv2.cvtColor(proc_canny, cv2.COLOR_GRAY2BGR)
            proc_img   = cv2.resize(image2_img,(proc_width, proc_height))
            proc_img   = cv2.bitwise_or(proc_img, proc_over)
            
            lines = cv2.HoughLines(proc_canny, 1.1, np.pi/180, 150)
            if lines is not None:
                for rho,theta in lines[0]:
                    a=np.cos(theta)
                    b=np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(proc_img, (x1, y1), (x2, y2), (255,0,0), 1)
        
            lines = cv2.HoughLinesP(proc_canny, 3, np.pi/180, 10, 50, 50)
            if lines is not None:
                for (x1, y1, x2, y2) in lines[0]:
                    cv2.line(proc_img, (x1, y1), (x2, y2), (0,255,0), 2)
    
            circles = cv2.HoughCircles(proc_canny, cv2.cv.CV_HOUGH_GRADIENT, dp=1.5, minDist=80, minRadius=10, maxRadius=80)
            if circles is not None:
                circles = np.uint16(np.around(circles))
                for (x, y, r) in circles[0,:]:
                    cv2.circle(proc_img, (x, y), r, (0,0,255), 2)
    
            view_img   = cv2.resize(proc_img, (view_width, view_height))
            cv2.putText(view_img, "CANNY",  (40,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255))
            str_fps = str(fps_class.get())
            cv2.putText(view_img, str_fps, (40,45), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
            cn_s.put( view_img.copy() )
            
            canny = cv2.resize(proc_img, (480, 270))
            cn_s.put( canny.copy() )
        
        time.sleep(0.1)

    print("canny end")



def view_procD(cn_r,cn_s):
    print("procD init")

    proc_width  = 320
    proc_height = 180
    view_width  = 480
    view_height = 270
    hit_width   = 240
    hit_height  = 240
    
    blue_img = np.zeros((135,240,3), np.uint8)
    cv2.rectangle(blue_img,(0,0),(240,135),(255,0,0),cv2.cv.CV_FILLED)
    cv2.putText(blue_img, "No Data !", (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
    
    casname = cn_r.get()
    cascade = cv2.CascadeClassifier(casname)
    haar_scale    = 1.1
    min_neighbors = 2
    min_size      = (10, 10)
    
    print("procD start")

    fps_class = fpsWithTick()
    while True:
        if not cn_r.empty():
            image = cn_r.get()
            if image is None:
                print("procD None")
                break

            #print("procD run")

            image2_height, image2_width = image.shape[:2]
            image2_hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image2_chan = cv2.split(image2_hsv)
            image2_chan[2] = cv2.equalizeHist(image2_chan[2])
            image2_hsv  = cv2.merge(image2_chan)
            image2_img  = cv2.cvtColor(image2_hsv, cv2.COLOR_HSV2BGR)
            
            proc_img  = cv2.resize(image2_img,(proc_width, proc_height))
            proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.equalizeHist(proc_gray)
            view_img  = cv2.resize(image2_img,(view_width, view_height))
    
            rects = cascade.detectMultiScale(proc_gray, scaleFactor=haar_scale, minNeighbors=min_neighbors, minSize=min_size)
            if rects is not None:
                for (hit_x, hit_y, hit_w, hit_h) in rects:
                    x  = int(hit_x * image2_width / proc_width)
                    y  = int(hit_y * image2_height / proc_height)
                    w  = int(hit_w * image2_width / proc_width)
                    h  = int(hit_h * image2_height / proc_height)
                    hit_img = cv2.resize(image2_img[y:y+h, x:x+w],(hit_width,hit_height))
                    cn_s.put( hit_img.copy() )

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

    print("procD end")



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
    image_img    = cv2.resize(blue_img, (image_width, image_height))
    image_thresh = image_img.copy()
    image_canny  = image_img.copy()
    
    cv2.namedWindow("Live",   1)
    cv2.imshow(     "Live",   blue_img)
    cv2.moveWindow( "Live",   0,   25)
    cv2.namedWindow("Thresh", 1)
    cv2.imshow(     "Thresh", blue_img)
    cv2.moveWindow( "Thresh", 240, 25)
    cv2.namedWindow("Canny",  1)
    cv2.imshow(     "Canny",  blue_img)
    cv2.moveWindow( "Canny",  480, 25)

    cv2.namedWindow("Detect", 1)
    cv2.imshow(     "Detect", image_img)
    cv2.moveWindow( "Detect", 0,   200)
    cv2.namedWindow("Hit",    1)
    cv2.imshow(     "Hit",    blue_img)
    cv2.moveWindow( "Hit",    480, 200)

    capture = None
    if dev.isdigit():
        capture = cv2.VideoCapture(int(dev))
        capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  640)
        capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.cv.CV_CAP_PROP_FPS, 15)
    else:
        capture = cv2.VideoCapture(dev)

    print("main start")

    thresh_proc = None
    canny_proc  = None
    procD_proc  = None
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
            
            if thresh_proc is None:
                thresh_s = Queue()
                thresh_r = Queue()
                thresh_proc = threading.Thread(target=view_thresh, args=(thresh_s,thresh_r,))
                thresh_proc.daemon = True
                thresh_beat = time.time()
                thresh_s.put( image_img.copy() )
                thresh_proc.start()
            if thresh_r.empty():
                cv2.line(live_img, (0, 1), (0+79, 1), (0,0,255), 3)
            else:
                cv2.line(live_img, (0, 1), (0+79, 1), (255,0,0), 3)
                thresh_beat = time.time()
                img = thresh_r.get()
                cv2.imshow("Thresh", img )
                image_thresh = thresh_r.get()
                thresh_s.put( image_img.copy()  )
            if (time.time() - thresh_beat) > 10:
                print("thresh 10s")
                break

            if canny_proc is None:
                canny_s = Queue()
                canny_r = Queue()
                canny_proc = threading.Thread(target=view_canny, args=(canny_s,canny_r,))
                canny_proc.daemon = True
                canny_beat = time.time()
                canny_s.put( image_img.copy() )
                canny_proc.start()
            if canny_r.empty():
                cv2.line(live_img, (80, 1), (80+79, 1), (0,0,255), 3)
            else:
                cv2.line(live_img, (80, 1), (80+79, 1), (255,0,0), 3)
                canny_beat = time.time()
                img = canny_r.get()
                cv2.imshow("Canny", img )
                image_canny = canny_r.get()
                canny_s.put( image_img.copy() )
            if (time.time() - canny_beat) > 10:
                print("canny 10s")
                break

            if procD_proc is None:
                procD_s = Queue()
                procD_r = Queue()
                procD_proc = threading.Thread(target=view_procD, args=(procD_s,procD_r,))
                procD_proc.daemon = True
                procD_beat = time.time()
                procD_s.put(cas)
                procD_s.put( image_img.copy() )
                procD_proc.start()
            if procD_r.empty():
                cv2.line(live_img, (160, 1), (160+79, 1), (0,0,255), 3)
            else:
                cv2.line(live_img, (160, 1), (160+79, 1), (255,0,0), 3)
                procD_beat = time.time()
                while not procD_r.empty():
                    img = procD_r.get()
                    img_h, img_w = img.shape[:2]
                    if img_w != 480:
                        cv2.imshow("Hit", img )
                    else:
                        cv2.imshow("Detect", img )
                procD_s.put( image_img.copy() )
            if (time.time() - procD_beat) > 10:
                print("procD 10s")
                break

            cv2.imshow("Live", live_img )

        if cv2.waitKey(10) >= 0:
            break

        time.sleep(0.01)
    
    
    
    print("main terminate")
    
    if thresh_proc is not None:
        thresh_s.put(None)
        canny_s.put(None)
        procD_s.put(None)
        time.sleep(3)
        thresh_s.close()
        thresh_r.close()
        canny_s.close()
        canny_r.close()
        procD_s.close()
        procD_r.close()
    
    capture.release()
    cv2.destroyAllWindows()
    
    print("main Bye!")



