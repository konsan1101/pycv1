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



def view_procC(cn_r,cn_s):
    print("procC init")

    proc_width  = 320
    proc_height = 320
    view_width  = 240
    view_height = 240
    hit_width   = 240
    hit_height  = 240
    
    blue_img = np.zeros((135,240,3), np.uint8)
    cv2.rectangle(blue_img,(0,0),(240,135),(255,0,0),cv2.cv.CV_FILLED)
    cv2.putText(blue_img, "No Data !", (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))

    cascadeC1 = cv2.CascadeClassifier("testM4_C1.xml")
    cascadeC2 = cv2.CascadeClassifier("testM4_C2.xml")
    haar_scale    = 1.1
    min_neighbors = 2
    min_size      = (10, 10)
    
    hitC_img  = blue_img.copy()
    hitC_time = time.time() - 12

    print("procC start")

    fps_class = fpsWithTick()
    while True:
        if not cn_r.empty():
            image  = cn_r.get()
            if image is None:
                print("procC None")
                break

            #print("procC run")

            image_height, image_width = image.shape[:2]
            y1 = image_height/4
            y2 = y1 + image_height/2
            x1 = y1
            x2 = y2
            image2_width = x2 - x1
            image2_height = y2 - y1
            image2 = cv2.resize(image[y1:y2, x1:x2],(image2_width, image2_height))
        
            image2_img  = image2.copy()
            #image2_hsv  = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
            #image2_chan = cv2.split(image2_hsv)
            #image2_chan[2] = cv2.equalizeHist(image2_chan[2])
            #image2_hsv  = cv2.merge(image2_chan)
            #image2_img  = cv2.cvtColor(image2_hsv, cv2.COLOR_HSV2BGR)

            proc_img  = cv2.resize(image2_img,(proc_width, proc_height))
            proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.equalizeHist(proc_gray)
    
            hit_count = 0
    
            if hit_count == 0:
                rects = cascadeC1.detectMultiScale(proc_gray, scaleFactor=haar_scale, minNeighbors=min_neighbors, minSize=min_size)
                if rects is not None:
                    for (hit_x, hit_y, hit_w, hit_h) in rects:
                        hit_count += 1
                        x  = int(hit_x * image2_width / proc_width)
                        y  = int(hit_y * image2_height / proc_height)
                        w  = int(hit_w * image2_width / proc_width)
                        h  = int(hit_h * image2_height / proc_height)
                        hit_img = cv2.resize(image2_img[y:y+h, x:x+w],(hit_width,hit_height))
                        cv2.rectangle(hit_img, (5,5), (hit_width-5,hit_height-5), (0,0,255), 10)
                        cn_s.put( hit_img.copy() )
                        hitC_img  = hit_img.copy()
                        hitC_time = time.time()

            if hit_count == 0:
                rects = cascadeC2.detectMultiScale(proc_gray, scaleFactor=haar_scale, minNeighbors=min_neighbors, minSize=min_size)
                if rects is not None:
                    for (hit_x, hit_y, hit_w, hit_h) in rects:
                        hit_count += 1
                        x  = int(hit_x * image2_width / proc_width)
                        y  = int(hit_y * image2_height / proc_height)
                        w  = int(hit_w * image2_width / proc_width)
                        h  = int(hit_h * image2_height / proc_height)
                        hit_img = cv2.resize(image2_img[y:y+h, x:x+w],(hit_width,hit_height))
                        cv2.rectangle(hit_img, (5,5), (hit_width-5,hit_height-5), (0,255,255), 10)
                        cn_s.put( hit_img.copy() )
                        hitC_img  = hit_img.copy()
                        hitC_time = time.time()
        
            if hit_count == 0:
                sec = 15 - int(time.time() - hitC_time)
                if sec >= 0:
                    view_img  = cv2.resize(hitC_img,(view_width, view_height))
                    if int(time.time()) % 2 == 1:
                        cv2.rectangle(view_img, (5,5), (view_width-5,view_height-5), (255,255,255), 10)
                    cv2.putText(view_img, str(abs(sec)), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))
                else:
                    view_img  = cv2.resize(image2_img,(view_width, view_height))
                    cv2.putText(view_img, str(abs(sec)), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0))

                str_fps = str(fps_class.get())
                cv2.putText(view_img, str_fps, (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))
                cn_s.put( view_img.copy() )

        time.sleep(0.1)

    print("procC end")



def view_procB(cn_r,cn_s):
    print("procB init")

    proc_width  = 240
    proc_height = 135
    view_width  = 480
    view_height = 270
    hit_width   = 480
    hit_height  = 270
    
    blue_img = np.zeros((135,240,3), np.uint8)
    cv2.rectangle(blue_img,(0,0),(240,135),(255,0,0),cv2.cv.CV_FILLED)
    cv2.putText(blue_img, "No Data !", (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))

    cascadeB1 = cv2.CascadeClassifier("testM4_B1.xml")
    cascadeB2 = cv2.CascadeClassifier("testM4_B2.xml")
    cascadeB3 = cv2.CascadeClassifier("testM4_B3.xml")
    haar_scale    = 1.1
    min_neighbors = 2
    min_size      = (10, 10)
    
    hitB_img  = blue_img.copy()
    hitB_time = time.time() - 12

    print("procB start")

    fps_class = fpsWithTick()
    while True:
        if not cn_r.empty():
            image  = cn_r.get()
            if image is None:
                print("procB None")
                break
            thresh = cn_r.get()
            canny  = cn_r.get()

            #print("procB start")

            image_height, image_width = image.shape[:2]
            x1 = image_width/4
            x2 = x1 + image_width/2
            y1 = image_height/8
            y2 = y1 + (x2-x1) * image_height / image_width
            image2_width = x2 - x1
            image2_height = y2 - y1
            image2 = cv2.resize(image[y1:y2, x1:x2],(image2_width, image2_height))
            
            image2_img  = image2.copy()
            #image2_hsv  = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
            #image2_chan = cv2.split(image2_hsv)
            #image2_chan[2] = cv2.equalizeHist(image2_chan[2])
            #image2_hsv  = cv2.merge(image2_chan)
            #image2_img  = cv2.cvtColor(image2_hsv, cv2.COLOR_HSV2BGR)

            proc_img  = cv2.resize(image2_img,(proc_width, proc_height))
            proc_gray = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
            proc_gray = cv2.equalizeHist(proc_gray)
            
            hit_count = 0
            
            if hit_count == 0:
                rects = cascadeB1.detectMultiScale(proc_gray, scaleFactor=haar_scale, minNeighbors=min_neighbors, minSize=min_size)
                if rects is not None:
                    for (hit_x, hit_y, hit_w, hit_h) in rects:
                        hit_count += 1
                        x  = int(hit_x * image2_width / proc_width)
                        y  = int(hit_y * image2_height / proc_height)
                        w  = int(hit_w * image2_width / proc_width)
                        h  = int(hit_h * image2_height / proc_height)
                        hit_img = cv2.resize(image2_img[y:y+h, x:x+w],(hit_width,hit_height))
                        cv2.rectangle(hit_img, (5,5), (hit_width-5,hit_height-5), (0,0,255), 10)
                        cn_s.put( hit_img.copy() )
                        hitB_img  = hit_img.copy()
                        hitB_time = time.time()
        
            if hit_count == 0:
                rects = cascadeB2.detectMultiScale(proc_gray, scaleFactor=haar_scale, minNeighbors=min_neighbors, minSize=min_size)
                if rects is not None:
                    for (hit_x, hit_y, hit_w, hit_h) in rects:
                        hit_count += 1
                        x  = int(hit_x * image2_width / proc_width)
                        y  = int(hit_y * image2_height / proc_height)
                        w  = int(hit_w * image2_width / proc_width)
                        h  = int(hit_h * image2_height / proc_height)
                        hit_img = cv2.resize(image2_img[y:y+h, x:x+w],(hit_width,hit_height))
                        cv2.rectangle(hit_img, (5,5), (hit_width-5,hit_height-5), (255,0,0), 10)
                        cn_s.put( hit_img.copy() )
                        hitB_img  = hit_img.copy()
                        hitB_time = time.time()

            if hit_count == 0:
                rects = cascadeB3.detectMultiScale(proc_gray, scaleFactor=haar_scale, minNeighbors=min_neighbors, minSize=min_size)
                if rects is not None:
                    for (hit_x, hit_y, hit_w, hit_h) in rects:
                        hit_count += 1
                        x  = int(hit_x * image2_width / proc_width)
                        y  = int(hit_y * image2_height / proc_height)
                        w  = int(hit_w * image2_width / proc_width)
                        h  = int(hit_h * image2_height / proc_height)
                        hit_img = cv2.resize(image2_img[y:y+h, x:x+w],(hit_width,hit_height))
                        cv2.rectangle(hit_img, (5,5), (hit_width-5,hit_height-5), (0,255,0), 10)
                        cn_s.put( hit_img.copy() )
                        hitB_img  = hit_img.copy()
                        hitB_time = time.time()

            if hit_count == 0:
                sec = 15 - int(time.time() - hitB_time)
                if sec >= 0:
                    view_img  = cv2.resize(hitB_img,(view_width, view_height))
                    if int(time.time()) % 2 == 1:
                        cv2.rectangle(view_img, (5,5), (view_width-5,view_height-5), (255,255,255), 10)
                    cv2.putText(view_img, str(sec), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))
                elif sec >= -17:
                    view_img  = cv2.resize(image2_img,(view_width, view_height))
                    cv2.putText(view_img, str(abs(sec)), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,255))
                else:
                    view = ((abs(sec))//2) % 6
                    if view == 0:
                        view_img  = cv2.resize(image2_img,(view_width, view_height))
                        cv2.putText(view_img, str(abs(sec)), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0))
                    elif view == 1:
                        view_img  = cv2.resize(thresh,(view_width, view_height))
                        cv2.putText(view_img, str(abs(sec)), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,0))
                    elif view == 2:
                        view_img  = cv2.resize(image2_img,(view_width, view_height))
                        cv2.putText(view_img, str(abs(sec)), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0))
                    elif view == 3:
                        view_img = cv2.resize(canny,(view_width, view_height))
                        cv2.putText(view_img, str(abs(sec)), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,0))
                    elif view == 4:
                        view_img  = cv2.resize(image2_img,(view_width, view_height))
                        cv2.putText(view_img, str(abs(sec)), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,0))
                    elif view == 5:
                        view_img  = cv2.resize(canny,(view_width, view_height))
                        view_over = cv2.resize(thresh, (view_width, view_height))
                        view_img  = cv2.bitwise_or(view_img, view_over)
                        cv2.putText(view_img, str(abs(sec)), (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255,255,0))

                str_fps = str(fps_class.get())
                cv2.putText(view_img, str_fps, (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,255))
                cn_s.put( view_img.copy() )

        time.sleep(0.1)

    print("procB end")



if __name__ == '__main__':
    print("main init")
    
    dev = "0"
    if len(sys.argv)>=2:
        dev = sys.argv[1]

    base_width   = 720
    base_height  = 405
    live_width   = 240
    live_height  = 135
    image_width  = 480
    image_height = 270

    blue_img = np.zeros((live_height,live_width,3), np.uint8)
    cv2.rectangle(blue_img,(0,0),(live_width,live_height),(255,0,0),cv2.cv.CV_FILLED)
    cv2.putText(blue_img, "No Data !", (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
    base_img     = cv2.resize(blue_img, (base_width,  base_height))
    image_img    = cv2.resize(blue_img, (image_width, image_height))
    image_thresh = image_img.copy()
    image_canny  = image_img.copy()
    live_img     = blue_img.copy()
    thresh_img   = blue_img.copy()
    canny_img    = blue_img.copy()
    hitC_img     = cv2.resize(blue_img, (live_width, live_width))
    hitB_img     = image_img.copy()
    
    cv2.namedWindow("Base",  1)
    cv2.imshow(     "Base",  base_img)
    cv2.moveWindow( "Base",  0,   25)

    #cv2.namedWindow("Live",   1)
    #cv2.imshow(     "Live",   blue_img)
    #cv2.moveWindow( "Live",   0,   25)
    #cv2.namedWindow("Thresh", 1)
    #cv2.imshow(     "Thresh", blue_img)
    #cv2.moveWindow( "Thresh", 240, 25)
    #cv2.namedWindow("Canny",  1)
    #cv2.imshow(     "Canny",  blue_img)
    #cv2.moveWindow( "Canny",  480, 25)
    #cv2.namedWindow("HitC",   1)
    #cv2.imshow(     "HitC",   blue_img)
    #cv2.moveWindow( "HitC",   0,   200)
    #cv2.namedWindow("HitB",   1)
    #cv2.imshow(     "HitB",   image_img)
    #cv2.moveWindow( "HitB",   240, 200)

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
    procC_proc  = None
    procB_proc  = None
    fps_class = fpsWithTick()
    while True:
        ret, frame = capture.read()
        if not ret:
            #cv2.imshow("Live", blue_img )
            #cv2.imshow("Base", base_img )
            print("capture error")
            time.sleep(10.0)
            break

        else:

            #print("main run")

            frame_height, frame_width = frame.shape[:2]
            frame_height2 = frame_width * image_height / image_width
            #image_imgx = cv2.resize(frame, (image_width, image_height))
            if frame_height2 == frame_height:
                image_imgx = cv2.resize(frame, (image_width, image_height))
            else:
                h=(frame_height-frame_height2) / 2
                image_imgx  = cv2.resize(frame[h:h+frame_height2, 0:frame_width], (image_width, image_height))

            image_hsv  = cv2.cvtColor(image_imgx, cv2.COLOR_BGR2HSV)
            image_chan = cv2.split(image_hsv)
            image_chan[2] = cv2.equalizeHist(image_chan[2])
            image_hsv  = cv2.merge(image_chan)
            image_imgh = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

            #image_img = image_imgx.copy()
            image_img = image_imgh.copy()

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
                cv2.line(live_img, (0, 1), (0+59, 1), (0,0,255), 3)
            else:
                cv2.line(live_img, (0, 1), (0+59, 1), (255,0,0), 3)
                thresh_beat = time.time()
                thresh_img = thresh_r.get()
                #cv2.imshow("Thresh", thresh_img )
                image_thresh = thresh_r.get()
                thresh_s.put( image_img.copy() )
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
                cv2.line(live_img, (60, 1), (60+59, 1), (0,0,255), 3)
            else:
                cv2.line(live_img, (60, 1), (60+59, 1), (255,0,0), 3)
                canny_beat = time.time()
                canny_img = canny_r.get()
                #cv2.imshow("Canny", canny_img )
                image_canny = canny_r.get()
                canny_s.put( image_img.copy() )
            if (time.time() - canny_beat) > 10:
                print("canny 10s")
                break

            if procC_proc is None:
                procC_s = Queue()
                procC_r = Queue()
                procC_proc = threading.Thread(target=view_procC, args=(procC_s,procC_r,))
                procC_proc.daemon = True
                procC_beat = time.time()
                procC_s.put( image_img.copy() )
                procC_proc.start()
            if procC_r.empty():
                cv2.line(live_img, (120, 1), (120+59, 1), (0,0,255), 3)
            else:
                cv2.line(live_img, (120, 1), (120+59, 1), (255,0,0), 3)
                procC_beat = time.time()
                while not procC_r.empty():
                    hitC_img = procC_r.get()
                    #cv2.imshow("HitC", hitC_img )
                procC_s.put( image_img.copy() )
            if (time.time() - procC_beat) > 10:
                print("procC 10s")
                break

            if procB_proc is None:
                procB_s = Queue()
                procB_r = Queue()
                procB_proc = threading.Thread(target=view_procB, args=(procB_s,procB_r,))
                procB_proc.daemon = True
                procB_beat = time.time()
                procB_s.put( image_img.copy() )
                procB_s.put( image_thresh.copy() )
                procB_s.put( image_canny.copy() )
                procB_proc.start()
            if procB_r.empty():
                cv2.line(live_img, (180, 1), (180+59, 1), (0,0,255), 3)
            else:
                cv2.line(live_img, (180, 1), (180+59, 1), (255,0,0), 3)
                procB_beat = time.time()
                while not procB_r.empty():
                    hitB_img = procB_r.get()
                    #cv2.imshow("HitB", hitB_img )
                procB_s.put( image_img.copy() )
                procB_s.put( image_thresh.copy() )
                procB_s.put( image_canny.copy() )
            if (time.time() - procB_beat) > 10:
                print("procB 10s")
                break

            base_img[0:live_height, 0:live_width]       = live_img
            cv2.rectangle(base_img, (0,0), (live_width,live_height), (255,255,255), 1)
            base_img[0:live_height, 240:240+live_width] = thresh_img
            cv2.rectangle(base_img, (240,0), (240+live_width,live_height), (255,255,255), 1)
            base_img[0:live_height, 480:480+live_width] = canny_img
            cv2.rectangle(base_img, (480,0), (480+live_width,live_height), (255,255,255), 1)
            base_img[135:135+270, 0:240]                = cv2.resize(hitC_img,(240,270))
            cv2.rectangle(base_img, (0,135), (240,405), (255,255,255), 1)
            base_img[135:135+270, 240:240+480]          = hitB_img
            cv2.rectangle(base_img, (240,135), (720,405), (255,255,255), 1)

            cv2.imshow("Base", base_img)
            #cv2.imshow("Live", live_img )

        if cv2.waitKey(10) >= 0:
            break

        time.sleep(0.01)



    print("main terminate")
    
    if thresh_proc is not None:
        thresh_s.put(None)
        canny_s.put(None)
        procC_s.put(None)
        procB_s.put(None)
        time.sleep(3)
        thresh_s.close()
        thresh_r.close()
        canny_s.close()
        canny_r.close()
        procC_s.close()
        procC_r.close()
        procB_s.close()
        procB_r.close()
    
    capture.release()
    cv2.destroyAllWindows()
    
    print("main Bye!")


