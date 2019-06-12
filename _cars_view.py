#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
    print('image init')
    
    proc_width  = 320
    proc_height = 180
    view_width  = 720
    view_height = 405

    casname1 = cn_r.get()
    print('image ' + str(casname1))
    cascade1 = cv2.CascadeClassifier(casname1)
    casname2 = cn_r.get()
    print('image ' + str(casname2))
    cascade2 = cv2.CascadeClassifier(casname2)

    haar_scale1    = 1.1
    min_neighbors1 = 2
    min_size1      = ( 20, 20)
    haar_scale2    = 1.2
    min_neighbors2 = 3
    min_size2      = ( 20, 20)

    print('image start')
    
    fps_class = fpsWithTick()
    while True:
        if not cn_r.empty():
            image = cn_r.get()
            if image is None:
                print('image None')
                break
        
            #print('image run')
            
            #view_img = cv2.resize(image, (view_width, view_height))
            view_img  = image.copy()

            gray  = cv2.resize(image, (proc_width, proc_height))
            gray1 = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.equalizeHist(gray1)
            
            hit_count = 0

            rects1 = cascade1.detectMultiScale(gray2, scaleFactor=haar_scale1, minNeighbors=min_neighbors1, minSize=min_size1)
            if rects1 is not None:
                for (hit_x, hit_y, hit_w, hit_h) in rects1:
                    hit_count += 1
                    x  = int(hit_x * view_width  / proc_width )
                    y  = int(hit_y * view_height / proc_height)
                    w  = int(hit_w * view_width  / proc_width )
                    h  = int(hit_h * view_height / proc_height)
                    cv2.rectangle(view_img, (x,y), (x+w,y+h), (0,0,255), 2)
    
            rects2 = cascade2.detectMultiScale(gray2, scaleFactor=haar_scale2, minNeighbors=min_neighbors2, minSize=min_size2)
            if rects2 is not None:
                for (hit_x, hit_y, hit_w, hit_h) in rects2:
                    hit_count += 1
                    x  = int(hit_x * view_width  / proc_width )
                    y  = int(hit_y * view_height / proc_height)
                    w  = int(hit_w * view_width  / proc_width )
                    h  = int(hit_h * view_height / proc_height)
                    cv2.rectangle(view_img, (x,y), (x+w,y+h), (0,255,0), 2)

            cn_s.put( hit_count )

            str_fps = str(fps_class.get())
            if hit_count > 0:
                cv2.rectangle(view_img, (10,10), (view_width-10,view_height-10), (0,0,255), 20)
                cv2.putText(view_img, casname1, (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255))
                cv2.putText(view_img, str_fps, (40,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
                cn_s.put( view_img.copy() )

        time.sleep(0.1)
    
    print('image end')



if __name__ == '__main__':
    print('main init')

    rote  =   0  #normal
    #rote =  90  #left
    #rote = -90  #right
    #rote = 180  #back

    dev = '0'
    cas = 'cars.xml'
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
    elif len(sys.argv)==4:
        cas  = sys.argv[1]
        dev  = sys.argv[2]
        rote = sys.argv[3]

    base_width     = 720
    base_height    = 405
    live_width     = 240
    live_height    = 135
    image_width    = 720
    image_height   = 405

    blue_img = np.zeros((live_height,live_width,3), np.uint8)
    cv2.rectangle(blue_img,(0,0),(live_width,live_height),(255,0,0),-1)
    cv2.putText(blue_img, 'No Data !', (40,80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
    base_img     = cv2.resize(blue_img, (base_width,  base_height ))
    image_img    = cv2.resize(blue_img, (image_width, image_height))

    cv2.namedWindow('Base',  1)
    cv2.imshow(     'Base',  base_img)
    cv2.moveWindow( 'Base',  0,   25)

    #cv2.namedWindow('Live',  1)
    #cv2.moveWindow( 'Live',  0, 25)
    #cv2.imshow(     'Live',  blue_img )
    #cv2.namedWindow('Image', 1)
    #cv2.moveWindow( 'Image', 240, 25)
    #cv2.imshow(     'Image', image_img)

    capture = None
    if dev.isdigit():
        capture = cv2.VideoCapture(int(dev))
        #capture.set(cv2.CV_CAP_PROP_FRAME_WIDTH,  640)
        #capture.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, 480)
        #capture.set(cv2.CV_CAP_PROP_FPS, 15)
    else:
        capture = cv2.VideoCapture(dev)

    print('main start')
    
    image_proc = None
    fps_class = fpsWithTick()
    while True:
        ret, frame = capture.read()
        #if not ret:
        if 1 == 2:
            #cv2.imshow('Live', blue_img )
            #cv2.imshow('Base', base_img )
            print('capture error')
            time.sleep(10.0)
            break

        else:
        
            #print('main run')

            frame_height, frame_width = frame.shape[:2]
            frame_width2  = int((frame_width - frame_height)/2)

            frame_hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_chan = cv2.split(frame_hsv)
            frame_chan[2] = cv2.equalizeHist(frame_chan[2])
            frame_hsv  = cv2.merge(frame_chan)
            frame_img  = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
            frame_img  = frame.copy()

            if abs(int(rote)) == 90:
                rect_img  = cv2.resize(frame_img[0:frame_height, frame_width2:frame_height], (960,960))
                rect_mat  = cv2.getRotationMatrix2D((480, 480), int(rote), 1.0)
                rect_r    = cv2.warpAffine(rect_img, rect_mat, (960, 960), flags=cv2.INTER_LINEAR)
                image_img = cv2.resize(rect_r, (image_width, image_height))
            else:
                image_img = cv2.resize(frame_img, (image_width, image_height))

            if abs(int(rote)) == 180:
                image_img = cv2.flip(image_img,0)

            base_img = image_img.copy()

            if image_proc is None:
                image_s = Queue()
                image_r = Queue()
                image_proc = threading.Thread(target=view_image, args=(image_s,image_r,))
                #image_proc = Process(target=view_image, args=(image_s,image_r,))
                image_proc.setDaemon(True)
                image_beat = time.time()
                image_s.put(cas)
                image_s.put('fullbody.xml')
                image_s.put( image_img.copy() )
                hit_beat  = time.time()
                hit_image = image_img.copy()
                image_proc.start()
            if image_r.empty():
                cv2.line(base_img, (0, 1), (base_width, 1), (0,0,255), 3)
            else:
                cv2.line(base_img, (0, 1), (base_width, 1), (255,0,0), 3)
                image_beat = time.time()
                hit_count = image_r.get()
                if int(hit_count) != 0:
                    hit_beat  = time.time()
                    hit_image = image_r.get()
                image_s.put( image_img.copy() )
            if (time.time() - image_beat) > 10:
                print('image 10s')
                break

            str_src = str(frame_width) + 'x' + str(frame_height)
            str_fps = str(fps_class.get())
            cv2.putText(base_img, str_src, (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255))
            cv2.putText(base_img, str_fps, (40,70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))

            sec = int(time.time() - hit_beat)
            if sec < 5:
                live_img = cv2.resize(image_img, (live_width, live_height))
                cv2.putText(live_img, str_src, (40,30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255))
                cv2.putText(live_img, str_fps, (40,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255))
                #cv2.imshow('Live', live_img)
                
                base_img = hit_image.copy()
                if abs(int(rote)) != 180:
                    base_img[25:live_height+25, image_width-live_width-25:image_width-25] = live_img.copy()
                    cv2.rectangle(base_img, (image_width-live_width-25,25), (image_width-25,live_height+25), (255,255,255), 1)
                else:
                    base_img[image_height-live_height-25:image_height-25, image_width-live_width-25:image_width-25] = live_img.copy()
                    cv2.rectangle(base_img, (image_width-live_width-25,image_height-live_height-25), (image_width-25,image_height-25), (255,255,255), 1)

                if int(time.time()) % 2 == 1:
                    cv2.rectangle(base_img, (10,10), (image_width-10,image_height-10), (255,255,255), 20)

            cv2.imshow('Base', base_img )
            #cv2.imshow('Live', live_img )

        if cv2.waitKey(10) >= 0:
            break

        time.sleep(0.01)



    print('main terminate')
    
    if image_proc is not None:
        image_s.put(None)
        time.sleep(3)
        image_s.close()
        image_r.close()
    
    capture.release()
    cv2.destroyAllWindows()
    
    print('main Bye!')



