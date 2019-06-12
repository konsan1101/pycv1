#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
from   multiprocessing import Process, Queue, Pipe, Manager, Lock

import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import glob
import random
import subprocess

print(os.path.dirname(__file__))
print(os.path.basename(__file__))
print(sys.version_info)


def view_video(cn_r,cn_s):
    print('video init')

    sw     = int(cn_r.get())
    sh     = int(cn_r.get())
    sw720  = str(sw)
    sw715  = str(sw - 5)
    sw485  = str(sw - 240 + 5)
    sh480  = str(sh)
    sh475  = str(sh - 5)
    sh325  = str(sh - 160 + 5)

    print('video start')

    while True:
        if not cn_r.empty():
            fn = cn_r.get()
            if fn is None or fn=='END':
                print('video None')
                break

            if fn=='START':
                 cn_s.put('READY')
            else:
                ppos   = str(cn_r.get())
                psound = str(cn_r.get())

                wpos = ' -r -b '
                if ppos=='0':
                    #wpos = ' --win 0,0,720,480 --layer 78 '
                    wpos = ' --win 0,0,' +sw720+ ',' +sh480+ ' --layer 78 '
                if ppos=='1':
                    wpos = ' --win 5,5,235,155 --layer 81 '
                if ppos=='4578':
                    wpos = ' --win 237,5,720,320 --layer 79 '
                if ppos=='2':
                    wpos = ' --win 5,160,235,320 --layer 82 '
                if ppos=='3':
                    wpos = ' --win 5,' +sh325+ ',235,' +sh475+ ' --layer 83 '
                if ppos=='4':
                    wpos = ' --win 240,5,480,155 --layer 84 '
                if ppos=='5':
                    wpos = ' --win 240,160,480,320 --layer 99 '
                if ppos=='55':
                    wpos = ' --win 60,40,660,440 --layer 95 '
                if ppos=='6':
                    wpos = ' --win 240,' +sh325+ ',480,' +sh475+ ' --layer 86 '
                if ppos=='7':
                    wpos = ' --win ' +sw485+ ',5,' +sw715+ ',155 --layer 87 '
                if ppos=='8':
                    wpos = ' --win ' +sw485+ ',160,' +sw715+ ',320 --layer 88 '
                if ppos=='9':
                    wpos = ' --win ' +sw485+ ',' +sh325+ ',' +sw715+ ',' +sh475+ ' --layer 89 '
            
                wsound = ' --vol -9999 '
                if psound<>'0':
                    wsound = ' -o both --vol 1 '

                if psound<>'0':
                    print('')
                    print(fn)

                cmd = 'omxplayer --fps 5 ' + wpos + wsound + ' '' + fn + '''
                subprocess.call(cmd, shell=True)
                cn_s.put('OK')

        time.sleep(1.0)

    print('video end')



if __name__ == '__main__':
    print('main init')

    app = QtGui.QApplication(sys.argv)
    app_desktop = app.desktop()
    sw = app_desktop.width()
    sh = app_desktop.height()
    print('screen size ' + str(sw) + 'x' + str(sh))
    print('')

    pmode    = '0'
    psound   = '1'
    pfolder1 = '/home/pi/Videos/*'
    pfolder2 = '/home/pi/Videos2/*'
    if len(sys.argv)>=2:
        pmode    = str(sys.argv[1])
        print('pmode:' + pmode)
    if len(sys.argv)>=3:
        psound   = str(sys.argv[2])
        print('psound:' + psound)
    if len(sys.argv)>=4:
        pfolder1 = str(sys.argv[3])
        print('pfolder1:' + pfolder1)
    if len(sys.argv)>=5:
        pfolder2 = str(sys.argv[4])
        print('pfolder2:' + pfolder2)

    files = []
    files1 = glob.glob(pfolder1)
    random.shuffle(files1)
    for fn in files1:
        files.append(fn)

    if pmode<>'00' and pmode<>'55' and pmode<>'99':
        files2 = glob.glob(pfolder2)
        random.shuffle(files2)
        for fn in files2:
            files.append(fn)
        if pmode<>'Z':
            random.shuffle(files)

    if psound<>'0':
        print('play lists')
        for fn in files:
            print(fn)
        print('')

    print('main start')
    
    pmax=1
    video_proc=[0]*pmax
    video_s   =['']*pmax
    video_r   =['']*pmax
    for p in range(pmax):
        video_proc[p] = None
    cnt=0
    while True:

        for p in range(pmax):
            if video_proc[p] is None:
                video_s[p] = Queue()
                video_r[p] = Queue()
                video_proc[p] = threading.Thread(target=view_video, args=(video_s[p],video_r[p],))
                video_proc[p].setDaemon(True)
                video_s[p].put(str(sw))
                video_s[p].put(str(sh))
                video_s[p].put('START')
                video_proc[p].start()

        for p in range(pmax):
            if not video_r[p].empty():
                video_result = video_r[p].get()
                print(video_result)

                cnt+=1
                if cnt>1:
                    if pmode=='00' or pmode=='55' or pmode=='99':
                        break

                if cnt>len(files):
                    break

                if cnt<=len(files):
                    video_s[p].put(files[cnt-1])
                    if   pmode=='00':
                        video_s[p].put('0')
                    elif pmode=='55':
                        video_s[p].put('55')
                    elif pmode=='99':
                        video_s[p].put('9')
                    elif pmode=='1397':
                        if cnt % 4 == 1:
                            video_s[p].put('1')
                        if cnt % 4 == 2:
                            video_s[p].put('3')
                        if cnt % 4 == 3:
                            video_s[p].put('9')
                        if cnt % 4 == 0:
                            video_s[p].put('7')
                    elif pmode=='Z':
                        if cnt<2:
                            video_s[p].put('4578')
                        else:
                            video_s[p].put('0')
                    elif pmode=='A':
                        if cnt % 2 == 1:
                            video_s[p].put('1')
                        else:
                            video_s[p].put('7')
                    elif pmode=='B':
                        if cnt % 2 == 1:
                            video_s[p].put('2')
                        else:
                            video_s[p].put('8')
                    elif pmode=='C':
                        if cnt % 2 == 1:
                            video_s[p].put('9')
                        else:
                            video_s[p].put('3')
                    elif pmode=='D':
                        if cnt % 2 == 1:
                            video_s[p].put('6')
                        else:
                            video_s[p].put('4')
                    else:
                         video_s[p].put(pmode)
                    video_s[p].put(psound)


        if cnt>1:
            if pmode=='00' or pmode=='55' or pmode=='99':
                break

        if cnt>len(files):
            if pmode=='1397' or pmode=='Z' or pmode=='A' or pmode=='B' or pmode=='C' or pmode=='D':
                cnt=0
            else:
                break

        time.sleep(1.0)

    print('')
    print('main terminate')
    
    for p in range(pmax):
        if not video_proc[p] is None:
            video_s[p].put(None)

    time.sleep(3)

    for p in range(pmax):
        if not video_proc[p] is None:
            video_s[p].close()
            video_r[p].close()
      
    print('main Bye!')



