#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import threading
from   multiprocessing import Process, Queue, Pipe, Manager, Lock

import subprocess

print(os.path.dirname(__file__))
print(os.path.basename(__file__))
print(sys.version_info)


def view_kiosk(cn_r,cn_s):
    print('kiosk init')
    print('kiosk start')

    while True:
        if not cn_r.empty():
            url = cn_r.get()
            if url is None or url=='END':
                print('kiosk None')
                break

            if url=='START':
                 cn_s.put('READY')
            else:
                print('')
                print(url)

                cmd = 'chromium-browser --kiosk --disable-infobars --disable-session-crashed-bubble ' + url + ' &'
                subprocess.call(cmd, shell=True)
                time.sleep(30.0)
                cn_s.put('OK')

        time.sleep(1.0)

    print('kiosk end')



if __name__ == '__main__':
    print('main init')
    print('main start')
    
    urls =['']*9

    urls[ 0]='http://news.yahoo.co.jp/flash'
    #urls[ 1]='https://news.google.co.jp/news?cf=all&pz=1&ned=jp'
    urls[ 1]='http://news.yahoo.co.jp/'
    #urls[ 2]='https://news.google.co.jp/news/section?cf=all&pz=1&ned=jp&topic=t'
    urls[ 2]='http://news.yahoo.co.jp/hl?c=c_sci'
    #urls[ 3]='https://news.google.co.jp/news/section?cf=all&pz=1&ned=jp&topic=y'
    urls[ 3]='http://news.yahoo.co.jp/hl?c=dom'
    #urls[ 4]='https://news.google.co.jp/news/section?cf=all&pz=1&ned=jp&topic=w'
    urls[ 4]='http://news.yahoo.co.jp/hl?c=c_int'
    #urls[ 5]='https://news.google.co.jp/news/section?cf=all&pz=1&ned=jp&topic=b'
    #urls[ 5]='https://news.google.co.jp/news/section?cf=all&pz=1&ned=jp&topic=p'
    urls[ 5]='http://news.yahoo.co.jp/hl?c=bus'
    #urls[ 6]='https://news.google.co.jp/news/section?cf=all&pz=1&ned=jp&topic=e'
    urls[ 6]='http://news.yahoo.co.jp/hl?c=c_ent'
    #urls[ 7]='https://news.google.co.jp/news/section?cf=all&pz=1&ned=jp&topic=s'
    urls[ 7]='http://news.yahoo.co.jp/hl?c=c_spo'
    #urls[ 8]='https://news.google.co.jp/news/section?cf=all&pz=1&ned=jp&topic=t'
    urls[ 8]='http://news.yahoo.co.jp/hl?c=c_sci'

    pmax=1
    kiosk_proc=[0]*pmax
    kiosk_s   =['']*pmax
    kiosk_r   =['']*pmax
    for p in range(pmax):
        kiosk_proc[p] = None
    cnt=0
    while True:

        for p in range(pmax):
            if kiosk_proc[p] is None:
                kiosk_s[p] = Queue()
                kiosk_r[p] = Queue()
                kiosk_proc[p] = threading.Thread(target=view_kiosk, args=(kiosk_s[p],kiosk_r[p],))
                kiosk_proc[p].daemon = True
                kiosk_s[p].put('START')
                kiosk_proc[p].start()

        for p in range(pmax):
            if not kiosk_r[p].empty():
                kiosk_result = kiosk_r[p].get()
                print(kiosk_result)

                if cnt>=len(urls):
                    cnt=0

                cnt+=1
                if cnt<=len(urls):
                    kiosk_s[p].put(urls[cnt-1])

        if cnt>len(urls):
            cnt=0

        time.sleep(1.0)

    print('')
    print('main terminate')
    
    for p in range(pmax):
        if not kiosk_proc[p] is None:
            kiosk_s[p].put(None)

    time.sleep(3)

    for p in range(pmax):
        if not kiosk_proc[p] is None:
            kiosk_s[p].close()
            kiosk_r[p].close()
      
    print('main Bye!')



