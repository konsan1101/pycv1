#!/bin/bash
cd /home/pi/cv
#pgrep -f ".py" | xargs kill

python /home/pi/cv/_VideoPlay.py 00 0 "/home/pi/Videos3/*" "/home/pi/Videos4/*"
