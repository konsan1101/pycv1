#!/bin/bash
cd /home/pi/cv
#pgrep -f ".py" | xargs kill

python /home/pi/cv/_VideoPlay.py 00 0 0>/dev/null 2>/dev/null &
python /home/pi/cv/_VideoPlay.py 55 1
