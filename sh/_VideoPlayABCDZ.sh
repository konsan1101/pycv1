#!/bin/bash
cd /home/pi/cv
#pgrep -f ".py" | xargs kill

python /home/pi/cv/_VideoPlay.py A 0 1>/dev/null 2>/dev/null &
sleep 2
python /home/pi/cv/_VideoPlay.py B 0 1>/dev/null 2>/dev/null &
sleep 2
python /home/pi/cv/_VideoPlay.py C 0 1>/dev/null 2>/dev/null &
sleep 2
python /home/pi/cv/_VideoPlay.py D 0 1>/dev/null 2>/dev/null &
sleep 2
python /home/pi/cv/_VideoPlay.py Z 1 
