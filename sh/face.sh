#!/bin/bash
cd /home/pi/cv
pgrep -f ".py" | xargs kill
python testM3.py face.xml 0
