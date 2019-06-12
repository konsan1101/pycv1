#!/bin/bash
cd /home/pi/cv
pgrep -f ".py" | xargs kill
python hitomi.py face.xml 0
