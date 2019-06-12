#!/bin/bash
cd /home/pi/cv
pgrep -f ".py" | xargs kill
python testM2.py cars.xml 0
