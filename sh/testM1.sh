#!/bin/bash
cd /home/pi/cv
pgrep -f ".py" | xargs kill
python testM1.py cars.xml 0
