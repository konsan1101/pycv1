#!/bin/bash
cd /home/pi/cv
#pgrep -f ".py" | xargs kill

python _cars_view.py cars.xml 0 180
