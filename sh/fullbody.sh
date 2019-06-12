#!/bin/bash
cd /home/pi/cv
pgrep -f ".py" | xargs kill
python testM3.py fullbody.xml 0
