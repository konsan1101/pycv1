#!/bin/bash
cd /home/pi/cv
pgrep -f ".py" | xargs kill
python hitomi.py hitomi_kero.xml 0
