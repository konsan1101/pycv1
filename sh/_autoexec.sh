#!/bin/bash

pgrep -f ".py" | xargs kill

echo "20"
sleep  1
echo "19"
sleep  1
echo "18"
sleep  1
echo "17"
sleep  1
echo "16"
sleep  1
echo "15"
sleep  1
echo "14"
sleep  1
echo "13"
sleep  1
echo "12"
sleep  1
echo "11"
sleep  1
echo "10"
sleep  1
echo " 9"
sleep  1
echo " 8"
sleep  1
echo " 7"
sleep  1
echo " 6"
sleep  1
echo " 5"
sleep  1
echo " 4"
sleep  1
echo " 3"
sleep  1
echo " 2"
sleep  1
echo " 1"
sleep  1

echo " "
echo "first play"

#1#
/home/pi/cv/_VideoPlay55.sh
#2#
#/home/pi/cv/_NewsPlay.sh &
#/home/pi/cv/_VideoPlay99.sh
#pgrep -f "_NewsPlay.py" | xargs kill
#pgrep -f "chromium"     | xargs kill
#3#
#/home/pi/cv/_VideoPlayABCDZ.sh

echo " "
echo "front play"

#1,2#
/home/pi/cv/_VideoPlay1397.sh &

echo " "
echo "image proc play"
while :
do

#1
/home/pi/cv/_cars_left.sh
sleep  2
/home/pi/cv/_VideoPlay00.sh
sleep  2

#2
#/home/pi/cv/_cars_front.sh
#/home/pi/cv/testM4.sh
#sleep  2
#/home/pi/cv/_VideoPlay00.sh
#sleep  2

echo " "
echo " "
echo " "
#pgrep -f ".py" | xargs kill
sleep  2
done
