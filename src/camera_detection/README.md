#Camera Detection

This directory contains the codes for doing realtime camera detection. 

## webcam_server.py
This script should be installed on the Raspberry Pi 4. It sets streams a live camera feed 
over http at whatever IP address has been assigned (192.168.50.1) on port 8080. 

## streamPredict.py
This script needs to be placed in the GunBlockModel directory within the Mask_RCNN codebase.
The computer needs to be on the network provided by the Raspberry Pi 4 (DeepPi4Net).
When launched, it will connect to the stream provided by the Pi4 device. It will pull one image every
2 seconds or so and display the image in a live monitor window. It runs the gun detection neural net
on the image and if a gun is detected, it will open up the detected image with mask region in another window.

## Still to do
The streamPredict.py code is slow, mostly becasue the detection process eats up a lot of time.
It also takes many frames to "catch up" to the live feed after closing the gun-detection window of a 
detected image. We need to figure out how to fix this.
