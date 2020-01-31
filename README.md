# GunBlock
A machine learning system for detecting a gun in a video stream.

# GunBlockModel
The files that make up the model to be used with the [Mask RCNN software package](https://github.com/matterport/Mask_RCNN "Mask RCNN Repo") are stored in the directory "GunBlockModel". The h5 files which store the weights to the trained model are 243MB in size, exceeding the limits allowed by Github. In addition, the Python class files that define the gun model and various other utility files are also kept here.

The files in this directory have been split from a larger tar archvive file. To reassemble this file, first retrieve all the individual files into a local archive, onto the machine that you will use for detection and on which you have installed the base Mask RCNN software.

Reassemble the archive file with
```
$ cat x* > GunBlockModel.tar.bz2
```

Then copy the GunBlockModel.tar.bz2 file to the root Mask_RCNN directory and untar it.
```
$ mv GunBlockModel.tar.bz2 /path/to/Mask_RCNN/
$ cd /path/to/Mask_RCNN
$ tar jxvf GunBlockModel.tar.bz2
```

If the training weights are ever udpated, create a new GunBlockModel.tar.bz2 file using these instructions.
```
$ cd /path/to/Mask_RCNN
$ mv ./logs/gun***/<most recent h5 file> ./logs/
$ tar --exclude='GunBlockModel/logs/gun20200129T0654' --exclude='./GunBlockModel/__pycache__' -jcvf GunBlockModel.tar.bz2 GunBlockModel/
$ split --bytes=99M GunBlockModel.tar.bz2
```
Remove the older split files
```
$ rm -f /path/to/GunBlock/GunBlockModel/x*
```
And move the new files into place
```
$ mv x* /path/to/GunBlock/GunBlockModel/
```



