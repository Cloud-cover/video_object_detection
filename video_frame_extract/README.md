
# Video Frame Extract Tool

The vidoe extract tool `vextract.py` is a tool that allows extraction of
individual frames from a video (mp4) file as JPEGs. These file can be then
classified and used for training of your neural network.

# Prerequisites

In general the prerequisites needed are defined in the `requirements.txt` file
in the parent directory. Before you get started you should create your virtualenv
using the following commands:

```
$ cd video_object_detection
$ virtualenv venv
$ source ./venv/bin/activate
$ pip install -r requirements.txt
```

In addition to the above prerequisites you will need to install OpenCV library.
As of writing of this readme, this library is not available on PiPy server. For
Windows you can download and installer but for MacOS it is a bit more complicated.

## Setup OpenCV for MacOS

In order to install OpenCV library using Homebrew you need to perform the following
steps:
* Install XCode and accept the developer license
* Install Apple Command Line Tools
* Install Homebrew
* Install OpenCV with Homebrew `$ brew install opencv3`
* Install libfree6 with Homebrew `$ brew install freetype`

Homebrew will install the OpenCV into system site-packages. If you are using
virtualenv like I am, you will need to install it manually. First confirm
that you know where Homebrew installed it. It should look like this returning
the `cv2.so` file:

```
$ ls -l /usr/local/opt/opencv/lib/python2.7/site-packages
-r--r--r--  1 me  admin   4.8M Feb 23 03:38 cv2.so
```

Once you figure out the location, then you need to add it into your virtualenv:

`$ echo /usr/local/opt/opencv/lib/python2.7/site-packages >> ./venv/lib/python2.7/site-packages/opencv3.pth`

The `freetype` install with Homebrew ensures that `libfreetype6` C-backend is
installed and can be used by Pillow. It should be installed before the
python requirements. Otherwise TensorFlow evaluation may fail when trying
to draw rectangles around recognized images (eval job during training).


# Frame extraction

Once the prerequisites are installed you can run the `vextract.py` with the
virtualenv activated:

`$ python vextract.py --skip 1 ./test/SampleVideo_1280x720_1mb.mp4`

The above command will extract every other frame from the sample video file and
write it out into the same directory as frameN.jpg where N is a simple frame
counter.  The script is written to resize the individual frame images into
640x360 which would preserve the aspect ratio of a typical HD video. The --skip
option allows for skipping every X frames which helps to keep the number of files
generated under control.
