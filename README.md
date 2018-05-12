# Video Object Capture

This is a proof-of-concept (PoC) project (still work in progress) on how one
could use Google's Tensorflow to train and use neural network on detecting
objects in a video stream and display a frame around the object on the screen
displaying the video.


## Training

The first step is to train the neural network. For that purpose we need to
acquire a set of raw training images, classify the object(s) in the images,
and actually training the neural network.


### Acquiring of Training images

I decided that for this PoC I will use my HD video camera to record footage of
the object I care about in a variety of lighting conditions, distance, angle
and background. Once I felt satisfied that I have enough footage, I downloaded
the individual scenes to my Mac, and using iMove cut out the parts I did not
want and spliced the remaining scenes together and exported it as mpeg4 file
in 720 HD.

The next step was to extract the individual frames out of the video so that
the images could be annotated. In order to extract the frames, I wrote a Python
script that uses OpenVPC to extract the images and resizes them. You can see it
(here)[https://github.com/petr-undercover/video_object_detection/blob/master/video_frame_extract/README.md].

Note that depending on the length of video file, this can produce a large number
of JPEGs. Use the `--skip n` option to extract every n-th frame. After that I
manually reviewed the image files and deleted those that I did not want.


### Annotating of Training Images

For image labeling I found the `labelImg` on the internet that produces
annotations in the Pascal VOC format but for whatever reason, I could not get
it installed and running on my Mac. After looking around I opted for an easy
way out by getting the `RectLabel` (image annotation tool)[https://rectlabel.com/]
by Ryo Kawamura in the Apple App Store for $0.99.


TODO
