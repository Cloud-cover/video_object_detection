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


### Creating of Training Dataset

We will be using the
(Tensorflow Object Detection API)[https://github.com/tensorflow/models/tree/master/research/object_detection]
 from now on. This requires an installation of additional pre-requisites into our
environment.  This installation consists of the following steps:

* Installation of the protobuf compiler (can be done with Homebrew)
```
$ brew install protobuf
```

* Installation of the (TensorFlow research models)[https://github.com/tensorflow/models].
This repo contains among other things the TensorFlow Object Detection API
```
$ cd ./venv/lib/python2.7/site-packages/tensorflow
$ git clone https://github.com/tensorflow/models.git
```

* Installation of COCO API Installation as documented (here)[https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md]

Note that the pip-installable pre-requisites are captured in the requirements.txt
but you may still want to check the documentation of the COCO API in case there
were changes.
```
$ cd video_object_detection
$ git clone https://github.com/cocodataset/cocoapi.git
$ cd cocoapi/PythonAPI
$ make
$ cp -r pycocotools ../../venv/lib/python2.7/site-packages/tensorflow/models/research/
```

* Protobuf Compilation as documented in the object_detection installation
```
$ cd ./venv/lib/python2.7/site-packages/tensorflow/models/research
$ protoc object_detection/protos/*.proto --python_out=.
```

* Adding Libraries to PYTHONPATH as documented in the object_detection installation
```
$ cd ./venv/lib/python2.7/site-packages/tensorflow/models/research
$ export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

* Testing the Installation as documented in the object_detection installation
```
$ cd ./venv/lib/python2.7/site-packages/tensorflow/models/research
$ python object_detection/builders/model_builder_test.py
```

At this point the environment should be ready for creation of the training dataset
in the TFRecord file format. I borrowed the `create_pascal_tf_record.py` file
from `models/research/object_detection/dataset_tools` and customized it for my
own needs. You can find it here: `./TRFecord_prep/create_pascal_tf_record.py`.

The gist of my changes is how the code finds the training file, how it handles
the file path. The following is the label map `label.pbtxt` file that I defined
for my training annotated images (in my case only annotating my dog Jessie):
```
item {
 id: 1
 name: 'Jessie'
}
```

I then created the training data set by executing the following command:
```
$ cd TFRecord_prep
$ python create_pascal_tf_record.py --data_dir=../workspace --label_map_path=../data/label.pbtxt --output_path=../data
```

The resulting training file is this `./data/pascal_train.record` and the resulting
validation file is this `./data/pascal_value.record`.

### Training Neural Network

At this point I decided to start with an existing model rather than starting
from scratch. I was mainly motivated by the fact that I did not have the time,
and resources to build a model from scratch.

TODO
