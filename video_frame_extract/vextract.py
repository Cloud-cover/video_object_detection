""""
 vextract.py - This script can be used to extract individual frames
 from an mp4 video files into a set of JPEGs. It can be invoked using
 the following syntax:

 python vextract.py ./some/path/myvideo.mp4

 It will create a series of JPEG filex named frameN.jpg in the same
 directory where N is a integer counter starting at 0. It also supports
 --skip X option so that you can skip every X frames. This is handy
 when the scene is not changing rapidly and you need to control the
 number of images that you will utlimately use for classification and
 training of your neural network.  Example use:

 python vextract.py --skip 2 ./test/SampleVideo_1280x720_1mb.mp4

"""
import argparse
import os
import cv2
from PIL import Image


def extract(video_file, skip_frame):
    if not os.path.isfile(video_file):
        print("The %s does not appear to be a file" % video_file)
        return

    # get absolute path to the video file
    path = os.path.dirname(os.path.abspath(video_file))

    # target size of the JPG images
    # HD: 1080 lines with 1920 pixels (1920x1080)
    # HD: 720 lines with 1280 pixels (1280x720)
    size = 640, 360 # using 1/2 of 720 HD, same aspect ratio

    vidcap = cv2.VideoCapture(video_file)
    success,image = vidcap.read()
    count = 0
    skip_count = 0
    success = True

    while success:
        if 0 == skip_count:
            new_file_name = "%s/frame%d.jpg" % (path, count)

            # write out the frame as JPEG file
            cv2.imwrite(new_file_name, image)

            # transform the JPEG into a target size
            # TODO do all this in memory before writing this image to disk
            im = Image.open(new_file_name)
            new_im = im.resize(size)
            im.close()
            new_im.save(new_file_name)

        success,image = vidcap.read()
        count += 1
        skip_count += 1

        if skip_count > skip_frame:
            skip_count = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__file__, description='Extract still frames from a video file.')
    parser.add_argument('video_file', nargs=1, help='File name of the mp4 video file')
    parser.add_argument('--skip', type=int, nargs=1, default=0, help='Number of frames to skip')
    args = parser.parse_args()

    print(cv2.__version__)
    video_file = args.video_file[0]
    skip_frame = args.skip[0]
    extract(video_file, skip_frame)
