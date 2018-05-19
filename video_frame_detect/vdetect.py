""""
 vdetect.py - Object detection using supplied frozen inference graph on the
 provided video

 python vdetect.py ./path/to/frozen/inference/graph.pb ./path/to/label/map.pbtxt ./some/path/video.mp4

"""
import argparse
import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import object_detection.utils.label_map_util as label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2

NUM_CLASSES = 100


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_pil_image_from_opencv(cv2_image):
    ci = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(ci)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
              'num_detections', 'detection_boxes', 'detection_scores',
              'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

                # Reframe is required to translate mask from box coordinates to image
                # coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)

                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=__file__, description='Detect trained object in individual frames of a video file.')
    parser.add_argument('frozen_model', nargs=1, help='File name of the frozen model')
    parser.add_argument('label_map', nargs=1, help='File name of the object label map')
    parser.add_argument('video_file', nargs=1, help='File name of the mp4 video file')
    args = parser.parse_args()

    print(cv2.__version__)
    frozen_model = args.frozen_model[0]
    label_map = args.label_map[0]
    video_file = args.video_file[0]
    new_video_file = video_file + '.new.avi'

    # Load frozen TensorFlow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Label maps map indices to category names, so that when our convolution
    # network predicts `5`, we know that this corresponds to `airplane`.
    label_map = label_map_util.load_labelmap(label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # open the video
    vidcap = cv2.VideoCapture(video_file)
    success,cv2_image = vidcap.read()

    # load OpenCV image into PIL.Image
    image = load_pil_image_from_opencv(cv2_image)

    # setup the video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print(int(vidcap.get(cv2.CAP_PROP_FPS)))
    out = cv2.VideoWriter(new_video_file, fourcc, 30, image.size)

    while success:
        # Actual detection.
        output_dict = run_inference_for_single_image(cv2_image, detection_graph)

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(cv2_image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        # write the modified frame
        out.write(cv2_image)

        success,cv2_image = vidcap.read()
