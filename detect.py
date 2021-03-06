import argparse
import os
import cv2
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import pathlib
import tensorflow as tf
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='Download and process tf files')
parser.add_argument('--saved_model_path', required=True,
                    help='path to saved model')
parser.add_argument('--test_path', required=True,
                    help='path to test image')
parser.add_argument('--output_path', required=True,
                    help='path to output predicted image')
parser.add_argument('--min_score_thresh', required=False, default=0.0,
                    help='min score threshold')
parser.add_argument('--label_map', required=True,
                    help='label map path')
args = parser.parse_args()
PATH_TO_SAVED_MODEL = os.path.join(args.saved_model_path, "saved_model")
PATH_TO_TEST_IMAGE = args.test_path
PATH_TO_OUTPUT_IMAGE = args.output_path
MIN_SCORE_THRESH = float(args.min_score_thresh)
PATH_TO_LABELS = args.label_map

os.makedirs(PATH_TO_OUTPUT_IMAGE, exist_ok=True)
# Load the Labels

category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS,
    use_display_name=True
)

# Load saved model and build the detection function
print('Loading model...', end='')
start_time = time.time()
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

TOTAL_TEST_IMAGES = os.listdir(PATH_TO_TEST_IMAGE)
result_to_json = []
print(len(TOTAL_TEST_IMAGES))
for filename in TOTAL_TEST_IMAGES:
    image_id = int(filename[:-4])
    filename_path = os.path.join(PATH_TO_TEST_IMAGE, filename)
    print('Running inference for {}... '.format(filename_path))
    image_np = cv2.imread(filename_path)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {
        key: value[0, :num_detections].numpy()
        for key, value in detections.items()
    }
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    indexes = [i for i in range(len(detections["detection_scores"]))
               if detections["detection_scores"][i] > MIN_SCORE_THRESH]

    detections["detection_boxes"] = detections["detection_boxes"][indexes, ...]
    detections["detection_scores"] = detections["detection_scores"][indexes, ...]
    detections["detection_classes"] = detections["detection_classes"][indexes, ...]
    height, width = image_np_with_detections.shape[:2]
    for i in range(len(indexes)):
        det_box_info = {}
        ymin, xmin, ymax, xmax = detections["detection_boxes"][i]
        xmin, xmax = float(width*xmin), float(width*xmax)
        ymin, ymax = float(height*ymin), float(height*ymax)
        scores, classes = float(detections["detection_scores"][i]), detections["detection_classes"][i]
        category_id = int(category_index[classes]['name'])
        w = float(xmax - xmin)
        h = float(ymax - ymin)

        det_box_info["image_id"] = image_id
        det_box_info["bbox"] = [xmin, ymin, w, h]
        det_box_info["score"] = scores
        det_box_info["category_id"] = category_id

        result_to_json.append(det_box_info)

    cv2.imwrite(
        os.path.join(PATH_TO_OUTPUT_IMAGE, filename),
        image_np_with_detections[:, :, ::-1]
    )
    print('Done')


json_object = json.dumps(result_to_json, indent=4)
with open("answer.json", "w") as outfile:
    outfile.write(json_object)
print('Finish')
