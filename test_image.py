import sys
import argparse
import glob
import os
import shutil
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import cv2
import colorsys
import utils
import json

def _list_to_str(lst):
    """Convert a list to string.

    Elements are separated by comma, without any spaces inserted in between
    """

    if type(lst) is not list:
        return str(lst)

    return '[' + ','.join(_list_to_str(elem) for elem in lst) + ']'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
            
def detect_img(model):

    image_glob = FLAGS.image_glob
    test_file = FLAGS.test_file
    print(image_glob)
    print(FLAGS.model_path)
    result_name = os.path.basename(
            os.path.dirname(FLAGS.model_path))

    image_source = ""
    if image_glob:
        img_path_list = glob.glob(image_glob)
        image_source = image_glob
    else:
        with open(test_file) as f:
            img_path_list = [line.strip().split()[0] for line in f]
        image_source = test_file

    pdir = os.path.join(
        "results", FLAGS.network, os.path.basename(
            os.path.dirname(image_source)))

    output_dir = utils.get_unused_dir_num(pdir=pdir, pref=result_name)

    image_output_dir = os.path.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    prediction_output_dir = os.path.join(
        output_dir, "predictions")
    os.makedirs(prediction_output_dir, exist_ok=True)

    feature_output_dir = os.path.join(output_dir, "feature")
    os.makedirs(feature_output_dir, exist_ok=True)

    crop_output_dir = os.path.join(output_dir, "crop")
    os.makedirs(crop_output_dir, exist_ok=True)

    # Generate colors for drawing bounding boxes.
    class_num = 0
    classes_path = os.path.expanduser(FLAGS.classes_path)
    with open(classes_path) as f:
        class_num = len(f.readlines())
    colors = utils.generate_colors(class_num)

    train_bbox = ""
    train_polygon = ""
    train_json = ""
    for img_path in img_path_list:
        img_basename, _ = os.path.splitext(os.path.basename(img_path))
        try:
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            result = model.detect_image(image)
            objects = result['objects']
            objects = utils.take_contours(objects)

            # save result image with bounding box
            r_image = utils.make_r_image(image.copy(), objects, colors)
            r_image.save(
                os.path.join(
                    image_output_dir,
                    img_basename + ".jpg",
                ))

            # save feature map of middle layer
            if 'feature' in result:
                feature = result['feature']
                utils.visualize_and_save(feature, os.path.join(feature_output_dir, img_basename + ".png"))
                np.save(
                    os.path.join(
                        feature_output_dir,
                        img_basename +
                        ".npy"),
                    feature)


            train_bbox += img_path
            train_polygon += img_path
            train_json += img_path
            prediction = ""
            json_img_objs_list = []
            for obj in objects:
                # save cropped image
                class_name = obj["class_name"]
                score = obj["score"]
                image_base_name = class_name + "_" + "_".join([str(s) for s in obj["bbox"]])
                img_crop = image.crop(obj["bbox"])
                img_crop.save(os.path.join(crop_output_dir, image_base_name + ".png"))

                # train_bbox file
                x_min, y_min, x_max, y_max = obj["bbox"]
                coordinates = "{0},{1},{2},{3}".format(
                    x_min, y_min, x_max, y_max)
                train_bbox += " {coordinates},{class_id}".format(
                        coordinates=coordinates,
                        class_id=obj["class_id"],
                    )
                if 'polygon' in obj:
                    train_polygon += " [{coordinates},{class_id}]".format(
                            coordinates=_list_to_str(obj["polygon"]),
                            class_id=obj["class_id"],
                        )

                json_img_objs = {
                    "bbox": obj["bbox"],
                    "class_id": obj["class_id"],
                    "score": obj["score"],
                }
                if "all_points_x" in obj:
                    json_img_objs["all_points_x"] = obj["all_points_x"]
                    json_img_objs["all_points_y"] = obj["all_points_y"]
                # if "contours" in obj:
                #     json_img_objs["contours"] = [contour.tolist() for contour in obj["contours"]]
                #     json_img_objs["hierarchy"] = [hierarchy.tolist() for hierarchy in obj["hierarchy"]]
                json_img_objs_list.append(json_img_objs)

                # prediction file
                prediction += "{class_name}\t{score}\t{coordinates}\n".format(
                            score=score,
                            class_name=class_name,
                            coordinates="{0}\t{1}\t{2}\t{3}".format(
                                x_min, y_min, x_max, y_max),
                        )
            train_bbox += "\n"
            train_polygon += "\n"
            train_json += json.dumps(json_img_objs_list, cls = NumpyEncoder,
                                           sort_keys=True, separators=(',', ':'))
            train_json += "\n"
            # save prediction text for each image
            with open(
                    os.path.join(
                        prediction_output_dir, img_basename + ".txt"
                    ),
                    "w") as f:
                print(prediction, end="", file=f)

    shutil.copy(os.path.abspath(classes_path),
                os.path.join(output_dir, "classes.txt"))
    # save train_bbox text
    with open(os.path.join(output_dir, "train_bbox.txt"),"w") as f:
        print(train_bbox, end="", file=f)
    # save train_polygon text
    with open(os.path.join(output_dir, "train_polygon.txt"),"w") as f:
        print(train_polygon, end="", file=f)

    with open(os.path.join(output_dir, "train_json.txt"),"w") as f:
        print(train_json, end="", file=f)
    model.close_session()

def detect_video(model, video_path, output_path=""):
    import cv2

    # Generate colors for drawing bounding boxes.
    class_num = 0
    classes_path = os.path.expanduser(FLAGS.classes_path)
    with open(classes_path) as f:
        class_num = len(f.readlines())
    colors = utils.generate_colors(class_num)

    with open(classes_path) as f:
        class_num = len(f.readlines())
    hsv_tuples = [(x / class_num, 1., 1.)
                    for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        result = model.detect_image(image)
        objects = result['objects']
        r_image = utils.make_r_image(image, objects, colors)
        result = np.asarray(r_image)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    model.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '-m', '--model_path', type=str,
        help='path to model weight file'
    )

    parser.add_argument(
        '-a', '--anchors_path', type=str,
        help='path to anchor definitions'
    )

    parser.add_argument(
        "-c", '--classes_path', type=str,
        help='path to class definitions'
    )

    parser.add_argument(
        "-i", "--image_glob", nargs='?', type=str, default=None,
        help="Image glob pattern"
    )
    parser.add_argument(
        "-t", "--test_file", nargs='?', type=str, default=None,
        help="test file path"
    )
    parser.add_argument(
        '-n',
        '--network',
        type=str,
        choices=[
            'yolo',
            'mrcnn',
            'keras-centernet'],
        default='yolo',
        help='Network structure')

    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default=0,
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.network == "yolo":
        from yolo import YOLO
        model = YOLO(**vars(FLAGS))

    elif FLAGS.network == "mrcnn":
        from mask_rcnn import MaskRCNN
        model = MaskRCNN(FLAGS.model_path, FLAGS.classes_path)

    elif FLAGS.network == "keras-centernet":
        from centernet import CENTERNET
        model = CENTERNET(FLAGS.model_path, FLAGS.classes_path)

    else:
        parser.error("Unknown network")

    detect_img(model)
    # detect_video(model, FLAGS.input, FLAGS.output)  
