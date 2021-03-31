import argparse
import copy
import os
import glob
import json
import cv2


def coco_split(load_dir, save_dir, n_rows, n_columns, intersect_border):
    load_pathes = glob.glob(os.path.join(load_dir, '*.json'))
    load_pathes.sort()
    
    # (共通部分のbbox, 面積比) を返す
    def calc_intersect_area(bbox, i, j, height, width):
        before_area = bbox[2] * bbox[3]
        sx = max(bbox[0], j * width // n_columns)
        ex = min(bbox[0] + bbox[2], (j + 1) * width // n_columns)
        sy = max(bbox[1], i * height // n_rows)
        ey = min(bbox[1] + bbox[3], (i + 1) * height // n_rows)

        after_area = (ey - sy) * (ex - sx) if ex > sx and ey > sy else 0

        return [sx, sy, ex - sx, ey - sy], after_area / before_area

    os.makedirs(save_dir, exist_ok=True)

    for load_json_path in load_pathes:

        img_cnt = 1
        img_id_dict = {}

        with open(load_json_path) as f:
            data = json.load(f)

            save_json_path = os.path.join(save_dir, os.path.basename(load_json_path))
            save_json = { 'images': [], 'annotations': [], 'categories': data['categories']}

            img_sizes = {}

            for image_json in data['images']:
                id = image_json['id']
                path = os.path.join(os.path.dirname(load_json_path), image_json['path'])
                file_name = image_json['file_name']
                height = image_json['height']
                width = image_json['width']
                img_sizes[image_json['id']] = {'height': height, 'width': width}
                img = cv2.imread(path)
                for i in range(0, n_rows):
                    for j in range(0, n_columns):
                        body, ext = os.path.splitext(file_name)
                        suffix = "_%d_%d" % (i, j)
                        write_img_basename = body + suffix + ext
                        save_img_path = os.path.join(save_dir, write_img_basename)
                        cropped = img[i * height // n_rows : (i + 1) * height // n_rows, j * width // n_columns : (j + 1) * width // n_columns]
                        img_id_dict[(id, i, j)] = img_cnt
                        cv2.imwrite(save_img_path, cropped)

                        save_img_json = {}
                        save_img_json['id'] = img_cnt
                        save_img_json['dataset_id'] = image_json['dataset_id']
                        save_img_json['file_name'] = write_img_basename
                        save_img_json['path'] = save_img_path
                        save_img_json['height'] = cropped.shape[0]
                        save_img_json['width'] = cropped.shape[1]
                        save_json['images'].append(save_img_json)
                        img_cnt += 1

            for anot in data['annotations']:
                for i in range(0, n_rows):
                    for j in range(0, n_columns):
                        bbox, intersect_per = calc_intersect_area(anot['bbox'], i, j, img_sizes[anot['image_id']]['height'], img_sizes[anot['image_id']]['width'])
                        if intersect_per > intersect_border:
                            save_anot =copy.deepcopy(anot)
                            save_anot['image_id'] = img_id_dict[(id, i, j)]
                            save_anot['bbox'] = [bbox[0] - j * width // n_columns, bbox[1] - i * height // n_rows, bbox[2], bbox[3]]
                            save_anot['area'] = bbox[2] * bbox[3]
                            segmentation = []
                            for polygon in anot['segmentation']:
                                after_polygon = []
                                for idx, pos in enumerate(polygon):
                                    if idx % 2 == 0:
                                        after_polygon.append(max(j * width // n_columns, min((j + 1) * width // n_columns, pos)) - j * width // n_columns)
                                    else:
                                        after_polygon.append(max(i * height // n_rows, min((i + 1) * height // n_rows, pos)) - i * height // n_rows)
                                segmentation.append(after_polygon)
                            save_anot['segmentation'] = segmentation
                            save_json['annotations'].append(save_anot)

            with open(save_json_path, 'w') as outfile:
                json.dump(save_json, outfile)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--file_dir', required=True,
        help='path to annotation file directory'
    )
    parser.add_argument(
        '-s', '--save_dir', required=True,
        help='path to save file directory')

    parser.add_argument(
        '-b', '--border', default=0.0, required=False,
        help='border of intersect area percentage')

    parser.add_argument(
        '-r', '--rows', required=True, type=int,
        help='number of rows'
    )

    parser.add_argument(
        '-c', '--columns', required=True, type=int,
        help='number of columns'
    )

    args = parser.parse_args()
    coco_split(args.file_dir, args.save_dir, args.rows, args.columns, args.border)


if __name__ == '__main__':
    main()