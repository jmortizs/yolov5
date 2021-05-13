import cv2 as cv
import argparse
import pandas as pd
from pathlib import Path
from lxml import etree
from sklearn.model_selection import train_test_split
from hashlib import sha1

def process_annotation(xml_tree, source_path: str, size: int, classes: dict):

    img_path = Path(source_path) / xml_tree.find('folder').text / xml_tree.find('filename').text

    img = cv.imread(str(img_path))
    resize_ratio = size / max(img.shape[0], img.shape[1])
    img_res = cv.resize(img, None, fx=resize_ratio, fy=resize_ratio)

    padding = size - min(img_res.shape[0], img_res.shape[1])
    top_p = 0
    bottom_p = 0
    left_p = 0
    right_p = 0

    if img_res.shape[0] < img_res.shape[1]:
        top_p = padding // 2
        bottom_p = top_p if padding % 2 == 0 else top_p + 1
        img_res = cv.copyMakeBorder(img_res, top=top_p, bottom=bottom_p, left=left_p, right=right_p, borderType=cv.BORDER_CONSTANT, value=(0, 0, 0))

    elif img_res.shape[0] > img_res.shape[1]:
        left_p = padding // 2
        right_p = left_p if padding % 2 == 0 else left_p + 1
        img_res = cv.copyMakeBorder(img_res, top=top_p, bottom=bottom_p, left=left_p, right=right_p, borderType=cv.BORDER_CONSTANT, value=(0, 0, 0))

    else:
        pass

    yolo_labels = []
    for label in xml_tree.findall('object'):
        name = label.find('name').text
        idx = classes[name]

        x1 = int(label.find('bndbox').find('xmin').text)
        y1 = int(label.find('bndbox').find('ymin').text)
        x2 = int(label.find('bndbox').find('xmax').text)
        y2 = int(label.find('bndbox').find('ymax').text)

        # relative poinst in original image
        x1_r = x1 / img.shape[1]
        y1_r = y1 / img.shape[0]
        x2_r = x2 / img.shape[1]
        y2_r = y2 / img.shape[0]

        # absolute points and relative center, widht and height in processed image
        x1_p = x1_r * (img_res.shape[1] - (left_p + right_p)) + left_p
        y1_p = y1_r * (img_res.shape[0] - (top_p + bottom_p)) + top_p
        x2_p = x2_r * (img_res.shape[1] - (left_p + right_p)) + left_p
        y2_p = y2_r * (img_res.shape[0] - (top_p + bottom_p)) + top_p
        center = (x1_p + (x2_p - x1_p)/2, y1_p + (y2_p - y1_p)/2)

        center_r = (center[0]/img_res.shape[1], center[1]/img_res.shape[0])
        width_r = (x2_p - x1_p)/img_res.shape[1]
        height_r = (y2_p -  y1_p)/img_res.shape[0]

        # cv.rectangle(
        #     img_res,
        #     (int((center_r[0]*img_res.shape[1]) - (width_r * img_res.shape[1])/2), int((center_r[1]*img_res.shape[0]) - (height_r * img_res.shape[0])/2)),
        #     (int((center_r[0]*img_res.shape[1]) + (width_r * img_res.shape[1])/2), int((center_r[1]*img_res.shape[0]) + (height_r * img_res.shape[0])/2)),
        #     (0, 255, 0), 1
        #     )

        yolo_labels.append((idx, center_r[0], center_r[1], width_r, height_r))

    #cv.imwrite('result_img.png', img_res)
    return img_res, xml_tree.find('filename').text, yolo_labels


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=str, help='source path')
    ap.add_argument('--size', required=True, type=int, help='')
    ap.add_argument('--test', required=True, type=float, help='proportion of test set')
    ap.add_argument('--output', required=True, type=str, help='output folder')

    ARGS = ap.parse_args()

    annotations = []
    # get all posible classes in the dataset
    classes = []
    for annotation in Path(ARGS.source).rglob('*.xml'):
        root = etree.parse(str(annotation))
        for class_ in root.xpath('//annotation/object/name/text()'):
            classes.append(class_)
            break

        annotations.append(root)

    # train test split
    annotations_train, annotations_test = train_test_split(annotations, test_size=ARGS.test, stratify=classes, random_state=42)

    unique_classes = sorted(set(classes))
    unique_classes = {c:i for i, c in enumerate(unique_classes)}

    folder_train_img = Path(ARGS.output) / 'train' / 'images'
    folder_train_img.mkdir(parents=True, exist_ok=True)
    folder_train_labels = Path(ARGS.output) / 'train' / 'labels'
    folder_train_labels.mkdir(parents=True, exist_ok=True)
    for annotation in annotations_train:
        img, img_name, labels = process_annotation(annotation, ARGS.source, size=ARGS.size, classes=unique_classes)
        file_id = sha1(img).hexdigest()
        file_image_name = f'{img_name}.{file_id}.png'
        file_labels_name = f'{img_name}.{file_id}.txt'

        cv.imwrite(str(folder_train_img / file_image_name), img)

        with open(folder_train_labels / file_labels_name, 'w') as f:
            for l in labels:
                f.write(f'{l[0]} {l[1]} {l[2]} {l[3]} {l[4]}\n')

    folder_test_img = Path(ARGS.output) / 'test' / 'images'
    folder_test_img.mkdir(parents=True, exist_ok=True)
    folder_test_labels = Path(ARGS.output) / 'test' / 'labels'
    folder_test_labels.mkdir(parents=True, exist_ok=True)
    for annotation in annotations_test:
        img, img_name, labels = process_annotation(annotation, ARGS.source, size=ARGS.size, classes=unique_classes)
        file_id = sha1(img).hexdigest()
        file_image_name = f'{img_name}.{file_id}.png'
        file_labels_name = f'{img_name}.{file_id}.txt'

        cv.imwrite(str(folder_test_img / file_image_name), img)

        with open(folder_test_labels / file_labels_name, 'w') as f:
            for l in labels:
                f.write(f'{l[0]} {l[1]} {l[2]} {l[3]} {l[4]}\n')









