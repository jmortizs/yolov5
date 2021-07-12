import cv2 as cv
import argparse
import pandas as pd
from pathlib import Path
from lxml import etree
from sklearn.model_selection import train_test_split
from hashlib import sha1
import random
import numpy as np

def process_annotation(xml_tree, source_path: str, size: int, classes: dict, shift_range: float = None, brightness: int = None):

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

    # random shift
    if shift_range is not None:
            x_shift = random.randrange(int(-size * shift_range), int(size * shift_range), step=1)
            y_shift = random.randrange(int(-size * shift_range), int(size * shift_range), step=1)
            M = np.float32([
                [1, 0, x_shift],
                [0, 1, y_shift]
            ])

            img_res = cv.warpAffine(img_res, M, (img_res.shape[1], img_res.shape[0]), borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # random brightness
    if brightness is not None:
        value = random.uniform(1-brightness, 1+brightness)
        hsv = cv.cvtColor(img_res, cv.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        hsv[:,:,1] = hsv[:,:,1]*value
        hsv[:,:,1][hsv[:,:,1]>255]  = 255
        hsv[:,:,2] = hsv[:,:,2]*value
        hsv[:,:,2][hsv[:,:,2]>255]  = 255
        hsv = np.array(hsv, dtype = np.uint8)
        img_res = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

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

        # update label points if random shift applied
        if shift_range is not None:
            x1_p += x_shift
            y1_p += y_shift
            x2_p += x_shift
            y2_p += y_shift

            x1_p = size if x1_p > size else x1_p
            y1_p = size if y1_p > size else y1_p
            x2_p = size if x2_p > size else x2_p
            y2_p = size if y2_p > size else y2_p

            x1_p = 0 if x1_p < 0 else x1_p
            y1_p = 0 if y1_p < 0 else y1_p
            x2_p = 0 if x2_p < 0 else x2_p
            y2_p = 0 if y2_p < 0 else y2_p

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

    # cv.imwrite('result_img.png', img_res)
    return img_res, xml_tree.find('filename').text, yolo_labels


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True, type=str, help='source path')
    ap.add_argument('--size', required=True, type=int, help='')
    ap.add_argument('--test', required=True, type=float, help='proportion of test set')
    ap.add_argument('--validation', action='store_true', help='split test set in test-validation')
    ap.add_argument('--augment', required=True, type=int, help='Number of copies with random transformations for each image')
    ap.add_argument('--output', required=True, type=str, help='output folder')

    ARGS = ap.parse_args()
    MAX_SAMPLES = 220
    RAMDOM_STATE = 42

    annotations = []
    # get all posible classes in the dataset
    classes = []
    for annotation in Path(ARGS.source).rglob('*.xml'):
        root = etree.parse(str(annotation))
        for class_ in root.xpath('//annotation/object/name/text()'):
            classes.append(class_)
            break

        annotations.append(root)

    # balance
    df = pd.DataFrame({'annotation': annotations, 'class': classes})
    df_2 = pd.DataFrame()
    for c in df['class'].unique():
        df_tmp = df[df['class'] == c]
        if len(df_tmp) > MAX_SAMPLES:
            df_tmp = df_tmp.sample(n=MAX_SAMPLES, replace=False)

        df_2 = pd.concat([df_2, df_tmp])

    # train test split
    annotations_train, annotations_test = train_test_split(df_2['annotation'].values, test_size=ARGS.test, stratify=df_2['class'].values, random_state=RAMDOM_STATE)
    if ARGS.validation:
        annotations_test, annotations_val = train_test_split(annotations_test, test_size=0.5, random_state=RAMDOM_STATE)

    unique_classes = sorted(set(classes))
    unique_classes = {c:i for i, c in enumerate(unique_classes)}

    folder_train_img = Path(ARGS.output) / 'train' / 'images'
    folder_train_img.mkdir(parents=True, exist_ok=True)
    folder_train_labels = Path(ARGS.output) / 'train' / 'labels'
    folder_train_labels.mkdir(parents=True, exist_ok=True)
    # data augmentation loop
    for annotation in annotations_train:
        for n in range(ARGS.augment):
            if n == 0: # Keep the original sample
                img, img_name, labels = process_annotation(annotation, ARGS.source, size=ARGS.size, classes=unique_classes)
            else:
                img, img_name, labels = process_annotation(annotation, ARGS.source, size=ARGS.size, classes=unique_classes, shift_range=0.15, brightness=0.4)

            file_id = sha1(img).hexdigest()
            file_id = str(labels[0][0]) + file_id[:8] #class id + sha1 id
            file_image_name = f'{img_name}.{file_id}.png'
            file_labels_name = f'{img_name}.{file_id}.txt'

            cv.imwrite(str(folder_train_img / file_image_name), img)

            with open(folder_train_labels / file_labels_name, 'w') as f:
                for l in labels:
                    f.write(f'{l[0]} {l[1]} {l[2]} {l[3]} {l[4]}\n')


    # test
    folder_test_img = Path(ARGS.output) / 'test' / 'images'
    folder_test_img.mkdir(parents=True, exist_ok=True)
    folder_test_labels = Path(ARGS.output) / 'test' / 'labels'
    folder_test_labels.mkdir(parents=True, exist_ok=True)
    for annotation in annotations_test:
        img, img_name, labels = process_annotation(annotation, ARGS.source, size=ARGS.size, classes=unique_classes)
        file_id = sha1(img).hexdigest()
        file_id = str(labels[0][0]) + file_id[:8] #class id + sha1 id
        file_image_name = f'{img_name}.{file_id}.png'
        file_labels_name = f'{img_name}.{file_id}.txt'

        cv.imwrite(str(folder_test_img / file_image_name), img)

        with open(folder_test_labels / file_labels_name, 'w') as f:
            for l in labels:
                f.write(f'{l[0]} {l[1]} {l[2]} {l[3]} {l[4]}\n')

    # validation
    if ARGS.validation:
        folder_val_img = Path(ARGS.output) / 'val' / 'images'
        folder_val_img.mkdir(parents=True, exist_ok=True)
        folder_val_labels = Path(ARGS.output) / 'val' / 'labels'
        folder_val_labels.mkdir(parents=True, exist_ok=True)
        for annotation in annotations_val:
            img, img_name, labels = process_annotation(annotation, ARGS.source, size=ARGS.size, classes=unique_classes)
            file_id = sha1(img).hexdigest()
            file_id = str(labels[0][0]) + file_id[:8] #class id + sha1 id
            file_image_name = f'{img_name}.{file_id}.png'
            file_labels_name = f'{img_name}.{file_id}.txt'

            cv.imwrite(str(folder_val_img / file_image_name), img)

            with open(folder_val_labels / file_labels_name, 'w') as f:
                for l in labels:
                    f.write(f'{l[0]} {l[1]} {l[2]} {l[3]} {l[4]}\n')