from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from tensorflow.keras import backend as K
from matplotlib import pyplot
import os
import argparse
import sys

from PIL import Image
import face_recognition as fr
import os
import pandas as pd
import time
from datetime import datetime

Images = []
names = []
present = []
path = '\Directory'
dest = os.listdir(path)
df = pd.DataFrame(columns=['In time', 'Date[DD//MM//YY]', 'Name', 'Roll Number', 'Department','Late'])
data = {}

for cl in dest:
    curImg = cv2.imread(f'{path}/{cl}')
    Images.append(curImg)
    names.append(os.path.splitext(cl)[0])


def encoding(images):
    encodlis = []
    for i in images:
        fp = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        en = fr.face_encodings(fp)[0]
        encodlis.append(en)
    return encodlis


trueList = encoding(Images)

db = [ ]#NAMES
df = pd.DataFrame(
    columns=['In time', 'Date[DD//MM//YY]', 'Name', 'Roll Number', 'Department','Late'])  # ,'Section','Phone number'])
diction1 = {} #DATA SHOULD BE IN "NAME:ROLL NUMBER" FORMAT
diction2 = {} #DATA SHOULD BE IN "NAME:DEPARTMENT" FORMAT


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness.all() <= obj_thresh): continue

            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w
            y = (row + y) / grid_h
            w = anchors[2 * b + 0] * np.exp(w) / net_w
            h = anchors[2 * b + 1] * np.exp(h) / net_h
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    # Union of A,B = A + B - Intersection of (A,B)
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def load_image_pixels(filename, shape):
    image = load_img(filename)
    width, height = image.size

    image = load_img(filename, target_size=shape)

    image = img_to_array(image)

    image = image.astype('float32')
    image /= 255.0
    image = expand_dims(image, 0)
    return image, width, height


def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    for box in boxes:
        for i in range(len(labels)):
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)

    return v_boxes, v_labels, v_scores


# This is the function where embedding face recognition(pipeline 2) takes place
def draw_boxes(filename, v_boxes, v_labels, v_scores, output_dir):
    global df
    global data
    global present
    img = cv2.imread(filename)
    row = []
    for i in range(len(v_boxes)):

        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        start_point = (x1, y1)
        end_point = (x2, y2)
        color = (0, 0, 255)
        thickness = 2
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1.5
        img = cv2.rectangle(img, start_point, end_point, color, thickness)
        img3 = img[y1:y2, x1:x2]
        cv2.imwrite("facedet.jpg", img3)
        test = fr.load_image_file('facedet.jpg')
        # imgR = cv2.resize(test, (0, 0), None, 0.25, 0.25)
        imgR = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
        imgLoc = fr.face_locations(imgR)
        imgEnc = fr.face_encodings(imgR, imgLoc)
        for iL, iE in zip(imgLoc, imgEnc):
            result = fr.compare_faces(trueList, iE, tolerance=0.5)
            loc = fr.face_distance(trueList, iE)
            index = np.argmin(loc)
            if (result[index]):
                name = names[index].upper()
                y11, x22, y22, x11 = iL
                x11, y11, x22, y22 = x11 * 4, y11 * 4, x22 * 4, y22 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.rectangle(img, (x1, y2 - 6), (x2, y2), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 2, y2 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                # Report gen starts
                if name in db and name not in present:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    row.append(current_time)
                    x = datetime.now().date()
                    row.append(x)
                    row.append(name)
                    present.append(name)  # to fix that this person has already been recorded once for the day
                    row.append(diction1[name])  # diction1 contains key value pairs of name and roll numbers
                    row.append(diction2[name])  # diction2 contains key value pairs of name and department
                    curr=int(current_time[:2])
                    if curr>9 and curr<16:
                        row.append('Yes')
                    else:
                        row.append('No')
                    for k in range(len(df.columns)):
                        data[df.columns[k]] = row[k]
                        print(row[k])
                        print(data[df.columns[k]])
                    row.clear()
                    df = df.append(data, ignore_index=True)
                    print(df[['Name']])
        # draw text and score-top left
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        img = cv2.putText(img, label, (x1, y1), font,
                          fontScale, color, thickness, 2)
        text = "no.of faces detected faces: %s" % (len(v_boxes))
        img = cv2.putText(img, text, (10, 40), font,
                          fontScale, (255, 0, 0), thickness, 2)
    cv2.imshow("Yolov3+Facereg", img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    # df = df[1:]
    # print(df[['Name']])
    # path = 'H:\Project\Backend'
    # path1 = os.path.join(path, 'Report1.csv')
    # df.to_csv(path1, index=False)
    # print("Did")
    # cv2.destroyAllWindows()
    # exit()


def img_blur(filename, v_boxes, v_labels, v_scores, output_dir):
    img = cv2.imread(filename)
    rows, cols = img.shape[0], img.shape[1]
    blurred_img = cv2.GaussianBlur(img, (201, 201), 0)
    mask = np.zeros((rows, cols, 3), dtype=np.uint8)

    for i in range(len(v_boxes)):
        if (not v_boxes):
            x1, y1 = 0, 0
            x2, y2 = 0, 0
        else:
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    out = np.where(mask == np.array([255, 255, 255]), img, blurred_img)
    cv2.imshow("img_blur", out)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='',
                        help='path to image file')
    parser.add_argument('--output-dir', type=str, default='outputs/',
                        help='path to the output directory')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        print('==> Creating the {} directory...'.format(args.output_dir))
        os.makedirs(args.output_dir)
    else:
        print('==> Skipping create the {} directory...'.format(args.output_dir))
    return args


def _main():
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    cap.set(10, 150)
    # define the probability threshold for detected objects
    class_threshold = 0.6
    # define class
    labels = ["face"]
    output_dir = "\OutputDirectory"
    input_w, input_h = 416, 416
    # load and prepare image

    # load yolov3 model
    model = load_model('model.h5')
    k = 0
    if cap.isOpened():
        while True:
            global df
            global data
            print("Frame:", k)
            k = k + 1
            success, imgg = cap.read()
            im = Image.fromarray(imgg)
            im.save("filename.jpeg")
            photo_filename = "filename.jpeg"
            image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
            yhat = model.predict(image)
            # summarize the shape of the list of arrays
            boxes = list()
            for i in range(len(yhat)):
                boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
            correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
            do_nms(boxes, 0.5)  # Discard all boxes with pc less or equal to 0.5
            v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
            draw_boxes(photo_filename, v_boxes, v_labels, v_scores, output_dir)
            img_blur(photo_filename, v_boxes, v_labels, v_scores, output_dir)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(df[['Name']])
                # df = df[1:]
                path = '\OutputDirectory'
                now = datetime.now()
                outname = now.strftime("%H-%M-%S")
                path1 = os.path.join(path, outname + '-In-Report.csv')
                df.to_csv(path1, index=False)
                print("Did")
                cv2.destroyAllWindows()
                exit()
        cap.release()
    cv2.destroyAllWindows()
    K.clear_session()


if __name__ == "__main__":
    _main()








