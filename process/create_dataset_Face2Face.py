import argparse
import cv2
import numpy as np
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=300, help='the height / width of the input image to network')
parser.add_argument('--dataset', default ='', help='path to dataset')
# parser.add_argument('--dataset', default ='datasets/FaceForensics/selfreenactment', help='path to dataset')
parser.add_argument('--mask', default ='', help='mask videos')
parser.add_argument('--output', default = '', help= 'name of output folder')
parser.add_argument('--scale', type=float, default =1.6, help='enables resizing')

opt = parser.parse_args()
print(opt)

def to_bw(mask, thresh_binary=127, thresh_otsu=255):
    im_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, thresh_binary, thresh_otsu, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return im_bw


def get_bbox(mask, thresh_binary=127, thresh_otsu=255):
    im_bw = to_bw(mask, thresh_binary, thresh_otsu)

    # im2, contours, hierarchy = cv2.findContours(im_bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    locations = np.array([], dtype=np.int32).reshape(0, 5)

    for c in contours:
        # calculate moments for each contour
        M = cv2.moments(c)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
        else:
            cX = 0
        if M["m00"] > 0:
            cY = int(M["m01"] / M["m00"])
        else:
            cY = 0

        # calculate the rectangle bounding box
        x, y, w, h = cv2.boundingRect(c)
        locations = np.concatenate((locations, np.array([[cX, cY, w, h, w + h]])), axis=0)

    if len(locations) == 0 or len(locations[0]) < 4:
        return None

    max_idex = locations[:, 4].argmax()
    bbox = locations[max_idex, 0:4].reshape(4)

    return bbox


def extract_face(image, bbox, scale=2.0):
    h, w, d = image.shape
    radius = int(bbox[3] * scale / 2)

    y_1 = bbox[1] - radius
    y_2 = bbox[1] + radius
    x_1 = bbox[0] - radius
    x_2 = bbox[0] + radius

    if x_1 < 0:
        x_1 = 0
    if y_1 < 0:
        y_1 = 0
    if x_2 > w:
        x_2 = w
    if y_2 > h:
        y_2 = h

    crop_img = image[y_1:y_2, x_1:x_2]

    if crop_img is not None:
        crop_img = cv2.resize(crop_img, (opt.imageSize, opt.imageSize))

    return crop_img


def extract_face_videos(output_path):
    f_vid_mask = opt.mask
    f_vid_altered = opt.dataset

    f_img_altered = os.path.join(output_path, 'Face2Face')

    blank_img = np.zeros((opt.imageSize, opt.imageSize, 3), np.uint8)

    all = 0
    for vid in os.listdir(f_vid_mask):
        if not os.path.exists(os.path.join(f_img_altered, vid)):
            os.makedirs(os.path.join(f_img_altered, vid))
        print(vid + ':', end=' ')
        count = 0
        if not vid.endswith('mp4'):
            continue
        if os.path.exists(os.path.join(f_vid_mask, vid)):
            for f in os.listdir(os.path.join(f_vid_mask, vid)):
                image_mask = cv2.imread(os.path.join(f_vid_mask, vid, f))
                bbox = get_bbox(image_mask)

                if bbox is None:
                    count += 1
                    continue
                
                image_altered = cv2.imread(os.path.join(f_vid_altered, vid, f))
                altered_cropped = extract_face(image_altered, bbox, opt.scale)

                mask_cropped = to_bw(extract_face(image_mask, bbox, opt.scale))
                mask_cropped = np.stack((mask_cropped,mask_cropped, mask_cropped), axis=2)

                if altered_cropped is not None and (mask_cropped is not None):
                    altered_cropped = np.concatenate((altered_cropped, mask_cropped, mask_cropped), axis=1)

                    cv2.imwrite(os.path.join(f_img_altered, vid, f), altered_cropped)

                    count += 1
        
        print(count)

        all += count

    print('all: ', all)

    print('all: ', all)

def extract_face_datasets(output):

    extract_face_videos(output)

if __name__ == '__main__':
    extract_face_datasets(opt.output)