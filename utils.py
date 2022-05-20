import cv2
import numpy as np
import torch
from torchvision import transforms
import math
import os

trans = transforms.Compose([
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])


JA, MO, BA = 0, 1, 2 # 자음
BBOX_X, BBOX_Y, BBOX_W, BBOX_H = 0, 1, 2, 3

# input image preprocessing
def imgproc(img,mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    target_h, target_w, channel = img.shape

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)

    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = img

    # should be RGB order
    img = resized.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img



def getContourBoxes(image, X):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, thres = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY_INV)
    contours, hr = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x_arr = []
    y_arr = []

    for ctr in contours:
        x, y, w, h = cv2.boundingRect(ctr)
        x_arr.append(x)
        x_arr.append(x+w)
        y_arr.append(y)
        y_arr.append(y+h)

    x = X + min(x_arr)
    w = max(x_arr) - min(x_arr)
    y = min(y_arr)
    h = max(y_arr) - min(y_arr)

    box = [x, y, w, h, y + (h//2)]

    return box

def crop_img(image, size, width, height, x,y=0):
    img = image[y:y+height, x:x + width].copy()
    box = getContourBoxes(img, x)
    crop = np.ones((size, size, 3), dtype=np.uint8) * 255
    crop[0:height, 0:width] = img
    crop = cv2.resize(crop, (224, 224))

    return crop, box
#
# def check_bbox(bbox):
#     print('\ncheck bbox')
#     bbox = np.array(bbox, dtype=np.int32)
#     arr = np.sort(bbox,axis=0)[:-(len(bbox) // 3),:]
#     m = arr.sum(axis=0) // len(arr)
#     threshold = m[2] * m[3]
#     print(f'm {m} thres {threshold}')
#     new_bbox = []
#     for i in range(0, len(bbox)):
#         det_err = False
#         if i == len(bbox)-1:
#             new_bbox.append(bbox[i])
#             break
#         x, y, w, h, _ = bbox[i]
#         print(f'size {w*h}')
#         if w * h < threshold:
#             x_next, y_next, w_next, h_next, _ = bbox[i+1]
#             if w_next * h_next < threshold:
#                 det_err = True
#
#         if det_err:
#             x_new = x
#             w_new = x_next + w_next - x_new
#             y_new = min(y, y_next)
#             h_new = max(y+h, y_next+h_next) - y_new
#             new_box = [x_new, y_new, w_new, h_new, y_new + h_new//2 ]
#             new_bbox.append(new_box)
#             i += 1 # skip next
#         else:
#             new_bbox.append(bbox[i])
#
#     return new_bbox





##

def getDetBoxes(textmap, text_threshold=0.7, low_text=0.4):
    # prepare data
    textmap = textmap.copy()

    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score.astype(np.uint8), connectivity=4)
    det = []

    for k in range(1,nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue

        # thresholding
        if np.max(textmap[labels==k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1

        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        x, y, w, h = cv2.boundingRect(np_contours)
        box = [x, y, w, h, y+((h//2))]

        det.append(box)
    det.sort(key=lambda x: x[0])
    det = np.array(det, dtype=np.int32)
    det *= 2          # ratio_net = 2

    return det

# segmentation preprocessing
def seg_trans(img_list, threshold=170):
    x = np.where(img_list<threshold, 1, 0).astype(np.float32)
    x = trans(x)
    return x

def check_ja(char_list, char):
    '''
    받침이 없을 경우) 모음 중심보다 오른쪽 혹은 아래에 있거나, 자음 컨투어엔 속하지않지만 모음엔 속할 경우 모음으로 판단한다.
    받침이 있을 경우) 자음 컨투어에 속하지않고 + 모음의 중심보다 아래이면 받침으로 판단하고,
                  그렇지 않으며 모음 중심보다 오른쪽 혹은 모음과 같은 컨투어이면 모음으로 판단한다.
    '''
    # print(f'check_ja - char{char}')
    ja, mo, ba = char_list
    x, y, w, h = char['box']
    j_x, j_y, j_w, j_h = ja['box']
    m_x, m_y, m_w, m_h = mo['box']

    if -1 in ba['label']:
        if x > m_x + m_w//2 or y + h//2 > (m_y + m_h): #or (char['label'] not in ja['label'] and char['label'] in mo['label']):
            return MO
        elif x > j_x + j_w:
            return MO
        elif y+h//2 > j_y+j_h and x+w//2 > j_x+j_w:
            return MO
    else:
        if char['label'] not in ja['label']:
            if y > (m_y + m_h//2):
                return BA
            elif x > m_x + m_w//2: #or char['label'] in mo['label']:
                return MO

    return JA

def check_mo(char_list, char):
    '''
    받침이 없을 경우) 자음보다 왼쪽이거나 모음보다 높고, 모음의 오른쪽이 아닐 때 경우 자음으로 판단한다.
    받침이 있을 경우) 모음보다 아래에 있고 모음과 같은 컨투어가 아니거나 자음과 같은 컨투어일 경우 받침으로 판단한다.
                  만일 그렇지않고 자음보다 왼쪽이거나 모음보다 높고 모음의 오른쪼깅 아닐 경우 자음으로 판단한다.
    '''
    # print(f'check_mo - char{char}')

    ja, mo, ba = char_list
    x, y, w, h = char['box']
    j_x, j_y, j_w, j_h = ja['box']
    m_x, m_y, m_w, m_h = mo['box']
    #
    # if y+h < j_y+j_h//2:
    #     print('case1')
    #     return JA
    if y + h//2 < j_y + j_h//2 and x< j_x:
        # print('case0')
        return JA
    if char['label'] not in mo['label'] and y + h < j_y:
        return JA
    if char['label'] in ja['label'] and (x+w < j_x+j_w or y+h < m_y):
        # print('case1')
        return JA

    if x + w < j_x + j_w and (y + h//2 < m_y + m_h//2 or y < j_y + j_h//2) :
        # print('case2')
        return JA
    if y+h//2 > (m_y + m_h) and y+h//2 > j_y+j_h:
            # print('case3')
            return BA

        # elif x < j_x and y :
        #     print('case1')
        #     return JA
        # elif y + h < m_y + m_h and x + w < m_x + m_w:
        #     print('case2')
        #     return JA

    return MO

def check_ba(char_list, char):
    '''
    받침이 없는 경우 -1을 리턴한다.
    자음이나 모음보다 높을 경우, 모음보다 왼쪽에 있으면 자음으로, 그렇지 않을 경우 모음으로 판단한다.
    만일 받침과 모음에 속하지않고 자음에만 속할 경우 자음으로 판단한다
    '''
    # print(f'check_ba - char{char}')

    if char['label'] == -1: return -1
    ja, mo, ba = char_list
    x, y, w, h = char['box']
    j_x, j_y, j_w, j_h = ja['box']
    m_x, m_y, m_w, m_h = mo['box']

    if y + h//2 < j_y+j_h or y + h//2 < m_y + m_h:
        if char['label'] in ja['label'] and char['label'] not in mo['label']:
            # print('case1')
            return JA
        elif (char['label'] in mo['label'] and char['label'] not in ja['label']) or y > j_y + j_h:
            # print('case2')
            return MO
        if x+w < m_x:
            # print('case3')
            return JA
        else:
            return MO

    return BA

def concat_bbox(character, ch):
    labels = character['label']
    labels.append(ch['label'])

    c_x, c_y, c_w, c_h = character['box']
    x, y, w, h = ch['box']
    x1 = min(c_x,x)
    x2 = max(c_x+c_w, x+w)
    y1 = min(c_y, y)
    y2 = max(c_y+c_h, y+h)
    new_box = [x1, y1, x2-x1, y2-y1]

    return labels, new_box


def getDetBoxes_bbox(image, char_score):
    # get segmap
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    e = cv2.erode(img_gray, kernel, iterations=3)
    _, img_labels, _, _ = cv2.connectedComponentsWithStats(e.astype(np.uint8), connectivity=4)
    nLabels, seg_labels, stats, _ = cv2.connectedComponentsWithStats(char_score.astype(np.uint8), connectivity=4)

    boxes = []
    main_bbox = [0, 0]
    idx = 0
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 300: continue
        if main_bbox[1] < size:
            main_bbox = [idx, size]
        # make segmentation map
        charmap = np.zeros(char_score.shape, dtype=np.uint8)
        charmap[seg_labels == k] = 255
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        box = [x, y, w, h]
        # get label in contour
        vals, counts = np.unique(img_labels[seg_labels == k], return_counts=True)
        index = np.argmax(counts)
        label = vals[index]

        bbox = {'label':label, 'box':box}
        boxes.append(bbox)
        idx += 1
    if not boxes:
        return {'label':[-1],'box':[-1, -1, -1, -1]}, boxes
    else:
        box = boxes[main_bbox[0]]
        return {'label':[box['label']], 'box':box['box']}, boxes

def getDetBoxes_from_seg(image, y, char_threshold=0.5):
    det = []
    characters = []
    boxes = [] # [자음, 모음, 받침]의 모든 bbox
    boxes_for_test = [[],[],[]]
    check = [check_ja, check_mo, check_ba]
    bbox_none = {'label': [-1], 'box':[-1,-1,-1,-1]}
    # make bounding box
    for textmap in y:
        ret, char_score = cv2.threshold(textmap, char_threshold, 1, 0)
        main_box, box = getDetBoxes_bbox(image, char_score)
        characters.append(main_box)
        boxes.append(box)

    # check bounding box
    char_ba = {'label':characters[BA]['label'][0], 'box':characters[BA]['box']}
    char_num = check[BA](characters, char_ba)
    # print(f'char # {char_num}')
    if char_num != BA:
        characters[BA] = bbox_none
    # print(f'characters: {characters}')
    for i in range(3):
        # print(f'i: {i} ch: {len(boxes[i])}')
        # bbox가 없는 경우
        for ch in boxes[i]:
            if ch['label'] == -1: continue
            char_num = check[i](characters,ch)
            # print(f'char # {char_num}')
            if char_num != -1:
                boxes_for_test[char_num].append(ch['box'])
                character = characters[char_num]
                new_label, new_box = concat_bbox(character, ch)
                character['label'] = new_label
                character['box'] = new_box
    # err - 자음 혹은 모음 없음
    if characters[0]['label'] == [-1] or characters[1]['label'] == [-1]:
        # print('detect err')
        return np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]])
        # return [[{'label': -1, 'box':[-1,-1,-1,-1]}],[{'label': -1, 'box':[-1,-1,-1,-1]}],[{'label': -1, 'box':[-1,-1,-1,-1]}]]

    # print('\n')
    for c in characters:
        det.append(c['box'])

    det = np.array(det, dtype=np.int32)

    return det

def masks_to_colorimg(masks):
    colors = np.asarray([(201, 58, 64), (242, 207, 1), (0, 152, 75)])

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.5]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)

def show_img_arr(image):
    for r in image:
        for i in r:
            if i != 0:
                print(i,end='')
            else:
                print(' ',end='')
        print()