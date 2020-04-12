import cv2
import numpy as np
from math import ceil
import os
from os import path
from functools import cmp_to_key
import sys
import tempfile
import argparse
import qmg

OUTPUT_DIR = 'E:\\Image\\Manga\\'
MIN_WHITE = 245

class Rect(object):
    def __init__(self, center, size):
        self.width, self.height = size
        self.width += 2
        self.height += 2
        self.size = (self.width, self.height)
        w2, h2 = self.width/2, self.height/2
        self.cx, self.cy = center
        self.center = (self.cx, self.cy)
        self.sx, self.sy = self.cx-w2, self.cy-h2
        self.ex, self.ey = self.cx+w2, self.cy+h2
        self.s = (self.sx, self.sy)
        self.e = (self.ex, self.ey)


# ref: https://gist.github.com/atarabi/6a230bc8b3f7983fe596
def threshold(image, radius=15, C=5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 2 * radius + 1, C)

def find_external_contours(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    external_num = hierarchy.shape[1] if hierarchy is not None else 0
    return contours[0:external_num]

def extract_rects_from_controus(contours, min_perimeter, max_perimeter):
    frames = []
    for contour in contours:
        frame = cv2.minAreaRect(contour)
        center, size, angle = frame
        # 縦・横が逆になっている場合、90度回転させる
        if angle < -45:
            size = tuple(reversed(size))
            angle = angle + 90
        w, h = size
        perimeter = 2 * (w + h)
        if min_perimeter < perimeter < max_perimeter and abs(angle) < 3.0 and 0.1 <= min(w, h) / max(w, h) <= 1.0:
            # frames.append((center, (w + 2, h + 2), angle))  # パディングを加える
            p = Rect(center, size)
            # p = (center, (center[0]-w2, center[1]-h2), (center[0]+w2, center[1]+h2), (w, h))
            frames.append(p)
    return frames

def cmp_frame(tolerance):
    def _cmp(lhs, rhs):
        return (lhs > rhs) - (lhs < rhs)

    def _cmp_frame(lhs, rhs):
        if lhs.center == rhs.center:
            return 0
        x1, y1 = lhs.center
        x2, y2 = rhs.center
        if abs(x1 - x2) < tolerance:
            return _cmp(y1, y2)
        else:
            return _cmp(x2, x1)

    return _cmp_frame

def imread(path):
    return cv2.imdecode(np.fromfile(path,dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite(img, path):
    cv2.imencode('.png',img)[1].tofile(path)

def cut_koma(img):
    # 剪开跨页
    height, width, _ = img.shape
    if width > height:
        center = int(width/2)
        average = np.average(img[0:height, center])
        if average > MIN_WHITE:
            return _cut_koma(img[:, center:]) + _cut_koma(img[:, :center])
        else:
            return [img]
    else:
        return _cut_koma(img)

UP = 1
DOWN = 2
RIGHT = 4
LEFT = 8

def split_image(img, center, direction):
    # 根据方向指示和中点分割图片
    center_x, center_y = np.int0(center)
    if not direction:
        frames = [img]
    elif direction == DOWN|LEFT|RIGHT:
        # 正品
        frames = [
            img[:center_y, :],
            img[center_y:, center_x:],
            img[center_y:, :center_x]
        ]
    elif direction == UP|RIGHT|LEFT:
        # 倒品
        frames = [
            img[:center_y, center_x:],
            img[:center_y, :center_x],
            img[center_y:, :]
        ]
    elif direction == UP|DOWN|RIGHT|LEFT:
        # 双四格
        frames = [
            img[:center_y, center_x:],
            img[center_y:, center_x:],
            img[:center_y, :center_x],
            img[center_y:, :center_x]
        ]
    elif direction == LEFT|UP|DOWN:
        # 左四格
        frames = [
            img[:, center_x:],
            img[:center_y, :center_x],
            img[center_y:, :center_x]
        ]
    elif direction == RIGHT|UP|DOWN:
        # 右四格
        frames = [
            img[:center_y, center_x:],
            img[center_y:, center_x:],
            img[:, :center_x]
        ]
    else:
        frames = [img]
    return frames


def _cut_koma(img):
    height, width, _ = img.shape

    ch, cw = height/2, width/2
    center_x, center_y = np.int0((cw, ch))
    direction = 0

    error_height = height*0.05
    error_width = width*0.05
    min_center_x = cw - error_width
    max_center_x = cw + error_width
    min_center_y = ch - error_height
    max_center_y = ch + error_height

    min_center_x, min_center_y, max_center_x, max_center_y = \
        np.int0((min_center_x, min_center_y, max_center_x, max_center_y))
        
    im = np.mean(img, axis=2)
    mean_up = np.mean(im[:min_center_y, min_center_x:max_center_x+1], axis=0)
    mean_down = np.mean(im[max_center_y:, min_center_x:max_center_x+1], axis=0)
    mean_updown = np.mean([mean_up, mean_down], axis=0)
    mean_left = np.mean(im[min_center_y:max_center_y+1, :min_center_x], axis=1)
    mean_right = np.mean(im[min_center_y:max_center_y+1, max_center_x:], axis=1)
    mean_lr = np.mean([mean_left, mean_right], axis=0)

    max_updown_index = np.argmax(mean_updown)
    max_updown = mean_updown[max_updown_index]
    max_lr_index = np.argmax(mean_lr)
    max_lr = mean_lr[max_lr_index]

    if max_lr > MIN_WHITE:
        center_y = min_center_y + max_lr_index
        direction |= RIGHT|LEFT
    else:
        max_right_index = np.argmax(mean_right)
        max_right = mean_right[max_right_index]
        max_left_index = np.argmax(mean_left)
        max_left = mean_left[max_left_index]
        if max_left > MIN_WHITE:
            center_y = min_center_y + max_left_index
            direction |= LEFT
        elif max_right > MIN_WHITE:
            center_y = min_center_y + max_right_index
            direction |= RIGHT
    if max_updown > MIN_WHITE:
        center_x = min_center_x + max_updown_index
        direction |= UP|DOWN
    else:
        max_up_index = np.argmax(mean_up)
        max_up = mean_up[max_up_index]
        max_down_index = np.argmax(mean_down)
        max_down = mean_down[max_down_index]
        if max_up > MIN_WHITE:
            center_x = min_center_x + max_up_index
            direction |= UP
        elif max_down > MIN_WHITE:
            center_x = min_center_x + max_down_index
            direction |= DOWN

    return split_image(img, (center_x, center_y), direction)

def cut_dir(src, dst):
    for filepath, _, filenames in os.walk(src):
        for filename in filenames:
            srcpath = path.join(filepath, filename)
            new_file_path = path.join(dst, path.relpath(filepath, src))
            os.makedirs(new_file_path, exist_ok=True)
            print('\r\x1b[K'+srcpath, end='')
            img = imread(srcpath)
            if img is not None:
                frames = cut_koma(img)
                name = path.splitext(filename)[0]
                l = len(str(len(frames)))
                f = f'{{}}-{{:0{l}}}.png'
                for i, frame in enumerate(frames):
                    dstpath = path.join(new_file_path, f.format(name, i+1))
                    imwrite(frame, dstpath)
    print()


if __name__=='__main__':
# for file in ['0010.jpg']:
    parser = argparse.ArgumentParser(description='4koma splitter')
    parser.add_argument('srcs', nargs='+')
    parser.add_argument('-t', '--format', choices=['mobi', 'epub', 'cbz', 'raw'], default='mobi', help='archive format, if choice "raw" will not make archive')
    parser.add_argument('-s', '--save-path', dest='save_path', default=OUTPUT_DIR, help='save path')
    args = parser.parse_args()

    if args.format == 'raw':
        save_path = args.save_path
        if not path.exists(save_path):
            os.mkdir(save_path)
        td = None
    else:
        td = tempfile.TemporaryDirectory()
        save_path = td.name
    saves = []
    for src in args.srcs:
        d1, d2 = path.split(src)
        save = path.join(save_path, d2 if d2 else d1)
        saves.append(save)
        cut_dir(src, save)
    if args.format != 'raw':
        qmg.main(saves, args.save_path, args.format)
