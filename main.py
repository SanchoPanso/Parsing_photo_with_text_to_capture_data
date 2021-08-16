import pytesseract
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pytesseract import Output
import re
import math


class Box:
    def __init__(self, index):
        self.index = index


class DateBox:
    def __init__(self, index):
        self.index = index


class BornBox:
    def __init__(self, index):
        self.index = index
        self.date_box = None


class CodeBox:
    def __init__(self, index):
        self.index = index


class NumberBox:
    def __init__(self, index):
        self.index = index
        self.code_box = None
        self.born_box = None


def special_distance(x1, y1, x2, y2):
    """

    :param x1: left point
    :param y1: left point
    :param x2: right point
    :param y2: right point
    :return:
    """
    xd = x2 - x1
    yd = y2 - y1

    if xd + 10 < 0 or yd + 10 < 0:
        return -1
    else:
        return math.sqrt(xd ** 2 + yd ** 2)


def special_min(arr):
    minimum = 0
    index = 0
    for i in range(arr):
        if arr[i] != -1 and (minimum > arr[i] or minimum == 0):
            minimum = arr[i]
            index = i
    return minimum, index


def is_match(data, li, ri, conditions_keys):
    result = True
    for key in conditions_keys:
        cond = data['block_num'][li] == data['block_num'][ri]
        result = result and cond
    return result


def match_data(data, left_data_index_list, right_data_index_list):
    matched_right_data_index_list = [None] * len(left_data_index_list)

    for i in range(len(left_data_index_list)):
        li = left_data_index_list[i]
        for j in range(len(right_data_index_list)):
            ri = right_data_index_list[j]
            if is_match(data, li, ri, ['block_num']):
                matched_right_data_index_list[i] = ri

    return left_data_index_list, matched_right_data_index_list


def match_data_by_distance(data, left_data_index_list, right_data_index_list):
    matched_right_data_index_list = [None] * len(left_data_index_list)
    distances = [0] * len(left_data_index_list)

    for i in range(len(left_data_index_list)):
        li = left_data_index_list[i]

        minimum = 0
        index = 0
        for j in range(len(right_data_index_list)):
            ri = right_data_index_list[j]
            (x1, y1, x2, y2) = (data['left'][li.index], data['top'][li.index],
                                data['left'][ri.index], data['top'][ri.index])
            distance = special_distance(x1, y1, x2, y2)
            if distance != -1 and (minimum > distance or minimum == 0):
                minimum = distance
                index = ri

        matched_right_data_index_list[i] = index
        distances[i] = minimum

        if index in matched_right_data_index_list:
            for k in range(i):
                if index == matched_right_data_index_list[k]:
                    if minimum < distances[k]:
                        matched_right_data_index_list[k] = None
                    else:
                        matched_right_data_index_list[i] = None

    return matched_right_data_index_list


def find_splitted_data(d, number_pattern, born_pattern, code_pattern, date_pattern):
    n_boxes = len(d['text'])

    number_list = []
    born_list = []
    code_list = []
    date_list = []

    for i in range(n_boxes):
        if re.search(number_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            # cv2.putText(img, str(d['block_num'][i]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            number_instance = NumberBox(i)
            number_list.append(number_instance)

        if re.search(born_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
            # cv2.putText(img, str(d['block_num'][i]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            born_instance = BornBox(i)
            born_list.append(born_instance)

        if re.search(code_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
            # cv2.putText(img, str(d['line_num'][i]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            code_instance = CodeBox(i)
            code_list.append(code_instance)

        if re.search(date_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 4)
            date_instance = DateBox(i)
            date_list.append(date_instance)
    return number_list, born_list, code_list, date_list


def draw_text_bbox(img, d, index, color: tuple):
    (x, y, w, h) = (d['left'][index], d['top'][index], d['width'][index], d['height'][index])
    img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 4)


def draw_linking_line(img, d, index1, index2, color: tuple):
    (x1, y1, x2, y2) = (d['left'][index1] + d['width'][index1], d['top'][index1] + d['height'][index1],
                        d['left'][index2], d['top'][index2])
    cv2.line(img, (x1, y1), (x2, y2), color, 7)


def print_data(d, number_list):
    print("Собранные данные")
    for number_box in number_list:
        text = ""
        if number_box.code_box is None:
            text += f"Number: None, "
        else:
            text += f"Number: {d['text'][number_box.code_box.index]}, "

        if number_box.born_box.date_box is None:
            text += f"Born: None;"
        else:
            text += f"Born: {d['text'][number_box.born_box.date_box.index][1:]};"

        print(text)


def main():

    img = cv2.imread('example_fixed.jpg')
    pytesseract.pytesseract.tesseract_cmd = r"C:\ProgramFiles\Tesseract-OCR\tesseract.exe"

    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    print(d.keys())

    number_pattern = r'Number'
    born_pattern = 'Born'
    code_pattern = r'\d{9}'
    date_pattern = r'\d{2}[.]\d{2}[.]\d{4}'

    number_list, born_list, code_list, date_list = find_splitted_data(d,
                                                                      number_pattern,
                                                                      born_pattern,
                                                                      code_pattern,
                                                                      date_pattern)

    date_list = match_data_by_distance(d, born_list, date_list)
    for i in range(len(date_list)):
        born_list[i].date_box = date_list[i]

    code_list = match_data_by_distance(d, number_list, code_list)
    for i in range(len(code_list)):
        number_list[i].code_box = code_list[i]

    born_list = match_data_by_distance(d, number_list, born_list)
    for i in range(len(born_list)):
        number_list[i].born_box = born_list[i]

    for number_box in number_list:
        draw_text_bbox(img, d, number_box.index, (0, 255,255))

        born_box = number_box.born_box
        if born_box != None:
            draw_text_bbox(img, d, born_box.index, (0, 255, 255))

            date_box = born_box.date_box
            if date_box != None:
                draw_text_bbox(img, d, date_box.index, (0, 255, 0))

        code_box = number_box.code_box
        if code_box != None:
            draw_text_bbox(img, d, code_box.index, (0, 255, 0))

    img = cv2.resize(img, (640, 640))

    cv2.imwrite("example_with_highlighted_data.jpg", img)

    print_data(d, number_list)


def bboxes():
    img = cv2.imread('example_fixed.jpg')
    # img = cv2.resize(img, (640, 640))

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    print(d.keys())

    number_pattern = r'Number'
    born_pattern = 'Born'
    code_pattern = r'\d{9}'
    date_pattern = r'\d{2}[.]\d{2}[.]\d{4}'

    number_list = []
    born_list = []
    code_list = []
    date_list = []

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if re.search(number_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            # cv2.putText(img, str(d['block_num'][i]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            number_instance = NumberBox(i)
            number_list.append(number_instance)

        if re.search(born_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 4)
            # cv2.putText(img, str(d['block_num'][i]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            born_instance = BornBox(i)
            born_list.append(born_instance)

        if re.search(code_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)
            # cv2.putText(img, str(d['line_num'][i]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 3)
            code_instance = CodeBox(i)
            code_list.append(code_instance)

        if re.search(date_pattern, d['text'][i]):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 4)
            date_instance = DateBox(i)
            date_list.append(date_instance)

    date_list = match_data_by_distance(d, born_list, date_list)
    for i in range(len(date_list)):
        born_list[i].date_box = date_list[i]

    code_list = match_data_by_distance(d, number_list, code_list)
    for i in range(len(code_list)):
        number_list[i].code_box = code_list[i]

    born_list = match_data_by_distance(d, number_list, born_list)
    for i in range(len(born_list)):
        number_list[i].born_box = born_list[i]

    for number_box in number_list:
        born_box = number_box.born_box
        if born_box != None:
            (x1, y1, x2, y2) = (d['left'][number_box.index], d['top'][number_box.index],
                                d['left'][born_box.index], d['top'][born_box.index])
            cv2.line(img, (x1, y1), (x2, y2), (0, 200, 200), 7)
            date_box = born_box.date_box
            if date_box != None:
                (x1, y1, x2, y2) = (d['left'][born_box.index], d['top'][born_box.index],
                                    d['left'][date_box.index], d['top'][date_box.index])
                cv2.line(img, (x1, y1), (x2, y2), (255, 200, 100), 7)
        code_box = number_box.code_box
        if code_box != None:
            (x1, y1, x2, y2) = (d['left'][number_box.index], d['top'][number_box.index],
                                d['left'][code_box.index], d['top'][code_box.index])
            cv2.line(img, (x1, y1), (x2, y2), (255, 200, 100), 7)

    img = cv2.resize(img, (640, 640))

    cv2.imshow('img', img)
    cv2.waitKey(0)



def find_rect():
    img = cv2.imread("example.jpg")

    lower = np.array([85, 30, 10])
    upper = np.array([122, 235, 255])
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("aaaa", imgResult)
    cv2.waitKey(0)

    gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 30, 60)
    plt.imshow(edged)
    plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    # plt.imshow(closed)
    # plt.show()

    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    con_poly = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        con_poly.append(approx)
    poly = cv2.drawContours(img.copy(), con_poly, -1, (255, 0, 255), 8)

    con_poly_rect = []
    for c in con_poly:
        if 4 <= len(c) <= 7:
            con_poly_rect.append(c)

    plt.imshow(poly)
    plt.show()


if __name__ == '__main__':
    main()
    # os.system('heic-to-jpg -s example.heic --keep')
