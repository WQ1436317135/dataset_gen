import sys
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import glob
import numpy as np
import os
import cv2
import alphabets
from math import *
import argparse
import time
import dataset
import dataset_unit
import alphabet
import datetime


str_alphabet = alphabet.alphabet

alphabet_str = alphabets.alphabet


def str_choice_from_info_str(language="chinese", type_long=0):  # 随机生成长度不一的随机带有空格的字符串 语言自动设置为中文
    if language == "chinese":  # 文字选择
        info_str = info_chinese
    else:
        info_str = info_english

    if type_long == 0:
        quantity = random.randint(5, 15)  # 随机选取文字长度
        if random.randint(0, 10) < 2:
            start = random.randint(0, len(info_str) - 21)  # 随机选取文字的起始位置
            end = start + quantity  # 根据开始位置获得结束位置
            random_word = info_str[start:end]  # 获得代显示在图片中的字符串

        else:
            random_word = ""
            for i in range(quantity):
                random_word = random_word + random.choice(str_alphabet)

        block_flag = random.randint(1, 10)  # 设置出现空格的概率
        space_flag = []

        if block_flag <= 3:  # 有30%的概率不产生空格
            return random_word

        else:
            block_num = random.randint(1, 3)  # 随机设置空格产生的个数
            for i in range(block_num):
                space_flag.append(random.randint(1, quantity - 1))  # 随机设置空格产生的位置
            space_flag = list(set(space_flag))  # 删除产生空格位置中重复的位置

            space_flag_np = np.asarray(space_flag)
            space_num_sum = 0
            for j in range(len(space_flag_np)):  # 按照位置随机插入随机长度的空格
                space_num = random.randint(0, 2)  # 设置该位置产生的空格个数

                space_set_flag = space_flag_np[j] + space_num_sum
                random_word = random_word[0:space_set_flag] + " " * space_num + random_word[
                                                                                space_set_flag: len(random_word)]
                # 插入空格
                space_num_sum = space_num_sum + space_num
            return random_word

    else:
        quantity = random.randint(2, 4)  # 随机选取文字长度
        start = random.randint(0, len(info_str) - 21)  # 随机选取文字的起始位置
        end = start + quantity  # 根据开始位置获得结束位置
        random_word = info_str[start:end]  # 获得代显示在图片中的字符串
        return random_word


def random_font_size(language, str):  # 随机设置字体大小 语言默认设置为中文
    font_size = 0
    if language == "english":
        if len(str) > 20:

            font_size = random.randint(10, 30)
        elif len(str) > 16:

            font_size = random.randint(30, 70)
        elif len(str) > 12:

            font_size = random.randint(50, 80)
        elif len(str) > 6:

            font_size = random.randint(70, 120)
        elif len(str) > 3:

            font_size = random.randint(120, 300)

        else:
            font_size = random.randint(300, 600)

    if language == "chinese":
        if len(str) > 20:
            font_size = random.randint(30, 50)
        elif len(str) > 16:

            font_size = random.randint(30, 50)
        elif len(str) > 12:

            font_size = random.randint(50, 60)
        elif len(str) > 7:

            font_size = random.randint(50, 80)
        else:

            font_size = random.randint(60, 100)

    return font_size


def random_font(language="chinese"):
    if language == "chinese":
        font_path = './font/Chinese/'
    else:
        font_path = './font/English/'
    font_list = os.listdir(font_path)
    random_font = random.choice(font_list)
    return font_path + random_font


def random_word_color(px):
    flag = random.randint(0, 9)
    if flag < 0:
        font_color_choice = [[50, 10, 10], [20, 50, 20], [10, 50, 10], [20, 20, 50], [40, 40, 40]]  # 较为常规的文字颜色
        font_color = random.choice(font_color_choice)

        x = random.randint(0, 2)

        px[x] = random.randint(0, 20)

        noise = np.array([random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)])

        font_color = np.array(font_color) + noise + px

        font_color[font_color > 255] = 255

    elif flag < 9:
        font_color_choice = [[10, 10, 10], [20, 20, 20], [10, 10, 10], [20, 20, 20], [40, 40, 40]]  # 较为常规的文字颜色
        font_color = random.choice(font_color_choice)

        # print(px)
        px = 255 - px

        x = random.randint(0, 2)

        px[x] = 0
        # print(px)
        noise = np.array([random.randint(-10, 10), random.randint(-10, 10), random.randint(-10, 10)])

        font_color = np.array(font_color) + noise + px

        font_color[font_color > 255] = 255
    else:
        font_color = np.array([random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)])
        font_color = 255 - px
    font_color = font_color.tolist()
    return tuple(font_color)


def random_background(bground_path):  # 随机选取背景图片
    bground_list = os.listdir(bground_path)
    bground_choice = random.choice(bground_list)
    # print(bground_choice)
    bground = cv2.imread(bground_path + bground_choice, cv2.IMREAD_COLOR)

    if random.randint(0, 9) < 7:
        perspective = np.eye(3, dtype=np.float32) + np.random.uniform(-0.001, 0.001, (3, 3))
        perspective[2][2] = 1.0
        bground = cv2.warpPerspective(bground, perspective, (bground.shape[1], bground.shape[0]),
                                      borderValue=(255, 255, 255))

    h, w = bground.shape[0:2]

    if min(h, w) < 1024:
        if h < w:
            scale = 1024 / h
            bground = cv2.resize(bground, (int(w * scale), 1024), cv2.INTER_AREA)
        else:
            scale = 1024 / w
            bground = cv2.resize(bground, (1024, int(h * scale)), cv2.INTER_AREA)
    h, w = bground.shape[0:2]
    x_start = random.randint(0, w - 1024)
    y_start = random.randint(0, h - 1024)
    x_end = x_start + 1024
    y_end = y_start + 1024

    bground = bground[y_start:y_end, x_start:x_end]  # 剪裁图片至512*512

    return bground


def random_x_y(background_w, background_h, font_size, str):
    width, height = background_w, background_h

    a = width - font_size * (len(str))
    if a < 0:
        a = 100
    x = random.randint(0, a)
    y = random.randint(0, max(0, height - font_size * 5))

    return x, y


def coordinate_trans(start_x_np, start_y_np, height, width, degree, w_change, h_change, scale_set):
    #   坐标转换
    # 程序用于处理旋转后的坐标的变化
    x = start_x_np
    y = height - start_y_np
    cX = width // 2
    cY = height - height // 2
    new_x = (x - cX) * cos(pi / 180.0 * degree) - (y - cY) * sin(pi / 180.0 * degree) + cX
    new_y = (x - cX) * sin(pi / 180.0 * degree) + (y - cY) * cos(pi / 180.0 * degree) + cY
    new_x = new_x
    new_y = height - new_y
    new_x = ((new_x + w_change) * scale_set).astype(int)
    new_y = ((new_y + h_change) * scale_set).astype(int)
    return new_x, new_y


def creat_str_pic(language, str, background):
    background_w, background_h = background.shape[0:2]
    im = Image.new("RGB", (background_w, background_h))  # 生成空白图像
    font_size = random_font_size(language, str)
    # 随机选取字体
    font_name = random_font(language)
    # print(font_name)

    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(im)
    draw_x, draw_y = random_x_y(background_w, background_h, font_size, str)
    px = background[draw_x, draw_y]

    # 随机选取字体颜色
    font_color = random_word_color(px)

    draw.text((draw_x, draw_y), str, fill=font_color, font=font)  # 绘图
    # draw.text((draw_x, draw_y), str, font=font)
    start_x = []
    start_y = []
    end_x = []
    end_y = []
    for i in range(1, len(str) + 1):
        if i == 1:
            offsetx, offsety = font.getoffset(str[0:i])  # 获得文字的offset位置
            width, height = font.getsize(str[0:i])  # 获得文件的大小
            start_x.append(offsetx + draw_x)
            start_y.append(offsety + draw_y)
            end_x.append(width + draw_x)
            end_y.append(height + draw_y)

        else:
            offsetx, offsety = font.getoffset(str[0:i])  # 获得文字的offset位置
            width, height = font.getsize(str[0:i])  # 获得文件的大小
            start_x.append(end_x[-1])
            start_y.append(offsety + draw_y)
            end_x.append(width + draw_x)
            end_y.append(height + draw_y)

            start_x_rec = offsetx + draw_x
            start_y_rec = offsety + draw_y
            end_x_rec = width + draw_x
            end_y_rec = height + draw_y

    x1_long = np.asarray(start_x_rec)
    y1_long = np.asarray(start_y_rec)
    x2_long = np.asarray(end_x_rec)
    y2_long = np.asarray(start_y_rec)
    x3_long = np.asarray(end_x_rec)
    y3_long = np.asarray(end_y_rec)
    x4_long = np.asarray(start_x_rec)
    y4_long = np.asarray(end_y_rec)

    start_x_np = np.asarray(start_x)
    start_y_np = np.asarray(start_y)
    end_x_np = np.asarray(end_x)
    end_y_np = np.asarray(end_y)

    # print(len(start_x_np))

    x1 = start_x_np
    y1 = start_y_np
    x2 = end_x_np
    y2 = start_y_np
    x3 = end_x_np
    y3 = end_y_np
    x4 = start_x_np
    y4 = end_y_np

    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    height_im, width_im = im.shape[:2]
    # print(height_im, width_im)

    degree = random.randint(-90, 90)
    # 旋转后的尺寸
    heightNew = int(width_im * fabs(sin(radians(degree))) + height_im * fabs(cos(radians(degree))))
    widthNew = int(height_im * fabs(sin(radians(degree))) + width_im * fabs(cos(radians(degree))))

    # print(heightNew, widthNew)

    matRotation = cv2.getRotationMatrix2D((width_im / 2, height_im / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width_im) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height_im) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(im, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))

    w_change = (widthNew - width_im) // 2
    h_change = (heightNew - height_im) // 2

    scale_set = height_im / heightNew
    x1_new, y1_new = coordinate_trans(x1, y1, height_im, width_im, degree, w_change, h_change, scale_set)
    x2_new, y2_new = coordinate_trans(x2, y2, height_im, width_im, degree, w_change, h_change, scale_set)
    x3_new, y3_new = coordinate_trans(x3, y3, height_im, width_im, degree, w_change, h_change, scale_set)
    x4_new, y4_new = coordinate_trans(x4, y4, height_im, width_im, degree, w_change, h_change, scale_set)

    x1_long, y1_long = coordinate_trans(x1_long, y1_long, height_im, width_im, degree, w_change, h_change, scale_set)
    x2_long, y2_long = coordinate_trans(x2_long, y2_long, height_im, width_im, degree, w_change, h_change, scale_set)
    x3_long, y3_long = coordinate_trans(x3_long, y3_long, height_im, width_im, degree, w_change, h_change, scale_set)
    x4_long, y4_long = coordinate_trans(x4_long, y4_long, height_im, width_im, degree, w_change, h_change, scale_set)

    box_rec = np.asarray([[x1_long, y1_long], [x2_long, y2_long], [x3_long, y3_long], [x4_long, y4_long]])
    # print(box_rec)

    imgRotation = cv2.resize(imgRotation, (width_im, height_im), cv2.INTER_AREA)

    boxes = []
    for i in range(0, len(x1_new)):
        box = np.asarray(
            [[x1_new[i], y1_new[i]], [x2_new[i], y2_new[i]], [x3_new[i], y3_new[i]], [x4_new[i], y4_new[i]]])
        # print(box)
        boxes.append(box)

    boxes = np.asarray(boxes)
    return imgRotation, boxes, box_rec


def cross(x1, y1, x2, y2, x3, y3):  # 跨立实验
    x_1 = x2 - x1
    y_1 = y2 - y1
    x_2 = x3 - x1
    y_2 = y3 - y1
    return float(x_1 * y_2 - x_2 * y_1)


def line_cross(x1, y1, x2, y2, x3, y3, x4, y4):  # 判断两线段是否相交

    # 快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if (max(x1, x2) >= min(x3, x4)
            and max(x3, x4) >= min(x1, x2)
            and max(y1, y2) >= min(y3, y4)
            and max(y3, y4) >= min(y1, y2)):

        if (cross(x1, y1, x2, y2, x3, y3) * cross(x1, y1, x2, y2, x4, y4) <= 0
                and cross(x3, y3, x4, y4, x1, y1) * cross(x3, y3, x4, y4, x2, y2) <= 0):
            d = 1
        else:
            d = 0
    else:
        d = 0
    return d


def rec_cross(box1, box2):
    [[x1_1, y1_1], [x2_1, y2_1], [x3_1, y3_1], [x4_1, y4_1]] = box1
    cross_flag = 1
    for i in range(4):
        cross_flag = 0
        if line_cross(x1_1, y1_1, x2_1, y2_1, box2[i][0], box2[i][1], box2[(i + 1) % 4][0], box2[(i + 1) % 4][1]) \
                or line_cross(x2_1, y2_1, x3_1, y3_1, box2[i][0], box2[i][1], box2[(i + 1) % 4][0],
                              box2[(i + 1) % 4][1]) \
                or line_cross(x3_1, y3_1, x4_1, y4_1, box2[i][0], box2[i][1], box2[(i + 1) % 4][0],
                              box2[(i + 1) % 4][1]) \
                or line_cross(x4_1, y4_1, x1_1, y1_1, box2[i][0], box2[i][1], box2[(i + 1) % 4][0],
                              box2[(i + 1) % 4][1]):
            cross_flag = 1
            break

    if cross_flag == 0:
        for i in range(4):
            dist = cv2.pointPolygonTest(box1, (box2[i][0], box2[i][1]), False)
            if dist != -1:
                cross_flag = 1
                break

        for i in range(4):
            dist = cv2.pointPolygonTest(box2, (box1[i][0], box1[i][1]), False)
            if dist != -1:
                cross_flag = 1
                break

    return cross_flag


def img_show(pic, time):
    cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('input_image', pic)
    cv2.waitKey(time)


def pic_mix(imgRotation, background):
    img2gray = cv2.cvtColor(imgRotation, cv2.COLOR_BGR2GRAY)
    # img_show(imgRotation, 0)
    # img_show(img2gray, 0)
    # mask = cv2.adaptiveThreshold(img2gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,5)
    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask)

    background = cv2.bitwise_and(background, background, mask=mask_inv)
    background = cv2.add(background, imgRotation)  # 进行融合

    return background


def main(num):
    background = random_background('./background/')

    str_num = random.randint(2, 7)
    # print(str_num)
    # background = background - 5 * str_num
    h, w = background.shape[0:2]
    rec_boxse = []
    str1 = ""
    for i in range(str_num):
        if random.randint(0, 9) < 0:
            language = "chinese"
        else:
            language = "english"

        if str_num < 7 and i == 1:
            random_word = str_choice_from_info_str(language, type_long=1)

        else:
            random_word = str_choice_from_info_str(language)

        # print(random_word)
        imgRotation, boxes, box_rec = creat_str_pic(language, random_word, background)

        cross_flag = 1
        cross_time = 0
        while (cross_flag and cross_time < 1000):
            imgRotation, boxes, box_rec = creat_str_pic(language, random_word, background)
            cross_flag = 0
            for j in range(len(rec_boxse)):
                # print(rec_boxse[j])
                # print(rec_cross(rec_boxse[j], box_rec))
                if rec_cross(rec_boxse[j], box_rec):
                    cross_flag = 1
                    cross_time += 1
                    continue

        # print(str_num, i)
        if cross_flag == 0:
            rec_boxse.append(box_rec)

            if background.mean() < 120:
                flag = 1
            else:
                flag = 0

            if flag == 1:
                background = cv2.addWeighted(background, 1, imgRotation, 1, 0)
            else:
                background = pic_mix(imgRotation, background)

            for k in range(0, len(boxes)):
                if random_word[k] == " ":
                    continue
                str1 += random_word[k]

            str1 += "\n"


    if flag == 0:
        kernel = np.ones((1, 2), np.uint8)
        background = cv2.erode(background, kernel, iterations=1)
        background = cv2.dilate(background, kernel, iterations=1)
        # background = cv2.erode(background, kernel, iterations=1)
        # background = cv2.dilate(background, kernel, iterations=1)
    # img_show(background, 0)

    cv2.imwrite('./data_set2/img/img_{}.jpg'.format(num), background)
    with open("./data_set2/txt/img_{}.txt".format(num), "w", encoding="utf-8") as f:
        f.write(str1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='th_num')
    parser.add_argument('--th_num', nargs='?', type=int, default=0,
                        help='第几个处理线程')

    args = parser.parse_args()
    # 处理具有工商信息语义信息的语料库，去除空格等不必要符

    with open('info_chinese.txt', 'r', encoding='utf-8-sig') as file_chinese:
        info_list = [part.strip().replace(' ', '') for part in file_chinese.readlines()]
        info_chinese = ''.join(info_list)

    with open('info_english.txt', 'r', encoding='utf-8-sig') as file_english:
        info_list = [part.strip().replace(' ', '') for part in file_english.readlines()]
        info_english = ''.join(info_list)

    total = 90000
    prev_time = time.time()
    for i in range(total):
        main(i)

        batches_done = i
        batches_left = total - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        sys.stdout.write(
            "\r[all %d/%d] ETA: %s"
            % (
                i,
                total,
                time_left,
            )
        )


