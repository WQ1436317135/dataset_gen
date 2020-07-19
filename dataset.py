import numpy as np
import cv2
from PIL import Image
import math
import os
import torch
import torchvision.transforms as transforms
from torch.utils import data
import time
import imageio
import sys
from dataset_unit import find_min_rect_angle, get_rotate_mat, \
    rotate_vertices, get_boundary, rotate_all_pixels, shrink_poly
import dataset_unit


def imshow(img):
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.imshow("img2", img)
    cv2.waitKey(0)


def extract_vertices(lines, dict, nclass):
    labels = []
    vertices = []
    for line in lines:
        line = line.rstrip('\n').lstrip('\ufeff').split(',')
        vertices.append(list(map(int, line[:8])))
        # print(list((map(float, line.rstrip('\n').lstrip('\ufeff').split(',')[:8]))))
        label = line[-1]
        if label == "":
            label = ","
        labels.append(dataset_unit.str_Converter(label, dict, nclass))
    return np.array(vertices), np.array(labels)


def get_score_geo(img, vertices, labels, nclass, scale, length):
    conf_map = np.zeros((int(img.shape[0] * scale), int(img.shape[1] * scale), 1), np.float32)
    geo_map = np.zeros((int(img.shape[0] * scale), int(img.shape[1] * scale), 5), np.float32)
    tag_map = np.zeros((int(img.shape[0] * scale), int(img.shape[1] * scale), 1), np.float32)
    index = np.arange(0, length, int(1 / scale))
    index_x, index_y = np.meshgrid(index, index)
    polys = []

    for i, vertice in enumerate(vertices):

        # poly = np.around(scale * vertice.reshape((4,2))).astype(np.int32)  # scaled & shrinked
        poly = np.around(scale * shrink_poly(vertice).reshape((4,2))).astype(np.int32)  # scaled & shrinked
        polys.append(poly)
        temp_mask = np.zeros(conf_map.shape[:-1], np.float32)
        cv2.fillPoly(temp_mask, [poly], 1)

        cv2.fillPoly(tag_map, [poly], int(labels[i]))

        theta = find_min_rect_angle(vertice)
        rotate_mat = get_rotate_mat(theta)

        rotated_vertices = rotate_vertices(vertice, theta)
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)
        d1 = rotated_y - y_min
        d1[d1 < 0] = 0
        d2 = y_max - rotated_y
        d2[d2 < 0] = 0
        d3 = rotated_x - x_min
        d3[d3 < 0] = 0
        d4 = x_max - rotated_x
        d4[d4 < 0] = 0
        geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map[:, :, 4] += theta * temp_mask

        time_end = time.time()

    cv2.fillPoly(conf_map, polys, 1)
    # imshow(tag_map)

    h, w = tag_map.shape[:2]

    # tag_map = torch.LongTensor(tag_map)

    # one_hot = torch.zeros(h, w, nclass).scatter_(-1, tag_map, 1)
    # print(tag_map.shape)
    # print(one_hot)
    # print(one_hot.shape)
    # conf_map = torch.Tensor(conf_map)

    # geo_map = torch.Tensor(geo_map)
    # one_hot = one_hot.numpy()
    return geo_map, conf_map, tag_map


def resize_img(img, boxes, len_img=512):
    h, w = img.shape[:2]
    img = cv2.resize(img, (len_img, len_img))
    ratio_w = img.shape[1] / w
    ratio_h = img.shape[0] / h
    # print(boxes)
    if boxes.size > 0:
        boxes[:, [0, 2, 4, 6]] = boxes[:, [0, 2, 4, 6]] * ratio_w
        boxes[:, [1, 3, 5, 7]] = boxes[:, [1, 3, 5, 7]] * ratio_h
    # print(boxes)
    return img, boxes


class custom_dataset(data.Dataset):
    def __init__(self, img_path, gt_path, len_img=512):
        super(custom_dataset, self).__init__()
        self.img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
        self.gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
        self.len_img = len_img
        self.dict, self.nclass = dataset_unit.str_Converter_init()

        print(len(self.img_files))
        for i in range(len(self.img_files)):
            img_id = [os.path.basename(self.img_files[i]).strip('.JPG').strip('.jpg'),
                      os.path.basename(self.gt_files[i]).strip('.npy')]
            if img_id[0] == img_id[1]:

                continue
            else:
                print(img_id[0])
                print(img_id[1])
            sys.exit('img list and txt list is not matched')

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        transform = transforms.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5))
        img = cv2.imread(self.img_files[index], cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.len_img, self.len_img))


        gt_np = np.load(self.gt_files[index])
        geo_map = gt_np[:, :, 0:5]
        conf_map = gt_np[:, :, 5]
        conf_map = np.expand_dims(conf_map, axis=2)
        tag_map = gt_np[:, :, 6]
        tag_map = np.expand_dims(tag_map, axis=2)
        h_tag, w_tag = tag_map.shape[:2]
        # print(geo_map.shape)
        # print(conf_map.shape)
        # print(tag_map.shape)
        tag_map = torch.LongTensor(tag_map)
        one_hot = torch.zeros(h_tag, w_tag, self.nclass).scatter_(-1, tag_map, 1)

        conf_map = torch.Tensor(conf_map)
        geo_map = torch.Tensor(geo_map)

        img = torch.Tensor(img).permute(2, 0, 1)
        return transform(img), torch.Tensor(geo_map).permute(2,0,1), conf_map.permute(2,0,1), one_hot.permute(2,0,1)


def tag_change(gt_files, img_files, save_path):
    with open(gt_files, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    dict, nclass = dataset_unit.str_Converter_init()
    img = cv2.imread(img_files, cv2.IMREAD_COLOR)
    bboxes, tags = extract_vertices(lines, dict, nclass)

    img, bboxes = resize_img(img, bboxes)
    geo_map, conf_map, tag_map = get_score_geo(img, bboxes, tags, nclass, scale=0.25, length=512)

    save_np = np.concatenate((geo_map, conf_map, tag_map), 2)
    print(save_np.shape)
    np.save(save_path, gt_np)


if __name__ == "__main__":
    img_path = '/data/wanqi/dataset/data_set/img'
    gt_path = '/data/wanqi/dataset/data_set/gt'

    img_files = [os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
    gt_files = [os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]


    time_start_all = time.time()
    for i in range(len(img_files)):
        print(i)
        print(img_files[i])
        img_id = os.path.basename(img_files[i]).strip('.JPG').strip('.jpg')
        print(img_id)
        print("/data/wanqi/dataset/data_set/gt_np/" + img_id + ".npy")
        # with open(gt_files[i], 'r', encoding='utf-8-sig') as f:
        #    lines = f.readlines()
        # dict, nclass = dataset_unit.str_Converter_init()
        # img = cv2.imread(img_files[i], cv2.IMREAD_COLOR)
        # bboxes, tags = extract_vertices(lines, dict)

        # img, bboxes = resize_img(img, bboxes)
        # geo_map, conf_map, tag_map = get_score_geo(img, bboxes, tags, nclass, scale=0.25, length=512)

        # save_np = np.concatenate((geo_map, conf_map, tag_map), 2)
        # print(save_np.shape)

        gt_np = np.load("/data/wanqi/dataset/data_set/gt_np/img_{}.npy".format(i))

        np.save("/data/wanqi/dataset/data_set/gt_np_right/" + img_id + ".npy", gt_np)

        time_end = time.time()
        time_cost = (int)(time_end - time_start_all)
        after_s = time_end - time_start_all - time_cost
        s = time_cost % 60
        m = time_cost // 60 % 60
        h = time_cost // 60 // 60
        print('totally cost:', h, "h", m, "min", s + after_s, "s")

        time_need = (int)((len(img_files) - i) * (time_end - time_start_all) / (i+0.00001))
        after_s = (len(img_files) - i) * (time_end - time_start_all) / (i+0.00001) - time_need
        s = time_need % 60
        m = time_need // 60 % 60
        h = time_need // 60 // 60
        print('Remaining time:', h, "h", m, "min", s + after_s, "s")
