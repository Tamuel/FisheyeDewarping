import numpy as np
import warpingUI as wUI
import check_db as cDB
from matplotlib import pyplot as plt
from cv_segmentation import Segmentator
from os import listdir
import os


def rename(src_path, start_index):
    file_paths = listdir(src_path)
    n_files = len(file_paths)
    index = start_index
    for idx, file_name in enumerate(file_paths):
        file_extension = file_name.split('.')[-1]
        # If image file
        if file_extension in ('jpg', 'JPG', 'jpeg', 'JPEG', 'tif', 'TIF', 'png', 'PNG', 'bmp', 'BMP'):
            os.rename(
                src_path + '/' + file_name,
                src_path + '/' + str(index) + '.' + file_extension
            )
            print('\'%s\' to \'%s\' Renamed [%d/%d]' % (file_name, str(index), idx + 1, n_files))
            index += 1


def copy(src, dir, end):
    start = int(src.split('/')[-1].split('.')[0]) + 1
    img = plt.imread(src)
    for i in range(start, end + 1):
        plt.imsave(
            dir + '/' + str(i).zfill(4) + '.png',
            img
        )


def copy_fix_file(check, ori, seg, out):
    ori_files = listdir(ori)
    check = np.loadtxt(
        fname=check,
        dtype=np.float32
    )
    for i in range(len(check)):
        print(check[i])
    for idx, f in enumerate(ori_files):
        if check[idx][1] == 1:
            ori_img = plt.imread(ori + '/' + f)
            seg_img = plt.imread(seg + '/' + f.split('.')[-2] + '.png')
            plt.imsave(
                fname=out + '/' + f,
                arr=ori_img
            )
            plt.imsave(
                fname=out + '/' + f.split('.')[-2] + '.png',
                arr=seg_img
            )


if __name__ == "__main__":
    # copy(
    #     src='C:/Users/YDK/Desktop/YDK/Graduate School/Work/Samsung Heavy Industries/DB/LifeBoat/1993.png',
    #     dir='C:/Users/YDK/Desktop/YDK/Graduate School/Work/Samsung Heavy Industries/DB/LifeBoat/',
    #     end=1999
    # )
    UI = wUI.WarpUI(
        window_name="Fisheye",
        image_path="D:/Samsung Heavy Industry Dataset/1. targets/1. target - vessels/"
    )
    # UI.convert_image(
    #     src_path='./fisheyeImage/sample_01.jpg',
    #     dst_path='./fisheyeImage/sample_01_convert.jpg'
    # )
    # UI = cDB.DbChecker(
    #     original_path='C:/Users/YDK/Desktop/YDK/Graduate School/Work/Samsung Heavy Industries/BridgeWingSegmentation/real_image',
    #     segmentation_path='C:/Users/YDK/Desktop/YDK/Graduate School/Work/Samsung Heavy Industries/BridgeWingSegmentation/segmentation_label3',
    #     start_index=0
    # )
    # copy_fix_file(
    #     check='./check.txt',
    #     ori='./result',
    #     seg='./segmentation',
    #     out='./need_to_fix'
    # )
    # seg = Segmentator(
    #     contrast_alpha=2.0,
    #     contrast_beta=-150,
    #     mean_filtering_sp=3,
    #     mean_filtering_sr=30,
    #     mean_filtering_level=0,
    #     segmentation_sigma=0.6,
    #     segmentation_k=750,
    #     segmentation_min=1
    # )
    #
    # results = seg.folder_segmentation(
    #     folder_path='./result',
    #     result_path='./segmented',
    #     start=0,
    #     show_result=False
    # )
