import cv_utils as cu
import cv2
import numpy as np
from os import listdir
from scipy.misc import imresize
from matplotlib import pyplot as plt


class DbChecker:

    def __init__(self, original_path, segmentation_path, start_index=0):
        # Paths
        self.original_path = original_path
        self.segmentation_path = segmentation_path

        # Files
        self.original_files = listdir(original_path)
        self.segmentation_files = listdir(segmentation_path)

        self.file_index = start_index
        self.seg_ratio = 0.3

        # Check
        self.check_seg = np.zeros([len(self.segmentation_files), 2], np.float32)
        for i in range(len(self.segmentation_files)):
            self.check_seg[i][0] = i + 1
        try:
            self.load_check()
            print('Load \"check.txt\"')
        except FileNotFoundError:
            self.save_check()
            print('Make new \"check.txt\"')

        # GUI
        self.canvas = np.zeros((1300, 1350, 3), np.uint8)
        self.image = None
        self.segmentation = None
        self.image_size = (1920, 1920)
        self.image_plane_size = (1300, 1300)

        self.next_button = cu.Button(
            canvas=self.canvas,
            left_top=(self.image_plane_size[0], 0),
            size=(50, 50),
            color=(150, 150, 150),
            text=' > '
        )
        self.next_button.set_function(self.load_next_image)
        self.next_button.draw()

        self.prev_button = cu.Button(
            canvas=self.canvas,
            size=(50, 50),
            color=(150, 150, 150),
            text=' < '
        )
        self.prev_button.locate_bottom(self.next_button)
        self.prev_button.set_function(self.load_prev_image)
        self.prev_button.draw()

        self.check_button = cu.Button(
            canvas=self.canvas,
            size=(50, 50),
            color=(150, 150, 150),
            text='0'
        )
        self.check_button.locate_bottom(self.prev_button)
        self.check_button.set_function(self.check_toggle)
        self.check_button.draw()

        self.load_image()

        # Show GUI
        cu.show_image(
            content=self.canvas,
            window_name='DB Checker',
            option=cv2.WINDOW_AUTOSIZE,
            mouse_callback=self.mouse_callback
        )

        # Key input
        while True:
            k = cv2.waitKeyEx(int(20))
            if k == 27:  # Esc
                cv2.destroyAllWindows()
                self.save_check()
                break
            elif k == 2424832:  # Left
                self.prev_button.press()
                self.prev_button.release()
            elif k == 2555904:  # Right
                self.next_button.press()
                self.next_button.release()

    def check_toggle(self):
        self.check_seg[self.file_index][1] = int(not self.check_seg[self.file_index][1])
        self.check_button.draw(
            text=str(self.check_seg[self.file_index][1])
        )

    def load_check(self):
        self.check_seg = np.loadtxt(
            fname='./check.txt',
            dtype=np.uint8
        )

    def save_check(self):
        np.savetxt(
            fname='./check.txt',
            X=self.check_seg
        )

    def load_prev_image(self):
        self.file_index -= 1
        if self.file_index < 0:
            self.file_index = len(self.original_files) - 1
        self.load_image()
        self.save_check()

    def load_next_image(self):
        self.file_index += 1
        if self.file_index >= len(self.original_files):
            self.file_index = 0
        self.load_image()
        self.save_check()

    def load_image(self):
        self.image = cv2.imread(
            self.original_path + '/' + self.original_files[self.file_index]
        )
        self.segmentation = cv2.imread(
            self.segmentation_path + '/' + self.segmentation_files[self.file_index]
        )
        augment_img = cv2.resize(
            src=self.image * (1 - self.seg_ratio) + self.segmentation * self.seg_ratio,
            dsize=self.image_plane_size
        )
        # augment_img = self.image * 0.7 + self.segmentation * 0.3
        self.check_button.draw(
            text=str(self.check_seg[self.file_index][1])
        )

        cu.copy_to(
            src_image=augment_img,
            dst_image=self.canvas,
            position=(0, 0)
        )
        print(self.segmentation_files[self.file_index])
        cv2.imshow('DB Checker', self.canvas)

    def mouse_callback(self, event, x, y, flags, param):
        redraw = False

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.next_button.contain(x, y):
                self.next_button.press()
                redraw = True
            elif self.prev_button.contain(x, y):
                self.prev_button.press()
                redraw = True
            elif self.check_button.contain(x, y):
                self.check_button.press()
                redraw = True

        elif event == cv2.EVENT_LBUTTONUP:
            if self.next_button.pressed:
                self.next_button.release()
                redraw = True
            elif self.prev_button.pressed:
                self.prev_button.release()
                redraw = True
            elif self.check_button.pressed:
                self.check_button.release()
                redraw = True

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                wheel_delta = 1
            else:
                wheel_delta = -1
            print('Wheel', wheel_delta)
            self.seg_ratio += wheel_delta * 0.1
            if self.seg_ratio > 1.0:
                self.seg_ratio = 1.0
            elif self.seg_ratio < 0:
                self.seg_ratio = 0
            self.load_image()

            redraw = True

        if redraw:
            cv2.imshow('DB Checker', self.canvas)
