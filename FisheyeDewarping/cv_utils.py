import cv2
import errno, sys
import numpy as np
import math

X = 0
Y = 1
Z = 2


# Convert BGR opencv image to RGB image
def to_rgb(image):
    b, g, r = cv2.split(image)
    return cv2.merge([r, g, b])


def show_image(content, window_name="Image", option=cv2.WINDOW_NORMAL, mouse_callback=None):
    cv2.namedWindow(window_name, option)
    if mouse_callback is not None:
        cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, content)


def copy_to(src_image, dst_image, position):
    shape = src_image.shape
    dst_image[
        position[1]:shape[0] + position[1],
        position[0]:shape[1] + position[0],
        :
    ] = src_image[:, :, :]


# Precision need to be integer, it means below numbers of 10 ^ precision will gone
def floor(src, precision):
    temp = round(src / math.pow(10, precision))
    temp = float(temp) * math.pow(10, precision)
    return temp


class Rect:
    def __init__(self, left_top=(0, 0), size=(0, 0), color=(0, 0, 0)):
        self.left_top = left_top
        self.size = size
        self.right_bottom = (self.left_top[X] + self.size[X], self.left_top[Y] + self.size[Y])
        self.color = color  # B, G, R

        if self.left_top[0] < 0 or self.left_top[1] < 0 or self.right_bottom[0] < 0 or self.right_bottom[1] < 0:
            print('Position of rectangle must be positive value')
            exit(errno.EFAULT)

    def contain(self, x, y):
        if self.left_top[X] <= x <= self.right_bottom[X] and self.left_top[Y] <= y <= self.right_bottom[Y]:
            return True
        else:
            return False

    def locate_left(self, rect):
        self.left_top = (rect.left_top[X] - self.size[X], rect.left_top[Y])
        self.right_bottom = (self.left_top[X] + self.size[X], self.left_top[Y] + self.size[Y])

    def locate_right(self, rect):
        self.left_top = (rect.right_bottom[X], rect.left_top[Y])
        self.right_bottom = (self.left_top[X] + self.size[X], self.left_top[Y] + self.size[Y])


class Warp:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.CDdefault = np.array(
            [0, 0, 0, 0,
             500, 0, 500, 0, 500, 500, 0.0, 0.0, 1.0,
             500, 0, 500, 0, 500, 500, 0.0, 0.0, 1.0,
             0.0],
            np.float32
        )
        self.CDC = np.array(
            self.CDdefault,
            np.float32
        )

        self.C = np.array(self.CDdefault[4:13], np.float32)
        self.D = np.array(self.CDdefault[0:4], np.float32)
        self.R = np.identity(3)
        self.Cnew = np.array(self.CDdefault[13:22], np.float32)
        self.rotate = self.CDdefault[22]
        self.map1 = np.zeros([3, 3])
        self.map2 = np.zeros([3, 3])

    def set_attr(self, i, value):
        self.CDC[i] = value
        if 0 <= i < 4:
            self.D = np.array(self.CDC[0:4], np.float32)
        if 4 <= i < 13:
            self.C = np.array(self.CDC[4:13], np.float32)
        if 13 <= i < 22:
            self.Cnew = np.array(self.CDC[13:22], np.float32)
        if i == 22:
            self.rotate = self.CDC[22]

    def set_attr_array(self, array):
        self.CDC = array
        self.D = np.array(self.CDC[0:4], np.float32)
        self.C = np.array(self.CDC[4:13], np.float32)
        self.Cnew = np.array(self.CDC[13:22], np.float32)
        self.rotate = self.CDC[22]

    def dewarp(self, src_image, size):
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.C.reshape([3, 3]),
            np.multiply(self.D, 0.001).reshape([4, 1]),
            self.R,
            self.Cnew.reshape([3, 3]),
            size, cv2.CV_16SC2
        )

        dewarped = cv2.remap(
            src_image, self.map1, self.map2,
            cv2.INTER_LINEAR
        )

        rows, cols, colors = dewarped.shape

        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), self.rotate, 1)

        return cv2.warpAffine(dewarped, m, (cols, rows))

    def print(self):
        print(self.D, self.C, self.Cnew, self.rotate)


class Button:

    def __init__(self, canvas, left_top=(0, 0), size=(0, 0), color=(0, 0, 0), text=""):
        self.canvas = canvas
        self.rect = Rect(left_top, size, color)
        self.text = text
        self.toggle = False
        self.on = False
        self.pressed = False
        self.released = True
        self.function_ptr = None

    def draw(self, text=None, color=None):
        cv2.rectangle(
            self.canvas,
            self.rect.left_top,
            self.rect.right_bottom,
            self.rect.color if color is None else color,
            -1
        )
        cv2.putText(
            self.canvas,
            self.text if text is None else text,
            (self.rect.left_top[X] + 5, self.rect.right_bottom[Y] - int(self.rect.size[Y] / 2)),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (255, 255, 255),
            1
        )

    def set_function(self, function_ptr):
        self.function_ptr = function_ptr

    def press(self):
        self.pressed = True
        self.released = False
        self.draw(None, tuple(c / 2 for c in self.rect.color))

    def release(self, parameters=None):
        if self.toggle:
            self.on = not self.on

        if self.function_ptr is not None:
            if parameters is not None:
                self.function_ptr(parameters)
            else:
                self.function_ptr()

        if self.toggle and self.on:
            self.draw(None, tuple(c / 2 for c in self.rect.color))
        elif self.toggle and not self.on:
            self.draw()
        elif self.pressed:
            self.draw()

        self.pressed = False
        self.released = True

    def contain(self, x, y):
        return self.rect.contain(x, y)

    def print(self):
        print(self.rect.left_top, self.rect.right_bottom, self.text)
