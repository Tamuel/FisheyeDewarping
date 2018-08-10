import sys, errno
import cv_utils as cu
import numpy as np
import os.path
import cv2
from os import listdir
from os.path import isfile, join
import threading
import time
import math

X = 0
Y = 1
Z = 2


class WarpUI:

    def __init__(self, window_name, image_path):
        self.image_index = 0
        self.N_WARP = 3

        self.selected_division = 0
        self.selected_warp = 0

        self.symmetric_mode = False
        self.grid_mode = False
        self.perspective_mode = True
        self.sym_select = 0

        self.DIVISION_ROW = 1
        self.DIVISION_COL = 1
        self.DIVISION_SIZE = self.DIVISION_ROW * self.DIVISION_COL

        self.window_name = window_name

        self.image = None
        self.grid_image = None
        self.canvas = np.zeros((1150, 2000, 3), np.uint8)
        self.number_of_grid = 10
        self.grid_points_per_line = 300
        self.img_size = np.array([1920, 1920])
        self.img_plane_size = np.array([1000, 1000])
        self.img_margin = 23
        self.img_division_size = np.array(
            [int(self.img_size[0] / self.DIVISION_COL), int(self.img_size[1] / self.DIVISION_ROW)]
        )
        self.warp_buttons = []
        self.warp_select_buttons = []
        self.pos_buttons = []
        self.lmouse_pressed = False
        self.video_play = False

        self.warps = []
        self.warped_img = []

        self.target = []
        self.sym_target = []
        self.grid_points = []

        # Load image
        self.img_path = image_path
        self.image_files = listdir(self.img_path)
        self.img_save_path = "dewarpedImage/" + image_path
        self.load_image()

        # Draw grid image
        r = self.img_plane_size[X] / 2.0 - self.img_margin
        for g in range(self.number_of_grid):
            k = (g + 1) * 0.1
            temp_pts = []
            for i in range(4):
                temp_pts.append([])
            # (x - w/2)^2 + k(y - h/2)^2 = r
            for pt in range(self.grid_points_per_line + 1):
                x = (r * 2) / self.grid_points_per_line * pt
                y = math.sqrt((pow(r, 2.0) - pow(x - (r * 2) / 2.0, 2.0)) * k) + (r * 2) / 2.0
                temp_pts[0].append((self.img_margin + x, self.img_margin + y))
                temp_pts[1].append((self.img_margin + y, self.img_margin + x))
                temp_pts[2].append((self.img_margin + x, self.img_margin + (r * 2) - y))
                temp_pts[3].append((self.img_margin + (r * 2) - y, self.img_margin + x))
            for i in range(4):
                temp_pts[i] = np.array(temp_pts[i], np.int32)
                temp_pts[i] = temp_pts[i].reshape((-1, 1, 2))
                self.grid_points.append(temp_pts[i])

        self.grid_image = np.array(self.image)
        self.draw_grid(self.grid_image)

        # Make Warp objects
        for d in range(self.DIVISION_SIZE):
            self.warps.append([])
            for w in range(self.N_WARP):
                self.warps[d].append(cu.Warp(self.img_division_size[X], self.img_division_size[Y]))
                file_name = 'warp_attr' + str(d) + '_' + str(w) + '.npy'
                if os.path.isfile(file_name):
                    self.warps[d][w].set_attr_array(np.load(file_name))

        # Set target division
        for w in range(self.N_WARP):
            self.target.append(self.warps[0][w])
            self.sym_target.append(self.warps[0][w])

        warp = self.warps[0][0]

        # Draw buttons at canvas
        for i in range(warp.CDC.size):
            off = 0
            if i >= warp.D.size:
                off = 1
            if i >= warp.C.size + warp.D.size:
                off = 2
            if i >= warp.D.size + warp.C.size + warp.Cnew.size:
                off = 3

            button = []
            for w in range(self.N_WARP):
                button.append(
                    cu.Button(
                        canvas=self.canvas,
                        left_top=(50 * i, 50 * w),
                        size=(50, 50),
                        color=(i * 2 + 20 * off + 30 * w, i * 5 + 20 * off + 20 * w, i * 7 + 30 * off + 5 * w * w)
                    )
                )
                button[w].draw()

            self.warp_buttons.append(button)

        self.save_button = cu.Button(
            canvas=self.canvas,
            left_top=(2000 - 50, 0),
            size=(50, 50),
            color=(50, 50, 255),
            text='Save'
        )
        self.save_button.set_function(self.save_warp_attr)
        self.save_button.draw()

        self.symmetry_button = cu.Button(
            canvas=self.canvas,
            size=(50, 50),
            color=(200, 50, 10),
            text='Sym'
        )
        self.symmetry_button.locate_left(self.save_button)
        self.symmetry_button.set_function(self.toggle_symmetric)
        self.symmetry_button.toggle = True
        self.symmetry_button.draw()

        self.grid_button = cu.Button(
            canvas=self.canvas,
            size=(50, 50),
            color=(50, 200, 10),
            text='Grid'
        )
        self.grid_button.locate_left(self.symmetry_button)
        self.grid_button.set_function(self.toggle_grid)
        self.grid_button.toggle = True
        self.grid_button.draw()

        self.perspective_button = cu.Button(
            canvas=self.canvas,
            size=(50, 50),
            color=(100, 100, 100),
            text='Persp'
        )
        self.perspective_button.locate_left(self.grid_button)
        self.perspective_button.set_function(self.toggle_perspective)
        self.perspective_button.toggle = True
        self.perspective_button.draw()

        self.play_pause_button = cu.Button(
            canvas=self.canvas,
            size=(50, 50),
            color=(100, 100, 100),
            text='||'
        )
        self.play_pause_button.locate_left(self.perspective_button)
        self.play_pause_button.set_function(self.toggle_play_pause)
        self.play_pause_button.toggle = True
        self.play_pause_button.draw()

        self.position_button = cu.Button(
            canvas=self.canvas,
            left_top=(
                self.canvas.shape[1] - 150,
                100
            ),
            size=(150, 50),
            color=(0, 0, 0),
            text='0/0'
        )
        self.grid_button.draw()

        self.next_button = cu.Button(
            canvas=self.canvas,
            size=(50, 50),
            color=(150, 150, 150),
            text='>'
        )
        self.next_button.locate_left(self.position_button)
        self.next_button.set_function(self.load_next_file)
        self.next_button.draw()

        self.prev_button = cu.Button(
            canvas=self.canvas,
            size=(50, 50),
            color=(150, 150, 150),
            text='<'
        )
        self.prev_button.locate_left(self.next_button)
        self.prev_button.set_function(self.load_prev_file)
        self.prev_button.draw()

        # Make buttons
        for w in range(self.N_WARP):
            cam_pos_button = cu.Button(
                canvas=self.canvas,
                left_top=(self.img_plane_size[1] + int(warp.CDC[6] - 10), int(warp.CDC[9]) + 150),
                size=(10, 10),
                color=(0, 255, 0)
            )

            new_cam_pos_button = cu.Button(
                canvas=self.canvas,
                left_top=(self.img_plane_size[1] + int(warp.CDC[15]), int(warp.CDC[18]) + 150),
                size=(10, 10),
                color=(255, 0, 0)
            )

            self.pos_buttons.append((cam_pos_button, new_cam_pos_button))

            warp_select_button = cu.Button(
                canvas=self.canvas,
                left_top=(50 * (warp.D.size + warp.C.size + warp.Cnew.size + 1), 50 * w),
                size=(50, 50),
                color=(125, 125, 0),
                text=str(w + 1)
            )
            warp_select_button.toggle = True
            warp_select_button.draw()

            self.warp_select_buttons.append(warp_select_button)

            self.warp_select_buttons[0].press()
            self.warp_select_buttons[0].release()

        # Draw number on buttons
        self.refresh_buttons_number()


        # # Load video
        # self.video_path = "D:/Samsung Heavy Industry Dataset/2017-10-28_12-00_PORTx6.mkv"
        # self.video_capture = cv2.VideoCapture(self.video_path)

        self.perspective_pts1 = np.float32([[40, 450], [150, 900], [960, 450], [850, 900]])
        self.perspective_pts2 = np.float32([[0, 300], [300, 800], [1000, 300], [700, 800]])
        self.perspective_pts3 = np.float32([[0, 500], [0, 1000], [1000, 500], [1000, 1000]])
        self.perspective_pts4 = np.float32([[0, 500], [300, 1000], [1000, 500], [700, 1000]])

        # Make dewarp image
        for i in range(0, self.DIVISION_SIZE):
            self.warped_img.append(None)
            self.draw_warped_image(i)

        # Draw red box at selected division
        self.draw_division_box(self.selected_division)

        # fps = 120  # int(self.video_capture.get(cv2.CAP_PROP_FPS))
        # video_length = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        # width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # end = False
        # print(fps, video_length, width, height)
        # while not end:
        #     frame_number = 0
        #     self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #     while True:
        #         frame_number += 1
        #         print(frame_number, '/', video_length)
        #         ret, frame = self.video_capture.read()
        #         if ret is False:
        #             break
        #
        #         dewarped = frame
        #
        #         dewarped = self.dewarp_image(dewarped, self.warps[0], self.img_size, 1.92)
        #
        #         division_pos = self.division_pos(0)
        #         cu.copy_to(
        #             cv2.resize(dewarped, tuple(self.img_plane_size), cv2.INTER_LINEAR),
        #             self.canvas,
        #             division_pos
        #         )
        #
        #         cu.copy_to(self.resize_img(frame), self.canvas, [0, 150])
        #         cv2.imshow(self.window_name, self.canvas)
        #
        #         # 50 fps.
        #         while not self.video_play:
        #             k = cv2.waitKey(int(20))
        #             if k == 27:
        #                 cv2.destroyAllWindows()
        #                 end = True
        #                 break
        #         if end:
        #             break
        #
        #         k = cv2.waitKey(int(1000 / fps))
        #         if k == 27:
        #             cv2.destroyAllWindows()
        #             end = True
        #             break
        #         if 48 <= k <= 57:
        #             self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, int((k - 48) * (video_length / 10)))
        #             frame_number = int((k - 48) * (video_length / 10))
        # k = cv2.waitKey()

        # Draw and show image
        cu.show_image(
            content=self.canvas,
            window_name=self.window_name,
            option=cv2.WINDOW_AUTOSIZE,
            mouse_callback=self.mouse_callback
        )

        # Save dewarped image
        for i in range(self.DIVISION_SIZE):
            row = int(i / self.DIVISION_COL)
            col = i % self.DIVISION_COL
            cu.copy_to(self.warped_img[i], self.image, (self.img_division_size[X] * col, self.img_division_size[Y] * row))

        cv2.imwrite(self.img_save_path, self.image)

        self.threads_kill = True
        # self.video_capture.release()
        cv2.destroyAllWindows()

    def convert_images(self, src_path, dst_path):
        file_paths = listdir(src_path)
        n_files = len(file_paths)
        for idx, file_name in enumerate(file_paths):
            file_extension = file_name.split('.')[-1]
            # If image file
            if file_extension in ('jpg', 'JPG', 'jpeg', 'JPEG', 'tif', 'TIF', 'png', 'PNG', 'bmp', 'BMP'):
                src_ = src_path + '/' + file_name
                dst_ = dst_path + '/' + file_name
                img = np.array(cv2.imread(src_, cv2.IMREAD_COLOR))
                dewarped_img = self.dewarp_image(
                    img,
                    self.warps[0],
                    tuple(self.img_size),
                    self.img_size[0] / self.img_plane_size[0]
                )
                cv2.imwrite(dst_, dewarped_img)
                print(' [%d/%d]' % (idx + 1, n_files))

    def convert_image(self, src_path, dst_path):
        # If image file
        img = np.array(cv2.imread(src_path, cv2.IMREAD_COLOR))
        dewarped_img = self.dewarp_image(
            img,
            self.warps[0],
            tuple(self.img_size),
            self.img_size[0]/self.img_plane_size[0]
        )
        cv2.imwrite(dst_path, dewarped_img)
        print('\'%s\' Converted' % src_path.split('/')[-1])

    def load_next_file(self):
        success = False
        while not success:
            self.image_index += 1
            if self.image_index == len(self.image_files):
                self.image_index = 0
            success = self.load_image()

    def load_prev_file(self):
        success = False
        while not success:
            self.image_index -= 1
            if self.image_index == -1:
                self.image_index = len(self.image_files) - 1
            success = self.load_image()

    def load_image(self):
        path = self.img_path + self.image_files[self.image_index]
        self.image = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.image is None:
            print('Cannot read image \'' + path + '\'')
            return False
        print('Load ', self.image_files[self.image_index])
        cu.copy_to(self.resize_img(self.image), self.canvas, (0, 150))
        return True

    def resize_img(self, img):
        return cv2.resize(img, tuple(self.img_plane_size), cv2.INTER_LINEAR)

    def toggle_symmetric(self):
        self.symmetric_mode = not self.symmetric_mode

    def toggle_grid(self):
        self.grid_mode = not self.grid_mode
        if self.grid_mode:
            grid_image = np.array(self.resize_img(self.image))
            self.draw_grid(grid_image)
            cu.copy_to(src_image=grid_image, dst_image=self.canvas, position=(0, 150))
        else:
            cu.copy_to(src_image=self.resize_img(self.image), dst_image=self.canvas, position=(0, 150))

    def toggle_perspective(self):
        self.perspective_mode = not self.perspective_mode

    def toggle_play_pause(self):
        self.video_play = not self.video_play
        if self.video_play:
            self.play_pause_button.text = '||'
        else:
            self.play_pause_button.text = '>'

    def save_warp_attr(self):
        print('Save warps attributes')
        for d in range(self.DIVISION_SIZE):
            for w in range(self.N_WARP):
                np.save('warp_attr' + str(d) + '_' + str(w) + '.npy', self.warps[d][w].CDC)

    def refresh_warped_image(self, except_img_idx=-1):
        for i in range(self.DIVISION_SIZE):
            if i is not except_img_idx:
                row = int(i / self.DIVISION_COL)
                col = i % self.DIVISION_COL

                cu.copy_to(
                    cv2.resize(
                        self.warped_img[i],
                        self.img_division_size * (self.img_size / self.img_plane_size),
                        cv2.INTER_LINEAR
                    ),
                    self.canvas,
                    (self.img_plane_size[Y] + int(self.img_division_size[X] * col),
                     150 + int(self.img_division_size[Y] * row))
                )

    def refresh_buttons_number(self):
        for w in range(self.N_WARP):
            for i in range(self.warps[self.selected_division][w].CDC.size):
                self.warp_buttons[i][w].draw(str(self.warps[self.selected_division][w].CDC[i]))

    def division_pos(self, index):
        return (self.img_plane_size[Y] + int(self.img_division_size[X] * (index % self.DIVISION_COL)),
                150 + int(self.img_division_size[Y] * int(index / self.DIVISION_COL)))

    def symmetric_pos(self, index, horizontal=True, vertical=True):
        result = 0
        row = int(index / self.DIVISION_COL)
        col = index % self.DIVISION_COL
        if horizontal and vertical:
            mid = int(self.DIVISION_SIZE / 2)
            result = mid + (mid - self.selected_division)
        elif horizontal and not vertical:
            result = row * self.DIVISION_COL + ((self.DIVISION_COL - 1) - col)
        elif not horizontal and vertical:
            result = (self.DIVISION_ROW - row) * self.DIVISION_COL + col

        return result

    def draw_division_box(self, index, color=(0, 0, 255)):
        chk_pos = self.division_pos(index)
        # Draw red box at selected division
        cv2.rectangle(
            self.canvas,
            (chk_pos[0] + 1, chk_pos[1] + 1),
            (chk_pos[0] + self.img_division_size[X] - 1, chk_pos[1] + self.img_division_size[Y] - 1),
            color,
            1
        )

    def draw_grid(self, image):
        row = 0
        for pts in self.grid_points:
            cv2.polylines(
                img=image,
                pts=[pts],
                isClosed=False,
                color=(0, 0, 255) if
                int(row / 4) == 7
                else (0, 0, 0),
                thickness=1,
                lineType=8
            )
            row += 1

    def dewarp_image(self, image, warp, img_size, ratio=1.0):
        dewarped = image

        for j in range(self.N_WARP):
            # if self.grid_mode and j is self.selected_warp:
            #     self.draw_grid(dewarped)
            dewarped = warp[j].dewarp(
                dewarped,
                tuple(img_size),
                ratio
            )

        if self.perspective_mode:
            M = cv2.getPerspectiveTransform(
                self.perspective_pts1 * ratio,
                self.perspective_pts2 * ratio
            )
            dewarped = cv2.warpPerspective(dewarped, M, tuple(img_size))

        return dewarped

    def draw_warped_image(self, index):
        self.refresh_buttons_number()

        # print("Warp arguments=============================================")
        # for i in range(self.N_WARP):
        #     self.target[i].print()

        dewarped = self.image

        if self.grid_mode:
            dewarped = np.array(self.grid_image)

        dewarped = self.dewarp_image(
            dewarped,
            self.warps[index],
            self.img_size,
            ratio=self.img_size[0]/self.img_plane_size[0]
        )

        row = int(index / self.DIVISION_COL)
        col = index % self.DIVISION_COL

        # Divide image
        x = int(self.img_division_size[X] * col)
        y = int(self.img_division_size[Y] * row)
        width = self.img_division_size[X]
        height = self.img_division_size[Y]
        self.warped_img[index] = dewarped[y:y + height, x:x + width]

        division_pos = self.division_pos(index)
        cu.copy_to(
            cv2.resize(
                self.warped_img[index],
                tuple((self.img_division_size * (self.img_plane_size / self.img_size)).astype(np.int32)),
                cv2.INTER_LINEAR
            ),
            self.canvas,
            division_pos
        )

        warp = self.warps[self.selected_division][self.selected_warp]

        self.pos_buttons[self.selected_warp][0].left_top = \
            (self.img_plane_size[1] + int(warp.CDC[6] - 10),
             int(warp.CDC[9] - 10) + 150)

        self.pos_buttons[self.selected_warp][0].right_bottom = \
            (self.img_plane_size[1] + int(warp.CDC[6] + 10),
             int(warp.CDC[9] + 10) + 150)

        self.pos_buttons[self.selected_warp][1].left_top = \
            (self.img_plane_size[1] + int(warp.CDC[15] - 7),
             int(warp.CDC[18] - 7) + 150)

        self.pos_buttons[self.selected_warp][1].right_bottom = \
            (self.img_plane_size[1] + int(warp.CDC[15] + 7),
             int(warp.CDC[18] + 7) + 150)

        self.pos_buttons[self.selected_warp][0].draw()
        self.pos_buttons[self.selected_warp][1].draw()

    def mouse_callback(self, event, x, y, flags, param):
        redraw = False
        chk = False

        # print("Mouse callback :", event, x, y, flags, param)

        if event == cv2.EVENT_MOUSEMOVE:
            redraw = False
            self.position_button.draw(
                text='x:' + str(x - self.img_plane_size[X]) + '/y:' + str(y - 150)
            )
            cv2.imshow(self.window_name, self.canvas)
            cv2.waitKey(1) & 0xFF

        elif event == cv2.EVENT_LBUTTONDOWN:
            self.lmouse_pressed = True
            if self.save_button.contain(x, y):
                self.save_button.press()
                redraw = True
                chk = True
            elif self.symmetry_button.contain(x, y):
                self.symmetry_button.press()
                redraw = True
                chk = True
            elif self.perspective_button.contain(x, y):
                self.perspective_button.press()
                redraw = True
                chk = True
            elif self.play_pause_button.contain(x, y):
                self.play_pause_button.press()
                redraw = True
                chk = True
            elif self.grid_button.contain(x, y):
                self.grid_button.press()
                redraw = True
                chk = True
            elif self.next_button.contain(x, y):
                self.next_button.press()
                redraw = True
                chk = True
            elif self.prev_button.contain(x, y):
                self.prev_button.press()
                redraw = True
                chk = True
            else:
                for w in range(self.N_WARP):
                    if self.warp_select_buttons[w].contain(x, y):
                        self.warp_select_buttons[w].press()
                        redraw = True
                        chk = True
                        break

        elif event == cv2.EVENT_LBUTTONUP:
            self.lmouse_pressed = False
            if self.save_button.pressed:
                self.save_button.release()
                redraw = True
            elif self.symmetry_button.pressed:
                self.symmetry_button.release()
                self.sym_select = self.symmetric_pos(
                    index=self.selected_division,
                    horizontal=True,
                    vertical=False
                )
                if self.symmetric_mode:
                    for w in range(self.N_WARP):
                        self.sym_target[w] = self.warps[self.sym_select][w]
                redraw = True
            elif self.grid_button.pressed:
                self.grid_button.release()
                redraw = True
            elif self.perspective_button.pressed:
                self.perspective_button.release()
                redraw = True
            elif self.play_pause_button.pressed:
                self.play_pause_button.release()
                redraw = True
            elif self.prev_button.pressed:
                self.prev_button.release()
                redraw = True
            elif self.next_button.pressed:
                self.next_button.release()
                redraw = True
            else:
                for w in range(self.N_WARP):
                    if self.warp_select_buttons[w].pressed:
                        if not w == self.selected_warp:
                            self.warp_select_buttons[w].release()
                            self.selected_warp = w
                            for i in range(self.N_WARP):
                                if i != w and self.warp_select_buttons[i].on:
                                    self.warp_select_buttons[i].release()
                            redraw = True
                            break
                        else:
                            chk = True
                            for i in range(self.N_WARP):
                                if i != w and self.warp_select_buttons[i].on:
                                    self.warp_select_buttons[i].release()
                            break

                if not chk:
                    for w in range(self.N_WARP):
                        if self.pos_buttons[w][0].pressed:
                            self.pos_buttons[w][0].release()
                            redraw = True
                            chk = True
                            break
                        elif self.pos_buttons[w][1].pressed:
                            self.pos_buttons[w][1].release()
                            redraw = True
                            chk = True
                            break

                if not chk:
                    for i in range(self.DIVISION_SIZE):
                        row = int(i / self.DIVISION_COL)
                        col = i % self.DIVISION_COL
                        img_x = self.img_plane_size[X] + int(self.img_division_size[X] * col)
                        img_y = 150 + int(self.img_division_size[Y] * row)
                        if img_x <= x <= img_x + self.img_division_size[X] and img_y <= y <= img_y + self.img_division_size[Y]:
                            if i is not self.selected_division:
                                self.selected_division = i
                                self.sym_select = self.symmetric_pos(
                                    index=self.selected_division,
                                    horizontal=True,
                                    vertical=False
                                )
                                redraw = True
                                for w in range(self.N_WARP):
                                    self.target[w] = self.warps[self.selected_division][w]
                                    if self.symmetric_mode:
                                        self.sym_target[w] = self.warps[self.sym_select][w]
                            chk = True
                            break

        elif event == cv2.EVENT_RBUTTONUP:
            for w in range(self.N_WARP):
                for i in range(23):
                    if self.warp_buttons[i][w].contain(x, y):
                        self.target[w].set_attr(i, self.target[w].CDdefault[i])
                        if self.symmetric_mode and self.selected_division is not self.sym_select:
                            self.sym_target[w].set_attr(i, self.sym_target[w].CDdefault[i])
                        chk = True
                        break
                if chk:
                    break

            if not chk:
                for w in range(self.N_WARP):
                    if self.warp_select_buttons[w].contain(x, y):
                        for j in range(23):
                            self.target[w].set_attr(j, self.target[w].CDdefault[j])
                            chk = True
                            if self.symmetric_mode and self.selected_division is not self.sym_select:
                                self.sym_target[w].set_attr(j, self.sym_target[w].CDdefault[j])
                        break

            if chk:
                redraw = True

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                wheel_delta = 10.0 if self.lmouse_pressed else 1
            else:
                wheel_delta = -10.0 if self.lmouse_pressed else -1

            chk = False
            for w in range(self.N_WARP):
                for i in range(23):
                    if self.warp_buttons[i][w].contain(x, y):
                        self.target[w].set_attr(i, self.target[w].CDC[i] + wheel_delta)
                        if self.symmetric_mode and self.selected_division is not self.sym_select:
                            if i is 6 or i is 9 or i is 15 or i is 18 or i is 22:
                                self.sym_target[w].set_attr(i, self.sym_target[w].CDC[i] - wheel_delta)
                            else:
                                self.sym_target[w].set_attr(i, self.sym_target[w].CDC[i] + wheel_delta)
                        chk = True
                        break
                if chk:
                    break

            if chk:
                redraw = True

        if redraw:
            self.refresh_buttons_number()
            if not self.video_play:
                self.refresh_warped_image(self.selected_division)
                self.draw_warped_image(self.selected_division)
                if self.symmetric_mode and self.selected_division is not self.sym_select:
                    self.draw_warped_image(self.sym_select)
                    self.draw_division_box(self.sym_select, (0, 255, 0))
                self.draw_division_box(self.selected_division)
                cv2.imshow(self.window_name, self.canvas)
                cv2.waitKey(1) & 0xFF

