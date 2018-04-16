import sys, errno
import cv_utils as cu
import numpy as np
import os.path
import cv2
import threading
import time

X = 0
Y = 1
Z = 2


class WarpUI:

    def __init__(self, window_name, image_path):
        self.N_WARP = 3

        self.selected_division = 0
        self.selected_warp = 0

        self.symmetric_mode = False
        self.grid_mode = False
        self.sym_select = 0

        self.DIVISION_ROW = 3
        self.DIVISION_COL = 3
        self.DIVISION_SIZE = self.DIVISION_ROW * self.DIVISION_COL

        self.window_name = window_name

        self.image = None
        self.canvas = np.zeros((1150, 2000, 3), np.uint8)
        self.number_of_grid = 10
        self.img_size = (1000, 1000)
        self.img_division_size = (int(self.img_size[0] / self.DIVISION_COL), int(self.img_size[1] / self.DIVISION_ROW))
        self.warp_buttons = []
        self.warp_select_buttons = []
        self.pos_buttons = []
        self.lmouse_pressed = False

        self.warps = []
        self.warped_img = []

        self.target = []
        self.sym_target = []
        self.grid_points = []

        # self.grid_img = None
        # self.check_grid_line_draw = []
        # self.threads = []
        # self.threads_kill = False
        # for i in range(self.number_of_grid):
        #     self.check_grid_line_draw.append(False)
        #     self.threads.append(threading.Thread(target=self.draw_grid_line, args=(i,)))
        #     self.threads[i].start()

        # Load image
        self.img_path = "fisheyeImage/" + image_path
        self.img_save_path = "dewarpedImage/" + image_path
        self.image = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        if self.image is None:
            print('Cannot read image \'' + self.img_path + '\'')
            sys.exit(errno.EFAULT)
        self.image = cv2.resize(self.image, self.img_size, interpolation=cv2.INTER_LINEAR)
        cu.copy_to(self.image, self.canvas, (0, 150))

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
        self.symmetry_button.rect.locate_left(self.save_button.rect)
        self.symmetry_button.set_function(self.toggle_symmetric)
        self.symmetry_button.toggle = True
        self.symmetry_button.draw()

        self.grid_button = cu.Button(
            canvas=self.canvas,
            size=(50, 50),
            color=(50, 200, 10),
            text='Grid'
        )
        self.grid_button.rect.locate_left(self.symmetry_button.rect)
        self.grid_button.set_function(self.toggle_grid)
        self.grid_button.toggle = True
        self.grid_button.draw()

        # Make buttons
        for w in range(self.N_WARP):
            cam_pos_button = cu.Button(
                canvas=self.canvas,
                left_top=(self.img_size[1] + int(warp.CDC[6] - 10), int(warp.CDC[9]) + 150),
                size=(10, 10),
                color=(0, 255, 0)
            )

            new_cam_pos_button = cu.Button(
                canvas=self.canvas,
                left_top=(self.img_size[1] + int(warp.CDC[15]), int(warp.CDC[18]) + 150),
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

        # Make dewarp image
        for i in range(0, self.DIVISION_SIZE):
            self.warped_img.append(None)
            self.draw_warped_image(i)

        # Draw number on buttons
        self.refresh_buttons_number()

        # Draw red box at selected division
        self.draw_division_box(self.selected_division)

        # Draw and show image
        cu.show_image(
            content=self.canvas,
            window_name=self.window_name,
            option=cv2.WINDOW_AUTOSIZE,
            mouse_callback=self.mouse_callback
        )
        key = cv2.waitKey(0) & 0xFF  # Because of 64 bit OS

        # Save dewarped image
        for i in range(self.DIVISION_SIZE):
            row = int(i / self.DIVISION_COL)
            col = i % self.DIVISION_COL
            cu.copy_to(self.warped_img[i], self.image, (self.img_division_size[X] * col, self.img_division_size[Y] * row))

        self.threads_kill = True

        cv2.imwrite(self.img_save_path, self.image)

        cv2.destroyAllWindows()

    def toggle_symmetric(self):
        self.symmetric_mode = not self.symmetric_mode

    def toggle_grid(self):
        self.grid_mode = not self.grid_mode

    def save_warp_attr(self):
        print('Save warps attributes')
        for d in range(self.DIVISION_SIZE):
            for w in range(self.N_WARP):
                np.save('warp_attr' + str(d) + '_' + str(w) + '.npy', self.warps[d][w].CDC)

    def refresh_warped_image(self, except_img_idx = -1):
        for i in range(self.DIVISION_SIZE):
            if i is not except_img_idx:
                row = int(i / self.DIVISION_COL)
                col = i % self.DIVISION_COL

                cu.copy_to(
                    self.warped_img[i],
                    self.canvas,
                    (self.img_size[Y] + int(self.img_division_size[X] * col),
                     150 + int(self.img_division_size[Y] * row))
                )

    def refresh_buttons_number(self):
        for w in range(self.N_WARP):
            for i in range(self.warps[self.selected_division][w].CDC.size):
                self.warp_buttons[i][w].draw(str(self.warps[self.selected_division][w].CDC[i]))

    def division_pos(self, index):
        return (self.img_size[Y] + int(self.img_division_size[X] * (index % self.DIVISION_COL)),
                150 + int(self.img_division_size[Y] * int(index / self.DIVISION_COL)))

    def symmetric_pos(self, index):
        mid = int(self.DIVISION_SIZE / 2)
        return mid + (mid - self.selected_division)

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

    # def draw_grid_line(self, index):
    #     while not self.threads_kill:
    #         time.sleep(0.0001)
    #         if self.check_grid_line_draw[index] is False and self.grid_img is not None:
    #             # Draw grid
    #             rows, cols, colors = self.grid_img.shape
    #             for p in range(rows):
    #                 pos = index * int(rows / self.number_of_grid)
    #                 self.grid_img[pos][p] = tuple(255 - c for c in self.grid_img[pos][p])
    #             for p in range(cols):
    #                 pos = index * int(cols / self.number_of_grid)
    #                 self.grid_img[p][pos] = tuple(255 - c for c in self.grid_img[p][pos])
    #             self.check_grid_line_draw[index] = True
    #
    # def wait_draw_grid_line(self):
    #     check = self.number_of_grid
    #     while check is not 0:
    #         check = self.number_of_grid
    #         for i in range(self.number_of_grid):
    #             if self.check_grid_line_draw[i] is True:
    #                 check -= 1

    def draw_grid(self, image):
        # self.grid_img = np.array(image)
        # wait = threading.Thread(target=self.wait_draw_grid_line)
        # wait.start()
        # wait.join()
        # result = np.array(self.grid_img)
        # self.grid_img = None
        # for i in range(self.number_of_grid):
        #     self.check_grid_line_draw[i] = False
        rows, cols, colors = image.shape
        for i in range(self.number_of_grid):
            cv2.polylines(
                img=image,
                pts=self.grid_points,
                isClosed=False,
                color=(200, 200, 200)
            )
            cv2.line(
                img=image,
                pt1=(0, int(rows / self.number_of_grid * i)),
                pt2=(cols, int(rows / self.number_of_grid * i)),
                thickness=1,
                color=(200, 200, 200)
            )
            cv2.line(
                img=image,
                pt1=(int(cols / self.number_of_grid * i), 0),
                pt2=(int(cols / self.number_of_grid * i), rows),
                thickness=1,
                color=(200, 200, 200)
            )

    def draw_warped_image(self, index):
        self.refresh_buttons_number()

        print("Warp arguments=============================================")
        for i in range(self.N_WARP):
            self.target[i].print()

        dewarped = self.image
        for j in range(self.N_WARP):
            if self.grid_mode and j is self.selected_warp:
                self.draw_grid(dewarped)
            dewarped = self.warps[index][j].dewarp(
                dewarped,
                self.img_size
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
            self.warped_img[index],
            self.canvas,
            division_pos
        )

        warp = self.warps[self.selected_division][self.selected_warp]

        self.pos_buttons[self.selected_warp][0].rect.left_top = \
            (self.img_size[1] + int(warp.CDC[6] - 10),
             int(warp.CDC[9] - 10) + 150)

        self.pos_buttons[self.selected_warp][0].rect.right_bottom = \
            (self.img_size[1] + int(warp.CDC[6] + 10),
             int(warp.CDC[9] + 10) + 150)

        self.pos_buttons[self.selected_warp][1].rect.left_top = \
            (self.img_size[1] + int(warp.CDC[15] - 7),
             int(warp.CDC[18] - 7) + 150)

        self.pos_buttons[self.selected_warp][1].rect.right_bottom = \
            (self.img_size[1] + int(warp.CDC[15] + 7),
             int(warp.CDC[18] + 7) + 150)

        self.pos_buttons[self.selected_warp][0].draw()
        self.pos_buttons[self.selected_warp][1].draw()

    def mouse_callback(self, event, x, y, flags, param):
        redraw = False
        chk = False

        # print("Mouse callback :", event, x, y, flags, param)

        if event == cv2.EVENT_MOUSEMOVE:
            redraw = False

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
            elif self.grid_button.contain(x, y):
                self.grid_button.press()
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
                self.sym_select = self.symmetric_pos(self.selected_division)
                if self.symmetric_mode:
                    for w in range(self.N_WARP):
                        self.sym_target[w] = self.warps[self.sym_select][w]
                redraw = True
            elif self.grid_button.pressed:
                self.grid_button.release()
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
                        img_x = self.img_size[X] + int(self.img_division_size[X] * col)
                        img_y = 150 + int(self.img_division_size[Y] * row)
                        if img_x <= x <= img_x + self.img_division_size[X] and img_y <= y <= img_y + self.img_division_size[Y]:
                            if i is not self.selected_division:
                                self.selected_division = i
                                self.sym_select = self.symmetric_pos(self.selected_division)
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
                wheel_delta = 10.0 if self.lmouse_pressed else 1.0
            else:
                wheel_delta = -10.0 if self.lmouse_pressed else -1.0

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
            self.refresh_warped_image(self.selected_division)
            self.draw_warped_image(self.selected_division)
            if self.symmetric_mode and self.selected_division is not self.sym_select:
                self.draw_warped_image(self.sym_select)
                self.draw_division_box(self.sym_select, (0, 255, 0))
            self.draw_division_box(self.selected_division)
            redraw = False
            cv2.imshow(self.window_name, self.canvas)
            cv2.waitKey(1) & 0xFF

