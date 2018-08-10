import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from os import listdir


def is_image(file_name):
    file_extension = file_name.split('.')[-1]
    if file_extension in ('jpg', 'JPG', 'jpeg', 'JPEG', 'tif', 'TIF', 'png', 'PNG', 'bmp', 'BMP'):
        return True
    else:
        return False


# https://docs.opencv.org/3.3.0/d5/df0/group__ximgproc__segmentation.html#ga5e3e721c5f16e34d3ad52b9eeb6d2860
class Segmentator:
    def __init__(
            self,
            contrast_alpha=1.5,
            contrast_beta=-100,
            mean_filtering_sp=7,
            mean_filtering_sr=25,
            mean_filtering_level=0,
            segmentation_sigma=0.8,
            segmentation_k=600,
            segmentation_min=1
    ):
        self.contrast_alpha = contrast_alpha
        self.contrast_beta = contrast_beta

        self.mean_filtering_sp = mean_filtering_sp
        self.mean_filtering_sr = mean_filtering_sr
        self.mean_filtering_level = mean_filtering_level

        self.segmentation_sigma = segmentation_sigma
        self.segmentation_k = segmentation_k
        self.segmentation_min = segmentation_min

        cv2.setUseOptimized(True)
        cv2.setNumThreads(16)

    def folder_segmentation(self, folder_path, result_path=None, start=0, show_result=False):
        result = list()
        files = listdir(folder_path)
        n_files = len(files)
        for idx, file in enumerate(files):
            if idx < start:
                continue
            if is_image(file):
                print('%s Completed [%d/%d]' % (file, n_files, idx + 1))
                result.append(
                    self.file_segmentation(
                        img_path=folder_path + '/' + file,
                        show_result=show_result
                    )
                )
                if result_path is not None:
                    if not os.path.exists(result_path):
                        os.makedirs(result_path)
                    plt.imsave(result_path + '/' + file, result[-1])

        return result

    def file_segmentation(self, img_path, show_result=False):
        img = np.array(
            plt.imread(img_path)[1504:1604, 380:1920 - 380, :]
        )
        return self.segmentation(
            img=img,
            img_name=img_path.split('/')[-1],
            show_result=show_result
        )

    def segmentation(self, img, img_name='Original', show_result=False):
        # Remove boundary data
        img_mask = np.array(img >= 20).astype(dtype=np.uint8)
        img_h = img.shape[0]
        img_w = img.shape[1]

        # Adjust contrast of image
        alpha = self.contrast_alpha
        beta = self.contrast_beta
        img_contrast = img * alpha + beta
        img_contrast = np.clip(img_contrast, 0, 255).astype(dtype=np.uint8)

        # Apply mean filtering to image
        mean_filtering = cv2.pyrMeanShiftFiltering(
            src=img_contrast,  # Original image
            sp=self.mean_filtering_sp,  # Size of spatial window
            sr=self.mean_filtering_sr,  # Size of color window
            maxLevel=self.mean_filtering_level  # Pyramid level
        )

        # Do graph segmentation at image
        ss = cv2.ximgproc.segmentation.createGraphSegmentation(
            sigma=self.segmentation_sigma,  # Smoothness
            k=self.segmentation_k,  # Neighbor regions
            min_size=self.segmentation_min  # Minimum segment size
        )
        segmented = ss.processImage(mean_filtering)
        segmented = np.repeat(
            np.reshape(segmented, [img_h, img_w, 1]),
            repeats=3,
            axis=2
        ).astype(dtype=np.float32)
        segmented = np.multiply(segmented, img_mask)

        # Select interest region
        interest_point = (int(img_w / 2), img_h - 4)
        selected_region = np.array(
            segmented == segmented[interest_point[1]][interest_point[0]]
        ).astype(dtype=np.float32)

        # Fill holes
        fill = np.zeros((img_h + 2, img_w + 2), dtype=np.uint8)
        cv2.floodFill(
            image=selected_region,
            mask=fill,
            seedPoint=(0, 0),
            newVal=0
        )
        fill = (fill == 0).astype(dtype=np.uint8)[1:-1, 1:-1]

        if show_result:
            # Show results =============================================
            # Merge original image and interest region for compare
            merge = (img_contrast * 0.5 + np.repeat(
                np.reshape(
                    fill * 125,
                    (img_h, img_w, 1)
                ),
                repeats=3,
                axis=2
            )).astype(dtype=np.uint8)

            images = [
                img,
                img_contrast,
                mean_filtering,
                segmented,
                selected_region,
                fill,
                merge
            ]
            titles = [
                img_name,
                'Contrast',
                'Mean filtering',
                'Segmented',
                'Extracted',
                'Fill',
                'Merge'
            ]

            for idx, (img, title) in enumerate(zip(images, titles)):
                plt.subplot(len(images), 1, idx + 1)
                plt.imshow(img)
                plt.title(title)
                plt.xticks([])
                plt.yticks([])

            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            plt.show()
            # ===========================================================

        return fill
