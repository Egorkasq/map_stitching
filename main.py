import numpy as np
import os
import glob
import cv2



def take_center_of_image(image):
    """
    Из-за особенностей снимаемой линзы, на краях изображений образуется наибольшее искревление, поэтому
    необходимо искать особые точки только в центральной части изображения
    """
    image = image[int(image.shape[0] * 0.25): int(image.shape[0] * 0.75),
            int(image.shape[1] * 0.25): int(image.shape[1] * 0.75)]
    return image


def calculate_matches() -> list:
    pass


def change_size_of_image(image, scale_percent):
    """
    Когда изображения слишком большого размера из необходдимо слегка уменьшить,
    так как разрешение слишком большое и количество точек, которые воспринимается как уникальные много,
    и многие ошибочно воспринмаются как пары
    :param image:
    :param scale_percent: степень сжатия (10 %, 20% ...)
    :return: bvtymityyjt bpj,hf;tybt
    """
    if scale_percent == 100:
        return image
    else:
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resize_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resize_image


def create_panorama(image_path, result_path, expansion, scale_percent=100, gaussian_blur=0):
    """
    :param scale_percent: степень сжатия
    :param image_path: folder with images
    :param expansion: expansion of the searched files
    :param scale_percent: used for reduce original images (helpful, when ultimate map is big)
    :param gaussian_blur: used for keypoint detect (use 3, 6, 9, 11 ....) полезно иногда размыть изображение
    перед сшитием, не понимаю как, но иногда этот вариант срабатывает, когда зона перекрытия небольшая

    :return:
    """
    list_of_images = os.path.join(image_path, expansion)
    list_of_images = glob.glob(list_of_images)
    if len(list_of_images) == 0:
        print('Can not find {} files in folder {}. Check it.'.format(expansion, image_path))

    list_of_images.sort()
    print('find {} images for stitching:{}'.format(len(list_of_images), list_of_images))

    # Базовое изолбражение, относительно него выстраивается общая карта
    base_image = cv2.imread(list_of_images[0], 0)
    base_image = cv2.GaussianBlur(base_image, (gaussian_blur, gaussian_blur), 0)

    base_image = change_size_of_image(base_image, scale_percent)
    list_of_images.pop(0)

    for num, file in enumerate(list_of_images):
        next_image = cv2.imread(file, 0)
        next_image = change_size_of_image(next_image, scale_percent)
        next_image = cv2.GaussianBlur(next_image, (gaussian_blur, gaussian_blur), 0)

        # объявление дескрипора (используется для поиска особых точкек)
        # orb = cv2.SIFT_create()
        # orb = cv2.ORB_create()
        orb = cv2.AKAZE_create()

        kp1, des1 = orb.detectAndCompute(base_image, None)
        kp2, des2 = orb.detectAndCompute(next_image, None)

        # исключение "выбросов"
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        # Сортировка пар точек по совместимости, чем ближе
        # к началу, тем более вероятно, что точки образуют пару
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 100:
            matches = matches[:int(len(matches) * 0.2)]  # Берем только 20% лучших

            '''cv2.imshow('12',
                       cv2.resize(cv2.drawMatches(base_image, kp1, next_image, kp2, matches, base_image), (1000, 500)))
            cv2.waitKey(0)'''

        if len(matches) < 5:
            print('not enought points for stitching. Process Stopping on {} images. Restarting from {}... '.
                  format(num, file))

            print('problem file is {}'.format(file))
            cv2.imwrite('{}/{}.JPG'.format(result_path, num), base_image)
            base_image = cv2.imread(list_of_images[num + 1], 0)
            base_image = change_size_of_image(base_image, scale_percent)  # restart map stitch from num image

        else:
            dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h1, w1 = base_image.shape[:2]
            h2, w2 = next_image.shape[:2]
            pts1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            pts2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

            # расчет матрицы поворота
            pts2 = cv2.perspectiveTransform(pts2, M)
            pts = np.concatenate((pts1, pts2), axis=0)
            [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
            [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
            t = [-xmin, -ymin]
            Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

            # изменение перспективы изображения
            align_image = cv2.warpPerspective(next_image, Ht.dot(M), (xmax - xmin, ymax - ymin))
            new_base_image = np.zeros((align_image.shape[0], align_image.shape[1]), np.uint8)
            new_base_image[t[1]:h1 + t[1], t[0]:w1 + t[0]] = base_image
            # черные грани заполняются информацией с базового изображения
            align_image = np.where(align_image == 0, new_base_image, align_image)

            # image_mask = take_image_without_black_line(base_image)
            # result[t[1]:h1 + t[1], t[0]:w1 + t[0]] = image_mask[1]
            # cv2.imshow('align_image', cv2.resize(align_image, (1000, 500)))
            # cv2.waitKey()
            base_image = align_image

    # print('map created {}'.format(align_image.shape[:2]))
    return base_image


if __name__ == '__main__':
    img_path = './map_stitching/images'
    res_path = './map_stitching/results'
    img_path = os.path.join(img_path)
    print(img_path)
    img = create_panorama(img_path, res_path, '*.JPG', scale_percent=30, gaussian_blur=7)

    # cv2.imwrite('{}'.format('123.JPG'), img)
